"""Transport processors — the boundary between a pipeline and the outside world.

Two implementations ship with nitpick:

:class:`RoomTransport`
    Connects a pipeline to a room in a :class:`~nitpick.rooms.RoomManager`.
    Audio frames flow in through the room event bus; output audio is
    published back to the room.

:class:`ChannelTransport`
    Drives a pipeline through plain :class:`asyncio.Queue` objects. No
    server, no network — the fastest way to test a pipeline.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from ..frames import (
    DOWNSTREAM,
    AudioChunk,
    Frame,
    FrameHandler,
    FrameProcessor,
    frame_input_audio,
    frame_output_audio,
    frame_vad_started,
    frame_vad_stopped,
)
from ..rooms import RoomEventType, RoomManager
from ..rooms.model import Participant

log = logging.getLogger(__name__)


@dataclass
class TransportParams:
    """Tunables shared by transports."""

    audio_in_enabled: bool = True
    audio_out_enabled: bool = True
    #: Sample rate the pipeline expects on the input side.
    audio_in_sample_rate: int = 16_000
    #: Sample rate the pipeline produces on the output side.
    audio_out_sample_rate: int = 24_000
    audio_in_channels: int = 1
    metadata: dict[str, str] = field(default_factory=dict)


class RoomTransport:
    """Bridges a room participant to a pipeline.

    The transport is not a :class:`FrameProcessor` itself — it exposes two
    of them (:meth:`input` and :meth:`output`) so a pipeline can be assembled
    the same way regardless of where the audio comes from.
    """

    def __init__(
        self,
        manager: RoomManager,
        participant: Participant,
        *,
        params: TransportParams | None = None,
    ) -> None:
        self.manager = manager
        self.participant = participant
        self.params = params or TransportParams()
        self._input = FrameProcessor("RoomTransport.input", _RoomInput(self))
        self._output = FrameProcessor("RoomTransport.output", _RoomOutput(self))
        # Subscribe immediately. Waiting for the START frame would silently
        # drop any audio that arrives in the window between the agent
        # joining the room and the pipeline booting.
        self._subscribed = False
        self.start()

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def input(self) -> FrameProcessor:
        """The head of the pipeline."""
        return self._input

    def output(self) -> FrameProcessor:
        """The tail of the pipeline."""
        return self._output

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Subscribe to room events. Idempotent."""
        if self._subscribed:
            return
        self.manager.add_event_listener(self._on_room_event)
        self._subscribed = True

    def stop(self) -> None:
        """Unsubscribe from room events."""
        if not self._subscribed:
            return
        self.manager.remove_event_listener(self._on_room_event)
        self._subscribed = False

    # ------------------------------------------------------------------
    # Room events
    # ------------------------------------------------------------------

    def _on_room_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("event_type")

        if event_type == RoomEventType.PARTICIPANT_SPEAKING_STARTED.value:
            if event.get("participant_id") == self.participant.participant_id:
                return
            asyncio.create_task(
                self._input.push_frame(frame_vad_started(0.0, time.time()), DOWNSTREAM)
            )
            return

        if event_type == RoomEventType.PARTICIPANT_SPEAKING_STOPPED.value:
            if event.get("participant_id") == self.participant.participant_id:
                return
            asyncio.create_task(
                self._input.push_frame(frame_vad_stopped(0.0, time.time()), DOWNSTREAM)
            )
            return

        if event_type != RoomEventType.PARTICIPANT_AUDIO.value:
            return
        if event.get("participant_id") == self.participant.participant_id:
            # Don't listen to ourselves.
            return

        audio = event.get("audio")
        if not audio:
            return

        chunk = AudioChunk(
            data=audio,
            sample_rate=int(event.get("sample_rate") or 16_000),
            num_channels=int(event.get("channels") or 1),
        )
        asyncio.create_task(
            self._input.push_frame(frame_input_audio(chunk), DOWNSTREAM)
        )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    async def publish_audio(self, chunk: AudioChunk) -> None:
        """Publish a chunk of output audio to the room."""
        import base64

        if not self.params.audio_out_enabled:
            return
        room_id = self.participant.room_id
        if room_id is None:
            return
        payload = {
            "type": "room_audio",
            "room_id": room_id,
            "from_participant_id": self.participant.participant_id,
            "mime_type": "audio/pcm",
            "audio_base64": base64.b64encode(chunk.data).decode("ascii"),
            "audio_bytes": len(chunk.data),
            "created_at": asyncio.get_event_loop().time(),
        }
        await self.manager.broadcast_message_to_room(
            room_id=room_id,
            payload=payload,
            exclude_participant_id=self.participant.participant_id,
        )


class _RoomInput:
    """Pushes the ``START`` frame through and otherwise passes frames on."""

    def __init__(self, transport: RoomTransport) -> None:
        self._transport = transport

    async def on_process_frame(
        self, processor: FrameProcessor, frame: Frame, direction: str
    ) -> None:
        await processor.push_frame(frame, direction)


class _RoomOutput:
    """Publishes output audio frames to the room."""

    def __init__(self, transport: RoomTransport) -> None:
        self._transport = transport

    async def on_process_frame(
        self, processor: FrameProcessor, frame: Frame, direction: str
    ) -> None:
        if frame.kind.value == "output_audio_raw":
            payload = frame.payload
            if isinstance(payload, AudioChunk):
                await self._transport.publish_audio(payload)
        await processor.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# Channel transport (testing)
# ---------------------------------------------------------------------------


@dataclass
class ChannelMessage:
    """A message on a :class:`ChannelTransport` channel."""

    kind: str
    payload: Any = None


class ChannelTransport:
    """A transport backed by plain asyncio queues.

    Feed PCM in, assert on what comes out — no server required::

        transport = ChannelTransport(params)
        await transport.incoming.put(ChannelMessage("audio", pcm))
        frame = await transport.outgoing.get()
    """

    def __init__(
        self,
        params: TransportParams | None = None,
        *,
        incoming: asyncio.Queue | None = None,
        outgoing: asyncio.Queue | None = None,
    ) -> None:
        self.params = params or TransportParams()
        self.incoming = incoming or asyncio.Queue()
        self.outgoing = outgoing or asyncio.Queue()
        self._input = FrameProcessor("ChannelTransport.input", _ChannelInput(self))
        self._output = FrameProcessor("ChannelTransport.output", _ChannelOutput(self))

    def input(self) -> FrameProcessor:
        return self._input

    def output(self) -> FrameProcessor:
        return self._output

    async def push_audio(self, pcm: bytes, *, sample_rate: int | None = None) -> None:
        """Push a PCM frame into the pipeline."""
        chunk = AudioChunk(
            data=pcm,
            sample_rate=sample_rate or self.params.audio_in_sample_rate,
            num_channels=self.params.audio_in_channels,
        )
        await self._input.push_frame(frame_input_audio(chunk), DOWNSTREAM)


class _ChannelInput:
    def __init__(self, transport: ChannelTransport) -> None:
        self._transport = transport

    async def on_process_frame(
        self, processor: FrameProcessor, frame: Frame, direction: str
    ) -> None:
        await processor.push_frame(frame, direction)


class _ChannelOutput:
    def __init__(self, transport: ChannelTransport) -> None:
        self._transport = transport

    async def on_process_frame(
        self, processor: FrameProcessor, frame: Frame, direction: str
    ) -> None:
        if frame.kind.value == "output_audio_raw":
            await self._transport.outgoing.put(
                ChannelMessage("audio", frame.payload)
            )
        await processor.push_frame(frame, direction)
