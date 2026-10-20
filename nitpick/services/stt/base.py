"""Speech-to-text: the provider protocol and the pipeline service.

Design
------
``SttProvider`` is a **wire-protocol adapter** and nothing more. It answers
four questions:

1. What PCM format do I want? (:attr:`SttProvider.sample_rate`)
2. How do I start a session? (:meth:`SttProvider.start`)
3. How do I hand over a chunk of PCM? (:meth:`SttProvider.push_audio`)
4. How do I ask for a final transcript? (:meth:`SttProvider.flush`)

Everything else — turn detection, buffering, the frame plumbing, the
ordering guarantee — lives in :class:`SttService`. Adding a provider is
therefore a protocol exercise, not a concurrency one.

The ordering guarantee
----------------------
A voice pipeline has two frame lanes: system frames (which include "the user
stopped speaking") overtake data frames (which include transcripts). Emitting
the transcript and the stop as two separate frames means the stop can arrive
first — and an agent that sees a turn close with no text **never answers**.

:class:`SttService` fixes this structurally: when the user stops speaking,
the stop frame is **stashed** until the provider delivers a final transcript,
and then a single frame carrying both is emitted downstream. One frame cannot
be reordered against itself.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from ...frames import (
    DOWNSTREAM,
    Frame,
    FrameHandler,
    FrameKind,
    FrameProcessor,
    Transcription,
    VadStopped,
    frame_transcription,
    frame_vad_stopped,
)

log = logging.getLogger(__name__)

EventCallback = Callable[["SttEvent"], Awaitable[None] | None]


@dataclass
class SttEvent:
    """Something the provider reported."""

    kind: str
    text: str = ""
    language: str | None = None
    is_final: bool = False
    audio_ms: float | None = None
    error: str | None = None

    @property
    def is_partial(self) -> bool:
        return self.kind == "partial"


class SttProvider:
    """Base class for speech-to-text providers.

    Subclasses override the four protocol methods. ``start`` receives a
    callback that the provider uses to report events.
    """

    #: Short identifier, used in logs and billing.
    name: str = "base"
    #: PCM sample rate the provider wants.
    sample_rate: int = 16_000

    async def start(self, on_event: EventCallback) -> None:
        """Open the session. ``on_event`` is how results are reported."""

    async def push_audio(self, pcm: bytes) -> None:
        """Hand over a chunk of PCM at :attr:`sample_rate`."""

    async def flush(self) -> None:
        """Ask for a final transcript for the current segment."""

    async def stop(self) -> None:
        """Close the session."""


class SttService(FrameProcessor):
    """Drives an :class:`SttProvider` inside a pipeline.

    Responsibilities:

    * accumulate PCM and forward it to the provider
    * watch for the "user stopped speaking" signal and stash that frame
    * release a single stop frame carrying the transcript once the provider
      delivers it
    * emit intermediate (partial) transcripts as they arrive
    """

    def __init__(
        self,
        provider: SttProvider,
        *,
        name: str | None = None,
        release_timeout: float = 5.0,
        emit_partials: bool = True,
    ) -> None:
        super().__init__(name or f"SttService({provider.name})", _SttHandler(self))
        self.provider = provider
        self.release_timeout = release_timeout
        self.emit_partials = emit_partials

        self._pending_stop: VadStopped | None = None
        self._pending_transcript: str | None = None
        self._release_timer: asyncio.Task | None = None
        # NB: distinct from ``FrameProcessor._started``, which is the
        # framework lifecycle flag. This one tracks the *provider session*.
        self._provider_started = False
        self._buffer = bytearray()
        self._turn_active = False

    # ------------------------------------------------------------------
    # Provider lifecycle
    # ------------------------------------------------------------------

    async def _provider_start(self) -> None:
        if self._provider_started:
            return
        self._provider_started = True
        await self.provider.start(self._on_provider_event)

    async def _provider_stop(self) -> None:
        if not self._provider_started:
            return
        self._provider_started = False
        await self.provider.stop()

    async def _on_provider_event(self, event: SttEvent) -> None:
        """Handle an event reported by the provider."""
        if event.kind == "error":
            log.warning("STT provider %s error: %s", self.provider.name, event.error)
            await self.push_error(f"STT error: {event.error}", fatal=False)
            return

        if event.kind == "partial":
            if not self.emit_partials:
                return
            transcription = Transcription(
                text=event.text,
                language=event.language,
                finalized=False,
            )
            await self.push_frame(frame_transcription(transcription), DOWNSTREAM)
            return

        if event.kind == "final":
            transcription = Transcription(
                text=event.text,
                language=event.language,
                finalized=True,
            )
            await self._release_stop(transcription)

    # ------------------------------------------------------------------
    # Turn handling
    # ------------------------------------------------------------------

    async def _release_stop(self, transcription: Transcription | None) -> None:
        """Emit the stashed stop frame, carrying the transcript."""
        stop = self._pending_stop
        if stop is None:
            # A final arrived with no open turn — surface it as a transcript
            # so it is not lost.
            if transcription is not None and transcription.text:
                await self.push_frame(frame_transcription(transcription), DOWNSTREAM)
            return

        self._pending_stop = None
        self._cancel_release_timer()

        if transcription is not None:
            stop.transcript = transcription
        await self.push_frame(frame_vad_stopped(stop.stop_secs, stop.timestamp, stop.transcript), DOWNSTREAM)

    def _cancel_release_timer(self) -> None:
        if self._release_timer is not None:
            self._release_timer.cancel()
            self._release_timer = None

    def _arm_release_timer(self) -> None:
        """Release the stop frame even if the provider never answers."""
        self._cancel_release_timer()

        async def _release() -> None:
            try:
                await asyncio.sleep(self.release_timeout)
            except asyncio.CancelledError:
                return
            log.warning(
                "STT provider %s produced no final within %ss — releasing stop",
                self.provider.name,
                self.release_timeout,
            )
            await self._release_stop(None)

        self._release_timer = asyncio.create_task(_release())


class _SttHandler:
    """The :class:`FrameHandler` for :class:`SttService`."""

    def __init__(self, service: SttService) -> None:
        self._service = service

    async def on_process_frame(
        self, processor: FrameProcessor, frame: Frame, direction: str
    ) -> None:
        service = self._service

        if frame.kind is FrameKind.START:
            await service._provider_start()
            await processor.push_frame(frame, direction)
            return

        if frame.kind is FrameKind.CANCEL:
            await service._provider_stop()
            await processor.push_frame(frame, direction)
            return

        if frame.kind is FrameKind.INPUT_AUDIO_RAW:
            chunk = frame.payload
            if chunk is not None and getattr(service.provider, "audio_enabled", True):
                await service.provider.push_audio(chunk.data)
            await processor.push_frame(frame, direction)
            return

        if frame.kind is FrameKind.VAD_USER_STOPPED_SPEAKING:
            # The user stopped speaking. Stash this frame — we will re-emit it
            # once the provider delivers the final transcript.
            payload = frame.payload
            stop = payload if isinstance(payload, VadStopped) else VadStopped()
            service._pending_stop = stop
            service._arm_release_timer()
            await service.provider.flush()
            return

        await processor.push_frame(frame, direction)
