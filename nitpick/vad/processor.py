"""The VAD processor — turns audio into turn-boundary frames.

Sits in the pipeline right after the transport and emits
``VAD_USER_STARTED_SPEAKING`` and ``VAD_USER_STOPPED_SPEAKING`` frames.
The STT service downstream watches for the stop and closes the turn.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

from ..frames import (
    DOWNSTREAM,
    AudioChunk,
    Frame,
    FrameKind,
    FrameProcessor,
    frame_vad_started,
    frame_vad_stopped,
)
from .core import StateMachine, VadParams, VadState

log = logging.getLogger(__name__)


class VadProcessor(FrameProcessor):
    """Detects the user's speaking turns from the audio stream."""

    def __init__(
        self,
        backend: object,
        *,
        params: VadParams | None = None,
        name: str = "VadProcessor",
        speaking_gate: object | None = None,
        on_speaking: Callable[[bool], Any] | None = None,
    ) -> None:
        super().__init__(name, _VadHandler(self))
        self.backend = backend
        self.params = params or VadParams()
        sample_rate = int(getattr(backend, "sample_rate", 16_000))
        self._machine = StateMachine(sample_rate, self.params)
        # NB: distinct from ``FrameProcessor._started``, which is the
        # framework lifecycle flag. This one tracks whether the *user* is
        # currently speaking.
        self._user_speaking = False
        #: Shared with the client-side VAD path so exactly one turn boundary
        #: is emitted per utterance.
        self.gate = speaking_gate
        #: Called with True/False when the user's speaking state changes.
        #: Used to publish the turn to the room.
        self.on_speaking = on_speaking

    @property
    def sample_rate(self) -> int:
        return self._machine.sample_rate

    @property
    def state(self) -> VadState:
        return self._machine.state

    # ------------------------------------------------------------------

    def _claim(self, speaking: bool) -> bool:
        if self.gate is None:
            return True
        if speaking:
            return bool(self.gate.claim_start())
        return bool(self.gate.claim_stop())

    async def _handle_audio(self, chunk: AudioChunk) -> None:
        """Feed a chunk through the VAD and emit any turn transitions."""
        if chunk.sample_rate != self.sample_rate:
            # The backend cannot use audio at another rate. Resampling here
            # would be the fix; for now, skip rather than analyse garbage.
            log.debug(
                "VAD skipping audio at %s Hz (backend expects %s Hz)",
                chunk.sample_rate,
                self.sample_rate,
            )
            return

        # Drain **every** complete window in the chunk, not just the first.
        # One call yields at most one window, so a chunk that carries more
        # audio than one window (a 20 ms browser frame is 320 samples
        # against a 512-sample window) would leave the remainder buffered
        # and the VAD would drift further and further behind the live
        # audio.
        pending = chunk.data
        while True:
            window = self._machine.next_window(pending)
            pending = b""
            if window is None:
                break

            confidence = float(self.backend.is_speech(window))
            state = self._machine.advance(confidence, window)

            if state is VadState.SPEAKING and not self._user_speaking:
                self._user_speaking = True
                if self._claim(True):
                    await self._notify(True)
                    await self.push_frame(
                        frame_vad_started(self.params.start_secs, time.time()), DOWNSTREAM
                    )
            elif state is VadState.QUIET and self._user_speaking:
                self._user_speaking = False
                if self._claim(False):
                    await self._notify(False)
                    await self.push_frame(
                        frame_vad_stopped(self.params.stop_secs, time.time()), DOWNSTREAM
                    )

    async def _notify(self, speaking: bool) -> None:
        """Tell the owner of this processor that the turn changed."""
        if self.on_speaking is None:
            return
        try:
            result = self.on_speaking(speaking)
            if asyncio.iscoroutine(result):
                await result
        except Exception:  # noqa: BLE001 - a listener must not break the VAD
            log.exception("Speaking listener failed")


class _VadHandler:
    def __init__(self, processor: VadProcessor) -> None:
        self._processor = processor

    async def on_process_frame(
        self, processor: FrameProcessor, frame: Frame, direction: str
    ) -> None:
        if frame.kind is FrameKind.INPUT_AUDIO_RAW:
            chunk = frame.payload
            if isinstance(chunk, AudioChunk):
                await self._processor._handle_audio(chunk)
            # The audio is analysed *and* forwarded — the STT downstream
            # needs the same samples the VAD just looked at.
            await processor.push_frame(frame, direction)
            return

        await processor.push_frame(frame, direction)
