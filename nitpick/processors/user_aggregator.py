"""The user aggregator.

Collects what the user said during a turn and hands it to the LLM.

This processor is the reason nitpick bundles the transcript into the
``VAD_USER_STOPPED_SPEAKING`` frame. The aggregator triggers the LLM when
the turn closes, and it reads the transcript **from that frame** — so it
cannot possibly run before the text arrives, no matter how the lanes
interleave.
"""
from __future__ import annotations

import logging

from ..context import LLMContext
from ..frames import (
    DOWNSTREAM,
    Frame,
    FrameKind,
    FrameProcessor,
    Transcription,
    VadStopped,
)

log = logging.getLogger(__name__)


class UserAggregator(FrameProcessor):
    """Turns finished user turns into an LLM trigger."""

    def __init__(
        self,
        context: LLMContext,
        *,
        name: str = "UserAggregator",
    ) -> None:
        super().__init__(name, _UserAggregatorHandler(self))
        self.context = context
        self._pending = ""
        self._turn_open = False

    @property
    def pending_text(self) -> str:
        """Text collected so far for the open turn."""
        return self._pending

    async def _open_turn(self) -> None:
        self._turn_open = True
        self._pending = ""

    async def _close_turn(self, stop: VadStopped) -> None:
        """Close the turn and push the user's text downstream."""
        transcript = stop.transcript
        text = transcript.text.strip() if transcript and transcript.text else ""
        text = text or self._pending.strip()
        self._pending = ""
        self._turn_open = False

        if not text:
            log.debug("User turn closed with no transcript — not triggering the LLM")
            return

        await self.context.add_user_message(text)
        # A context frame is the trigger for the LLM service.
        await self.push_frame(
            Frame(FrameKind.LLM_CONTEXT, self.context), DOWNSTREAM
        )


class _UserAggregatorHandler:
    def __init__(self, aggregator: UserAggregator) -> None:
        self._aggregator = aggregator

    async def on_process_frame(
        self, processor: FrameProcessor, frame: Frame, direction: str
    ) -> None:
        aggregator = self._aggregator
        kind = frame.kind

        if kind is FrameKind.VAD_USER_STARTED_SPEAKING:
            await aggregator._open_turn()
            await processor.push_frame(frame, direction)
            return

        if kind is FrameKind.VAD_USER_STOPPED_SPEAKING:
            payload = frame.payload
            stop = payload if isinstance(payload, VadStopped) else VadStopped()
            await aggregator._close_turn(stop)
            # The stop frame is consumed — it must not travel further, or
            # downstream processors would see the turn close twice.
            return

        if kind is FrameKind.TRANSCRIPTION:
            payload = frame.payload
            if isinstance(payload, Transcription) and payload.finalized:
                aggregator._pending = f"{aggregator._pending} {payload.text}".strip()
            # Interim transcripts are forwarded untouched so the UI can show
            # live captions, but never aggregated.
            await processor.push_frame(frame, direction)
            return

        await processor.push_frame(frame, direction)
