"""Barge-in handling.

When the user starts speaking while the bot is still talking, the bot
should stop. Without this the two talk over each other, which is the single
most common complaint about voice agents.

The rule is simple: a ``VAD_USER_STARTED_SPEAKING`` frame that arrives
while the bot is speaking triggers an interruption. The interruption is
broadcast, so the LLM service aborts its in-flight request and the TTS
service drops whatever it had buffered.
"""
from __future__ import annotations

import logging

from ..frames import (
    Frame,
    FrameKind,
    FrameProcessor,
)

log = logging.getLogger(__name__)


class InterruptionHandler(FrameProcessor):
    """Turns a user interruption into a pipeline-wide interruption."""

    def __init__(
        self,
        *,
        name: str = "InterruptionHandler",
        enabled: bool = True,
    ) -> None:
        super().__init__(name, _InterruptionHandler(self))
        self.enabled = enabled
        self._bot_speaking = False
        self.interruption_count = 0

    @property
    def bot_speaking(self) -> bool:
        return self._bot_speaking

    async def _handle_bot_speaking(self, speaking: bool) -> None:
        self._bot_speaking = speaking

    async def _handle_user_started(self) -> None:
        if not self.enabled:
            return
        if not self._bot_speaking:
            return
        self.interruption_count += 1
        log.info("User interrupted the bot — cancelling the reply in flight")
        await self.broadcast_interruption()


class _InterruptionHandler:
    def __init__(self, handler: InterruptionHandler) -> None:
        self._handler = handler

    async def on_process_frame(
        self, processor: FrameProcessor, frame: Frame, direction: str
    ) -> None:
        handler = self._handler
        kind = frame.kind

        if kind is FrameKind.BOT_STARTED_SPEAKING:
            await handler._handle_bot_speaking(True)
            await processor.push_frame(frame, direction)
            return

        if kind is FrameKind.BOT_STOPPED_SPEAKING:
            await handler._handle_bot_speaking(False)
            await processor.push_frame(frame, direction)
            return

        if kind is FrameKind.VAD_USER_STARTED_SPEAKING:
            await handler._handle_user_started()
            await processor.push_frame(frame, direction)
            return

        if kind is FrameKind.INTERRUPTION:
            await handler._handle_bot_speaking(False)
            await processor.push_frame(frame, direction)
            return

        await processor.push_frame(frame, direction)
