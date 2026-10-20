"""The assistant aggregator.

Collects the LLM's streamed text into the shared context so the next turn
has the full conversation history.
"""
from __future__ import annotations

import logging

from ..context import LLMContext
from ..frames import (
    DOWNSTREAM,
    Frame,
    FrameHandler,
    FrameKind,
    FrameProcessor,
)

log = logging.getLogger(__name__)


class AssistantAggregator(FrameProcessor):
    """Accumulates assistant text into the conversation context."""

    def __init__(
        self,
        context: LLMContext,
        *,
        name: str = "AssistantAggregator",
    ) -> None:
        super().__init__(name, _AssistantAggregatorHandler(self))
        self.context = context
        self._parts: list[str] = []

    @property
    def current_text(self) -> str:
        """The text of the response currently being generated."""
        return "".join(self._parts)

    async def _finish(self) -> None:
        text = "".join(self._parts).strip()
        self._parts.clear()
        if not text:
            return
        # The LLM service records the assistant message itself when the turn
        # carried tool calls. Skip it here so the same reply is not written
        # to the context twice.
        messages = self.context.messages
        if messages and messages[-1].role == "assistant" and messages[-1].content == text:
            return
        await self.context.add_assistant_message(text)


class _AssistantAggregatorHandler:
    def __init__(self, aggregator: AssistantAggregator) -> None:
        self._aggregator = aggregator

    async def on_process_frame(
        self, processor: FrameProcessor, frame: Frame, direction: str
    ) -> None:
        aggregator = self._aggregator
        kind = frame.kind

        if kind is FrameKind.LLM_FULL_RESPONSE_START:
            aggregator._parts.clear()
            await processor.push_frame(frame, direction)
            return

        if kind is FrameKind.LLM_TEXT:
            payload = frame.payload
            if isinstance(payload, str):
                aggregator._parts.append(payload)
            await processor.push_frame(frame, direction)
            return

        if kind is FrameKind.LLM_FULL_RESPONSE_END:
            await aggregator._finish()
            await processor.push_frame(frame, direction)
            return

        if kind is FrameKind.INTERRUPTION:
            aggregator._parts.clear()
            await processor.push_frame(frame, direction)
            return

        await processor.push_frame(frame, direction)
