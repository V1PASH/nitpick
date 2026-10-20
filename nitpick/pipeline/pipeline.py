"""Pipeline assembly.

A :class:`Pipeline` links processors into a chain and is itself a
:class:`~nitpick.frames.FrameProcessor`, so pipelines nest inside pipelines.
"""
from __future__ import annotations

import logging

from ..frames import DOWNSTREAM, Frame, FrameProcessor

log = logging.getLogger(__name__)


class Pipeline(FrameProcessor):
    """A linked chain of processors.

    Parameters
    ----------
    processors:
        The processors to chain, in order. An empty list is allowed and
        behaves as a pass-through.
    name:
        Name for this pipeline (used in logs and observer output).
    """

    def __init__(self, processors: list[FrameProcessor], name: str = "Pipeline") -> None:
        super().__init__(name, direct=True)
        self._chain: list[FrameProcessor] = list(processors)

        for index in range(len(self._chain) - 1):
            self._chain[index].link(self._chain[index + 1])

        self.set_sub_processors(self._chain)
        self.set_entry_processors(self._chain[:1])

    @property
    def chain(self) -> list[FrameProcessor]:
        """The processors in this pipeline, in order."""
        return list(self._chain)

    async def push_frame(self, frame: Frame, direction: str = DOWNSTREAM) -> None:
        """Push into the head of the chain (or out of the tail, upstream)."""
        if direction == DOWNSTREAM:
            head = self._chain[0] if self._chain else None
            if head is not None:
                await head.queue_frame(frame, direction)
            return
        await super().push_frame(frame, direction)

    async def queue_frame(self, frame: Frame, direction: str = DOWNSTREAM) -> None:
        """Entry point used by the task's push channel."""
        await self.push_frame(frame, direction)
