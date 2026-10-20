"""``PipelineTask`` — the lifecycle wrapper around a pipeline.

Responsibilities:

* inject the ``START`` frame that boots every processor
* provide a channel that external code (a transport, an SDK client) can push
  frames into
* optional heartbeats and idle timeout
* clean shutdown on ``END`` / ``STOP`` / ``CANCEL``
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

from ..frames import (
    DOWNSTREAM,
    UPSTREAM,
    Frame,
    FrameKind,
    FrameProcessor,
    StartPayload,
    frame_cancel,
    frame_end,
    frame_start,
    frame_stop,
)

log = logging.getLogger(__name__)


class FinishReason(str, Enum):
    """Why a pipeline stopped."""

    END = "end"
    STOP = "stop"
    CANCEL = "cancel"


@dataclass
class PipelineParams:
    """Tunables for a :class:`PipelineTask`."""

    allow_interruptions: bool = True
    enable_metrics: bool = False
    enable_usage_metrics: bool = False

    enable_heartbeats: bool = False
    heartbeat_seconds: float = 1.0

    #: Seconds of silence before the idle monitor fires. ``None`` disables it.
    idle_timeout: float | None = None
    #: Automatically cancel the pipeline when the idle timeout fires.
    cancel_on_idle_timeout: bool = True

    metadata: dict[str, str] = field(default_factory=dict)


class _TaskSource:
    """Sentinel at the head of the pipeline.

    Converts task-control frames that escape upstream into real lifecycle
    frames pushed downstream.
    """

    def __init__(self, task: "PipelineTask") -> None:
        self._task = task

    async def on_process_frame(
        self, processor: FrameProcessor, frame: Frame, direction: str
    ) -> None:
        if direction == DOWNSTREAM:
            await processor.push_frame(frame, direction)
            return

        # Upstream escape.
        if frame.kind is FrameKind.END_TASK:
            await processor.push_frame(frame_end(), DOWNSTREAM)
        elif frame.kind is FrameKind.CANCEL_TASK:
            await processor.push_frame(frame_cancel(), DOWNSTREAM)
        elif frame.kind is FrameKind.STOP_TASK:
            await processor.push_frame(frame_stop(), DOWNSTREAM)
        elif frame.kind is FrameKind.INTERRUPTION_TASK:
            await processor.broadcast_interruption()
        else:
            self._task._note_activity(frame)


class _TaskSink:
    """Sentinel at the tail of the pipeline.

    Watches for lifecycle frames leaving the downstream end and records the
    finish reason.
    """

    def __init__(self, task: "PipelineTask") -> None:
        self._task = task

    async def on_process_frame(
        self, processor: FrameProcessor, frame: Frame, direction: str
    ) -> None:
        if direction == UPSTREAM:
            await processor.push_frame(frame, direction)
            return

        kind = frame.kind
        if kind is FrameKind.END:
            self._task._finish(FinishReason.END)
        elif kind is FrameKind.STOP:
            self._task._finish(FinishReason.STOP)
        elif kind is FrameKind.CANCEL:
            self._task._finish(FinishReason.CANCEL)
        elif kind is FrameKind.START:
            self._task._started_event.set()
        else:
            self._task._note_activity(frame)

        await processor.push_frame(frame, direction)


class PipelineTask:
    """Runs a pipeline to completion.

    Typical use::

        task = PipelineTask([stt, llm, tts], PipelineParams())
        push = task.push_channel()
        task.start(push)
        await task.wait()
    """

    def __init__(self, processors: list[FrameProcessor], params: PipelineParams | None = None) -> None:
        self.params = params or PipelineParams()
        self._processors = list(processors)

        self._source = FrameProcessor("TaskSource", _TaskSource(self), direct=True)
        self._sink = FrameProcessor("TaskSink", _TaskSink(self), direct=True)

        chain = [self._source, *self._processors, self._sink]
        for index in range(len(chain) - 1):
            chain[index].link(chain[index + 1])

        self._chain = chain

        self._push_queue: asyncio.Queue = asyncio.Queue()
        self._finished = asyncio.Event()
        self._started_event = asyncio.Event()
        self._finish_reason: FinishReason | None = None
        self._tasks: list[asyncio.Task] = []
        self._run_task: asyncio.Task | None = None
        self._activity = asyncio.Event()
        self._on_started: list[Callable[[Frame], Awaitable[None] | None]] = []
        self._on_finished: list[Callable[[Frame, FinishReason], Awaitable[None] | None]] = []
        self._observers: list[Any] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def on_started(self, callback: Callable[[Frame], Awaitable[None] | None]) -> None:
        self._on_started.append(callback)

    def on_finished(
        self, callback: Callable[[Frame, FinishReason], Awaitable[None] | None]
    ) -> None:
        self._on_finished.append(callback)

    def add_observer(self, observer: Any) -> None:
        self._observers.append(observer)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def push_channel(self) -> asyncio.Queue:
        """The queue external code pushes frames into.

        Use :meth:`push_frame` to send a frame.
        """
        return self._push_queue

    async def push_frame(self, frame: Frame, direction: str = DOWNSTREAM) -> None:
        """Push a frame into the pipeline from outside it."""
        await self._push_queue.put((frame, direction))

    def start(self) -> None:
        """Boot the pipeline. Does not block."""
        self._run_task = asyncio.create_task(self._run(), name="nitpick-pipeline")

    async def _run(self) -> None:
        # Set up every processor, then inject START.
        for proc in self._chain:
            await proc.setup()

        await self._chain[0].queue_frame(
            frame_start(
                StartPayload(
                    allow_interruptions=self.params.allow_interruptions,
                    enable_metrics=self.params.enable_metrics,
                    enable_usage_metrics=self.params.enable_usage_metrics,
                    metadata=self.params.metadata,
                )
            ),
            DOWNSTREAM,
        )

        # Drain the external push queue.
        self._tasks.append(
            asyncio.create_task(self._drain_push_queue(), name="nitpick-push")
        )

        if self.params.enable_heartbeats:
            self._tasks.append(
                asyncio.create_task(self._heartbeat_loop(), name="nitpick-heartbeat")
            )
        if self.params.idle_timeout:
            self._tasks.append(
                asyncio.create_task(self._idle_loop(), name="nitpick-idle")
            )

        await self._finished.wait()

        # Stop the background tasks (push queue, heartbeat, idle monitor).
        # ``self._tasks`` holds only those — the run task is tracked
        # separately so it never cancels itself.
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        for proc in self._chain:
            await proc.cleanup()

    async def _drain_push_queue(self) -> None:
        while True:
            frame, direction = await self._push_queue.get()
            await self._chain[0].queue_frame(frame, direction)

    async def _heartbeat_loop(self) -> None:
        import time

        while True:
            await asyncio.sleep(self.params.heartbeat_seconds)
            await self._chain[0].queue_frame(
                Frame(FrameKind.HEARTBEAT, time.time()), DOWNSTREAM
            )

    async def _idle_loop(self) -> None:
        timeout = self.params.idle_timeout
        assert timeout is not None
        while True:
            try:
                await asyncio.wait_for(self._activity.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                if self.params.cancel_on_idle_timeout:
                    log.info("Pipeline idle for %ss — cancelling", timeout)
                    await self._chain[0].queue_frame(frame_cancel(), DOWNSTREAM)
                return
            self._activity.clear()

    def _note_activity(self, frame: Frame) -> None:
        """Reset the idle timer on activity frames."""
        if frame.kind in (FrameKind.BOT_SPEAKING, FrameKind.USER_SPEAKING):
            self._activity.set()

    def _finish(self, reason: FinishReason) -> None:
        if self._finish_reason is not None:
            return
        self._finish_reason = reason
        self._finished.set()

    # ------------------------------------------------------------------
    # Waiting
    # ------------------------------------------------------------------

    async def wait(self) -> FinishReason:
        """Block until the pipeline finishes; returns the finish reason."""
        await self._finished.wait()
        return self._finish_reason or FinishReason.END

    async def cancel(self) -> None:
        """Cancel the pipeline."""
        await self._chain[0].queue_frame(frame_cancel(), DOWNSTREAM)

    async def end(self) -> None:
        """End the pipeline gracefully."""
        await self._chain[0].queue_frame(frame_end(), DOWNSTREAM)

    @property
    def finish_reason(self) -> FinishReason | None:
        return self._finish_reason
