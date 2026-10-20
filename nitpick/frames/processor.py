"""The :class:`FrameProcessor` — nitpick's universal building block.

Every component (VAD, STT, LLM, TTS, transport, the pipeline itself) is a
``FrameProcessor``. A processor owns a handler plus **two lanes**:

``input lane``
    Frames arrive here. System frames are processed **inline**; everything
    else is forwarded to the process lane.

``process lane``
    Data and control frames are processed here, one at a time.

The split is the whole point. System frames (interruption, cancel, VAD stop)
always overtake queued data frames, so an in-flight LLM or TTS handler can
never delay an interruption. Without this, a user barge-in waits behind
whatever audio is already buffered.

Implementation note
-------------------
``asyncio.PriorityQueue`` cannot compare arbitrary items when priorities tie,
so every entry is pushed as ``(priority, sequence, item)`` — the sequence
number breaks ties and guarantees FIFO within a priority level.
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import weakref
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

from .frames import Frame, frame_error, frame_interruption
from .kinds import FrameKind

log = logging.getLogger(__name__)

#: Direction of travel through the pipeline.
DOWNSTREAM = "downstream"
UPSTREAM = "upstream"

Sequence = itertools.count()


@runtime_checkable
class FrameHandler(Protocol):
    """Processes a single frame."""

    async def on_process_frame(
        self, processor: "FrameProcessor", frame: Frame, direction: str
    ) -> None: ...


class PassthroughHandler:
    """Forwards every frame unchanged. Useful as a placeholder."""

    async def on_process_frame(
        self, processor: "FrameProcessor", frame: Frame, direction: str
    ) -> None:
        await processor.push_frame(frame, direction)


class FrameProcessor:
    """A single node in a pipeline.

    Parameters
    ----------
    name:
        Human-readable name — appears in logs and observer output.
    handler:
        Object implementing :class:`FrameHandler`. Defaults to a pass-through.
    direct:
        When true the processor skips both lanes and handles frames inline on
        the caller's stack. Used for the pipeline's source and sink
        sentinels, which must intercept frames without a scheduling hop.
    """

    def __init__(
        self,
        name: str,
        handler: FrameHandler | None = None,
        *,
        direct: bool = False,
    ) -> None:
        self.name = name
        self._handler: FrameHandler = handler or PassthroughHandler()
        self._direct = direct

        # Linked-list links. ``_next`` is strong, ``_prev`` is weak so a
        # pipeline does not form a reference cycle.
        self._next: "FrameProcessor | None" = None
        self._prev: "weakref.ReferenceType[FrameProcessor] | None" = None

        # Lanes.
        self._input_q: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._process_q: asyncio.Queue = asyncio.Queue()

        # Task handles.
        self._input_task: asyncio.Task | None = None
        self._process_task: asyncio.Task | None = None

        # State.
        self._started = False
        self._cancelling = False
        self._allow_interruptions = False
        self._enable_metrics = False
        self._observers: list[Any] = []

        # Sub-processors (compound processors such as Pipeline).
        self._sub_processors: list["FrameProcessor"] = []
        self._entry_processors: list["FrameProcessor"] = []

    # ------------------------------------------------------------------
    # Links
    # ------------------------------------------------------------------

    def link(self, nxt: "FrameProcessor") -> None:
        """Connect ``self -> nxt``."""
        self._next = nxt
        nxt._prev = weakref.ref(self)

    @property
    def next(self) -> "FrameProcessor | None":
        return self._next

    @property
    def previous(self) -> "FrameProcessor | None":
        return self._prev() if self._prev is not None else None

    def set_sub_processors(self, processors: list["FrameProcessor"]) -> None:
        self._sub_processors = list(processors)

    @property
    def sub_processors(self) -> list["FrameProcessor"]:
        return list(self._sub_processors)

    def set_entry_processors(self, processors: list["FrameProcessor"]) -> None:
        self._entry_processors = list(processors)

    @property
    def entry_processors(self) -> list["FrameProcessor"]:
        return list(self._entry_processors)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        """Initialise this processor and, recursively, its sub-processors."""
        if not self._direct:
            self._input_task = asyncio.create_task(
                self._input_loop(), name=f"nitpick-input-{self.name}"
            )
        for proc in self._sub_processors:
            await proc.setup()

    async def cleanup(self) -> None:
        """Tear down this processor and, recursively, its sub-processors."""
        await self._cancel_task(self._input_task)
        self._input_task = None
        await self._cancel_task(self._process_task)
        self._process_task = None
        for proc in self._sub_processors:
            await proc.cleanup()

    @staticmethod
    async def _cancel_task(task: asyncio.Task | None) -> None:
        """Cancel a task and wait for it to finish.

        ``asyncio.wait`` is used rather than ``await task`` so the target's
        ``CancelledError`` is not re-raised here — and so an outer
        cancellation is not accidentally absorbed by the inner one.
        """
        if task is None or task.done():
            return
        task.cancel()
        await asyncio.wait([task])

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def queue_frame(self, frame: Frame, direction: str = DOWNSTREAM) -> None:
        """Hand a frame to this processor.

        This is the only way frames enter a processor. It routes to the
        appropriate lane.
        """
        if self._cancelling:
            return

        if self._direct:
            await self._internal_process(frame, direction)
            return

        priority = 0 if frame.is_system else 1
        self._input_q.put_nowait((priority, next(Sequence), frame, direction))

    # ------------------------------------------------------------------
    # Lanes
    # ------------------------------------------------------------------

    async def _input_loop(self) -> None:
        """Drain the input lane.

        System frames are processed inline so they leap the process lane.
        Everything else is handed to the process lane.
        """
        while True:
            _prio, _seq, frame, direction = await self._input_q.get()
            if frame.is_system:
                await self._internal_process(frame, direction)
            else:
                await self._process_q.put((frame, direction))

    async def _process_loop(self) -> None:
        """Drain the process lane — data and control frames, one at a time."""
        while True:
            frame, direction = await self._process_q.get()
            await self._internal_process(frame, direction)

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    async def _internal_process(self, frame: Frame, direction: str) -> None:
        for observer in list(self._observers):
            try:
                result = observer.on_process_frame(self.name, frame, direction)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:  # pragma: no cover - observers must never break the pipeline
                log.exception("Observer failed on %s", self.name)

        try:
            await self._handle_system(frame)
            await self._handler.on_process_frame(self, frame, direction)
        except Exception as exc:  # noqa: BLE001 - surfaced as a frame, not a crash
            log.exception("Processor %s failed handling %s", self.name, frame.kind)
            await self.push_error(f"Error processing {frame.kind}: {exc}", fatal=False)

    async def _handle_system(self, frame: Frame) -> None:
        """Apply framework-level reactions to system frames."""
        kind = frame.kind
        if kind is FrameKind.START:
            self._started = True
            payload = frame.payload
            self._allow_interruptions = bool(
                getattr(payload, "allow_interruptions", False)
            )
            self._enable_metrics = bool(getattr(payload, "enable_metrics", False))
            if not self._direct and self._process_task is None:
                self._process_task = asyncio.create_task(
                    self._process_loop(), name=f"nitpick-process-{self.name}"
                )
        elif kind is FrameKind.CANCEL:
            self._cancelling = True
            await self._cancel_task(self._process_task)
            self._process_task = None
        elif kind is FrameKind.INTERRUPTION:
            await self._interrupt()

    async def _interrupt(self) -> None:
        """Drop queued work and restart the process lane."""
        await self._drain_queue(self._process_q)
        await self._cancel_task(self._process_task)
        self._process_task = None
        if not self._direct:
            self._process_task = asyncio.create_task(
                self._process_loop(), name=f"nitpick-process-{self.name}"
            )

    @staticmethod
    async def _drain_queue(queue: asyncio.Queue) -> None:
        """Drop every queued frame that is allowed to be dropped."""
        while not queue.empty():
            try:
                frame, _direction = queue.get_nowait()
            except asyncio.QueueEmpty:  # pragma: no cover - race with producers
                break
            if frame.is_uninterruptible:
                queue.put_nowait((frame, DOWNSTREAM))

    # ------------------------------------------------------------------
    # Pushing
    # ------------------------------------------------------------------

    async def push_frame(self, frame: Frame, direction: str = DOWNSTREAM) -> None:
        """Push a frame to the next (or previous) processor.

        The frame is handed to the neighbour's *queue*, so if the neighbour
        has not booted yet the frame simply waits there. The ``START`` frame
        is the first thing that flows through a pipeline, so anything queued
        behind it is delivered in order once the pipeline is up.

        Note that this deliberately does **not** re-queue into this
        processor's own input queue when it is not started: the input loop
        would pick the frame up, call the handler, the handler would push
        the frame again, and the two would spin forever.
        """
        if direction == DOWNSTREAM:
            target = self._next
        else:
            target = self.previous

        if target is None:
            return

        for observer in list(self._observers):
            try:
                result = observer.on_push_frame(self.name, target.name, frame, direction)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:  # pragma: no cover
                log.exception("Observer failed on push")

        await target.queue_frame(frame, direction)

    async def push_error(self, message: str, fatal: bool = False) -> None:
        """Raise an error frame upstream."""
        await self.push_frame(
            frame_error(message, fatal=fatal, processor_name=self.name), UPSTREAM
        )

    async def broadcast_frame(self, template: Frame) -> None:
        """Send a frame both downstream and upstream, linked by sibling id."""
        downstream = template.with_new_id()
        upstream = template.with_new_id()
        downstream.with_sibling(upstream.id)
        upstream.with_sibling(downstream.id)
        await self.push_frame(downstream, DOWNSTREAM)
        await self.push_frame(upstream, UPSTREAM)

    async def broadcast_interruption(self) -> None:
        """Interrupt the whole pipeline."""
        await self.broadcast_frame(frame_interruption())

    # ------------------------------------------------------------------
    # Observers
    # ------------------------------------------------------------------

    def add_observer(self, observer: Any) -> None:
        self._observers.append(observer)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def started(self) -> bool:
        return self._started

    @property
    def allow_interruptions(self) -> bool:
        return self._allow_interruptions

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"FrameProcessor({self.name!r})"


#: A frame handler may also be a plain coroutine function.
HandlerFactory = Callable[[FrameProcessor], Awaitable[None]]
