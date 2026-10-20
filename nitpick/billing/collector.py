"""Billing collection and storage.

The hot path must never block. A session records events by pushing them
onto an **unbounded** queue that a background task drains — a single
``put_nowait`` that cannot block, cannot fail and cannot drop. Cost
tracking therefore costs the audio path nothing.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from pathlib import Path
from typing import Protocol, runtime_checkable

from .events import BillingEvent, TranscriptEntry, TranscriptRole

log = logging.getLogger(__name__)


@runtime_checkable
class BillingStorage(Protocol):
    """Where billing events are written."""

    def write(self, event: BillingEvent) -> None:
        """Persist one event. Called from the drain task, never the hot path."""


class LogBillingStorage:
    """Writes events as JSON lines to a file.

    The file is appended to from a background task, but the write itself
    is blocking, so it runs in a worker thread to keep the event loop free.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def write(self, event: BillingEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event.to_dict(), ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


class MemoryBillingStorage:
    """Keeps events in memory. Useful for tests and for a live dashboard."""

    def __init__(self) -> None:
        self.events: list[BillingEvent] = []

    def write(self, event: BillingEvent) -> None:
        self.events.append(event)


class BillingCollector:
    """Records billing events for one session."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._queue: asyncio.Queue[BillingEvent] = asyncio.Queue()
        self._storage: BillingStorage | None = None
        self._task: asyncio.Task | None = None
        self._closed = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def attach(self, storage: BillingStorage) -> None:
        """Send recorded events to a storage backend."""
        self._storage = storage

    def start(self) -> None:
        """Start the drain task."""
        if self._task is None:
            self._task = asyncio.create_task(self._drain(), name="nitpick-billing")

    async def close(self) -> None:
        """Flush and stop the drain task."""
        if self._closed:
            return
        self._closed = True

        if self._task is not None:
            # Stop the drain loop, then write whatever is still queued.
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        await self._flush_remaining()

    async def _drain(self) -> None:
        while True:
            event = await self._queue.get()
            await self._write(event)

    async def _flush_remaining(self) -> None:
        while not self._queue.empty():
            try:
                event = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            await self._write(event)

    async def _write(self, event: BillingEvent) -> None:
        if self._storage is None:
            return
        try:
            await asyncio.to_thread(self._storage.write, event)
        except Exception:  # noqa: BLE001 - billing must never break the session
            log.exception("Failed to write a billing event")

    # ------------------------------------------------------------------
    # Recording — the hot path
    # ------------------------------------------------------------------

    def record(self, event: BillingEvent) -> None:
        """Record an event. Never blocks, never drops, never raises."""
        if self._closed:
            return
        self._queue.put_nowait(event)

    # ------------------------------------------------------------------
    # Convenience recorders
    # ------------------------------------------------------------------

    def record_session_start(
        self, metadata: dict[str, str] | None = None, *, started_at: float | None = None
    ) -> None:
        self.record(
            BillingEvent(
                session_id=self.session_id,
                kind="session_start",
                started_at=started_at,
                metadata=dict(metadata or {}),
            )
        )

    def record_session_end(
        self, finish_reason: str = "end", *, ended_at: float | None = None
    ) -> None:
        self.record(
            BillingEvent(
                session_id=self.session_id,
                kind="session_end",
                ended_at=ended_at,
                finish_reason=finish_reason,
            )
        )

    def record_llm_usage(
        self,
        *,
        model: str,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        latency_ms: float | None = None,
    ) -> None:
        total = None
        if prompt_tokens is not None and completion_tokens is not None:
            total = prompt_tokens + completion_tokens
        self.record(
            BillingEvent(
                session_id=self.session_id,
                kind="llm_usage",
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total,
                latency_ms=latency_ms,
            )
        )

    def record_tts_usage(
        self, *, characters: int, model: str | None = None
    ) -> None:
        by_model = {model: characters} if model else None
        self.record(
            BillingEvent(
                session_id=self.session_id,
                kind="tts_usage",
                characters=characters,
                characters_by_model=by_model,
                model=model,
            )
        )

    def record_stt_usage(
        self, *, provider: str, audio_seconds: float | None = None
    ) -> None:
        self.record(
            BillingEvent(
                session_id=self.session_id,
                kind="stt_usage",
                provider=provider,
                audio_seconds=audio_seconds,
            )
        )

    def record_transcript(
        self, text: str, role: TranscriptRole, *, turn_id: str | None = None
    ) -> None:
        self.record(
            BillingEvent(
                session_id=self.session_id,
                kind="transcript",
                transcript=TranscriptEntry(text=text, role=role, turn_id=turn_id),
            )
        )

    @property
    def pending(self) -> int:
        """How many events are waiting to be written."""
        return self._queue.qsize()


class SessionBilling(BillingCollector):
    """A billing collector bound to one session id."""

    def __init__(self, session_id: str) -> None:
        super().__init__(session_id)
