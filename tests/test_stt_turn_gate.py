"""Tests for the STT turn gate and the transcript-ordering guarantee.

These are the tests that matter most in the whole framework: they lock in
the behaviour that makes a voice agent actually answer.
"""
from __future__ import annotations

import asyncio

import pytest

from nitpick.frames import (
    Frame,
    FrameKind,
    FrameProcessor,
    Transcription,
    frame_vad_stopped,
)
from nitpick.pipeline import PipelineParams, PipelineTask
from nitpick.services.stt.base import SttEvent, SttProvider, SttService


class FakeProvider(SttProvider):
    """A provider that returns a final transcript when flushed."""

    name = "fake"

    def __init__(self, text: str = "hello world", delay: float = 0.0) -> None:
        self.text = text
        self.delay = delay
        self.flushed = False
        self.pushed: list[bytes] = []
        self._on_event = None

    async def start(self, on_event) -> None:
        self._on_event = on_event

    async def push_audio(self, pcm: bytes) -> None:
        self.pushed.append(pcm)

    async def flush(self) -> None:
        self.flushed = True
        if self.delay:
            await asyncio.sleep(self.delay)
        await self._on_event(SttEvent(kind="final", text=self.text))

    async def stop(self) -> None:
        pass


class SilentProvider(FakeProvider):
    """A provider that never answers."""

    async def flush(self) -> None:
        self.flushed = True


@pytest.mark.asyncio
async def test_stop_frame_carries_the_transcript():
    """The turn close and the transcript must arrive as one frame."""
    provider = FakeProvider(text="the answer")
    stt = SttService(provider, release_timeout=2.0)
    seen: list[Frame] = []

    class Capture:
        async def on_process_frame(self, processor, frame, direction) -> None:
            seen.append(frame)
            await processor.push_frame(frame, direction)

    task = PipelineTask([stt, FrameProcessor("capture", Capture())], PipelineParams())
    task.start()
    await asyncio.sleep(0.05)

    await task.push_frame(frame_vad_stopped(0.0, 0.0))
    await asyncio.sleep(0.15)

    stops = [f for f in seen if f.kind is FrameKind.VAD_USER_STOPPED_SPEAKING]
    assert len(stops) == 1, "the stop frame must be emitted exactly once"
    assert stops[0].payload.transcript is not None
    assert stops[0].payload.transcript.text == "the answer"

    await task.cancel()


@pytest.mark.asyncio
async def test_no_transcript_frame_overtakes_the_stop():
    """The transcript must never be delivered after the turn closed."""
    provider = FakeProvider(text="ordering matters")
    stt = SttService(provider, release_timeout=2.0)
    order: list[str] = []

    class Capture:
        async def on_process_frame(self, processor, frame, direction) -> None:
            if frame.kind is FrameKind.VAD_USER_STOPPED_SPEAKING:
                has_text = frame.payload.transcript is not None
                order.append(f"stop(transcript={'yes' if has_text else 'no'})")
            elif frame.kind is FrameKind.TRANSCRIPTION:
                order.append("transcription")
            await processor.push_frame(frame, direction)

    task = PipelineTask([stt, FrameProcessor("capture", Capture())], PipelineParams())
    task.start()
    await asyncio.sleep(0.05)

    await task.push_frame(frame_vad_stopped(0.0, 0.0))
    await asyncio.sleep(0.15)

    assert order == ["stop(transcript=yes)"]
    await task.cancel()


@pytest.mark.asyncio
async def test_stop_is_released_even_when_the_provider_never_answers():
    provider = SilentProvider()
    stt = SttService(provider, release_timeout=0.1)
    seen: list[Frame] = []

    class Capture:
        async def on_process_frame(self, processor, frame, direction) -> None:
            seen.append(frame)
            await processor.push_frame(frame, direction)

    task = PipelineTask([stt, FrameProcessor("capture", Capture())], PipelineParams())
    task.start()
    await asyncio.sleep(0.05)

    await task.push_frame(frame_vad_stopped(0.0, 0.0))
    await asyncio.sleep(0.3)

    stops = [f for f in seen if f.kind is FrameKind.VAD_USER_STOPPED_SPEAKING]
    assert len(stops) == 1
    assert stops[0].payload.transcript is None

    await task.cancel()


@pytest.mark.asyncio
async def test_user_aggregator_triggers_the_llm_from_the_bundled_transcript():
    from nitpick.context import LLMContext
    from nitpick.processors.user_aggregator import UserAggregator

    context = LLMContext("be helpful")
    aggregator = UserAggregator(context)
    triggered: list[Frame] = []

    class Capture:
        async def on_process_frame(self, processor, frame, direction) -> None:
            triggered.append(frame)
            await processor.push_frame(frame, direction)

    task = PipelineTask(
        [aggregator, FrameProcessor("capture", Capture())], PipelineParams()
    )
    task.start()
    await asyncio.sleep(0.05)

    stop = frame_vad_stopped(0.0, 0.0)
    stop.payload.transcript = Transcription(text="what is the weather")
    await task.push_frame(stop)
    await asyncio.sleep(0.1)

    assert any(f.kind is FrameKind.LLM_CONTEXT for f in triggered)
    messages = context.messages
    assert messages[-1].role == "user"
    assert messages[-1].content == "what is the weather"

    await task.cancel()
