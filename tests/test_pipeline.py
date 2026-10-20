"""Tests for the frame system and the pipeline."""
from __future__ import annotations

import asyncio

import pytest

from nitpick.frames import (
    DOWNSTREAM,
    UPSTREAM,
    Frame,
    FrameKind,
    FrameProcessor,
    Transcription,
    frame_input_audio,
    frame_start,
    frame_transcription,
    frame_vad_stopped,
)
from nitpick.pipeline import PipelineParams, PipelineTask


class Recorder:
    """A processor that records every frame it sees."""

    def __init__(self, name: str = "recorder") -> None:
        self.frames: list[Frame] = []
        self.processor = FrameProcessor(name, self)

    async def on_process_frame(self, processor, frame, direction) -> None:
        self.frames.append(frame)
        await processor.push_frame(frame, direction)


@pytest.mark.asyncio
async def test_frames_flow_downstream_in_order():
    recorder = Recorder()
    task = PipelineTask([recorder.processor], PipelineParams())
    task.start()
    await asyncio.sleep(0.05)

    await task.push_frame(frame_transcription(Transcription(text="hello")))
    await asyncio.sleep(0.05)

    kinds = [f.kind for f in recorder.frames]
    assert FrameKind.START in kinds
    assert FrameKind.TRANSCRIPTION in kinds
    await task.cancel()


@pytest.mark.asyncio
async def test_system_frames_overtake_data_frames():
    """A system frame must not wait behind queued data frames."""
    gate = asyncio.Event()
    seen: list[FrameKind] = []

    class Blocker:
        async def on_process_frame(self, processor, frame, direction) -> None:
            if frame.kind is FrameKind.DATA:
                await gate.wait()
            seen.append(frame.kind)
            await processor.push_frame(frame, direction)

    blocker = FrameProcessor("blocker", Blocker())
    task = PipelineTask([blocker], PipelineParams())
    task.start()
    await asyncio.sleep(0.05)

    # Queue a pile of data frames, then a system frame.
    for _ in range(20):
        await task.push_frame(Frame(FrameKind.DATA, b"x"))
    await task.push_frame(Frame(FrameKind.HEARTBEAT, 1.0))

    await asyncio.sleep(0.1)
    # The heartbeat must have been processed while the data frames are stuck.
    assert FrameKind.HEARTBEAT in seen
    assert FrameKind.DATA not in seen

    gate.set()
    await task.cancel()


@pytest.mark.asyncio
async def test_upstream_frames_propagate_backwards():
    """A frame pushed upstream reaches the processor before the source."""
    seen: list[tuple[str, str]] = []

    class Capture:
        def __init__(self, name: str) -> None:
            self.name = name

        async def on_process_frame(self, processor, frame, direction) -> None:
            seen.append((self.name, direction))
            await processor.push_frame(frame, direction)

    first = FrameProcessor("first", Capture("first"))
    second = FrameProcessor("second", Capture("second"))

    task = PipelineTask([first, second], PipelineParams())
    task.start()
    await asyncio.sleep(0.05)

    await second.push_frame(Frame(FrameKind.DATA, b"x"), UPSTREAM)
    await asyncio.sleep(0.05)

    assert ("first", UPSTREAM) in seen
    await task.cancel()


@pytest.mark.asyncio
async def test_upstream_frames_escaping_the_source_are_consumed():
    """The source is the upstream boundary — escaping frames stop there."""
    seen: list[str] = []

    class Capture:
        async def on_process_frame(self, processor, frame, direction) -> None:
            seen.append(direction)
            await processor.push_frame(frame, direction)

    first = FrameProcessor("first", Capture())
    task = PipelineTask([first], PipelineParams())
    task.start()
    await asyncio.sleep(0.05)

    await first.push_frame(Frame(FrameKind.DATA, b"x"), UPSTREAM)
    await asyncio.sleep(0.05)

    # The frame reached the source and was consumed there, not bounced back.
    assert seen == [DOWNSTREAM]
    await task.cancel()


@pytest.mark.asyncio
async def test_pipeline_task_finishes_on_end():
    recorder = Recorder()
    task = PipelineTask([recorder.processor], PipelineParams())
    task.start()
    await asyncio.sleep(0.05)

    await task.end()
    reason = await asyncio.wait_for(task.wait(), timeout=2.0)

    assert reason.value == "end"
