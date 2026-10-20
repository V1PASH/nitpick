"""Regression tests for bugs found while building nitpick.

Each test here corresponds to a defect that shipped at least once and is
now locked in.
"""
from __future__ import annotations

import asyncio

import pytest

from nitpick.frames import (
    DOWNSTREAM,
    AudioChunk,
    Frame,
    FrameKind,
    FrameProcessor,
    frame_input_audio,
    frame_llm_text,
)
from nitpick.pipeline import PipelineParams, PipelineTask
from nitpick.processors.assistant_aggregator import AssistantAggregator
from nitpick.rooms import RoomManager


# ---------------------------------------------------------------------------
# The signaling WebSocket route must actually match.
#
# The module uses ``from __future__ import annotations``, so FastAPI resolves
# the endpoint's annotations by name from the module globals. With the
# FastAPI imports done inside the function, ``WebSocket`` was unresolvable
# and every connection was rejected with a policy-violation close.
# ---------------------------------------------------------------------------


def test_signaling_app_registers_the_websocket_route():
    from nitpick.signaling.server import create_app

    app = create_app()
    paths = {getattr(route, "path", None) for route in app.routes}
    assert "/ws" in paths
    assert "/health" in paths


@pytest.mark.asyncio
async def test_websocket_route_accepts_a_connection():
    from nitpick.signaling.server import create_app

    app = create_app()
    sent: list[dict] = []

    scope = {
        "type": "websocket",
        "path": "/ws",
        "raw_path": b"/ws",
        "query_string": b"",
        "headers": [(b"host", b"testserver")],
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
        "scheme": "ws",
        "root_path": "",
        "subprotocols": [],
        "asgi": {"version": "3.0", "spec_version": "2.3"},
    }

    received = [
        {"type": "websocket.connect"},
        {"type": "websocket.disconnect", "code": 1000},
    ]

    async def receive():
        if received:
            return received.pop(0)
        await asyncio.sleep(0.05)
        return {"type": "websocket.disconnect", "code": 1000}

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)

    types = [message.get("type") for message in sent]
    assert "websocket.accept" in types, f"connection was not accepted: {types}"
    assert "websocket.close" not in types


# ---------------------------------------------------------------------------
# A "speaking stopped" with no matching "speaking started" must still fire.
#
# The event bus edge-triggers the *start* to avoid duplicates, but the stop
# closes the user's turn. Dropping it strands the turn open and the agent
# never answers.
# ---------------------------------------------------------------------------


def test_stop_speaking_is_emitted_even_without_a_start():
    manager = RoomManager()
    participant = manager.register_participant(None, "a")
    manager.join_room(participant, "demo")

    assert participant.is_speaking is False

    event = manager.update_participant_speaking(participant, False)

    assert event is not None
    assert event["event_type"] == "participant_speaking_stopped"


def test_start_speaking_is_still_edge_triggered():
    manager = RoomManager()
    participant = manager.register_participant(None, "a")
    manager.join_room(participant, "demo")

    first = manager.update_participant_speaking(participant, True)
    second = manager.update_participant_speaking(participant, True)

    assert first is not None
    assert second is None, "a repeated 'speaking started' must not emit again"


# ---------------------------------------------------------------------------
# The assistant message must land in the context exactly once.
#
# Both the LLM service and the assistant aggregator write it, which
# duplicated every reply in the history.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_assistant_message_is_not_duplicated():
    from nitpick.context import LLMContext

    context = LLMContext("be helpful")
    aggregator = AssistantAggregator(context)

    class Capture:
        async def on_process_frame(self, processor, frame, direction):
            await processor.push_frame(frame, direction)

    task = PipelineTask([aggregator, FrameProcessor("capture", Capture())], PipelineParams())
    task.start()
    await asyncio.sleep(0.05)

    await task.push_frame(Frame(FrameKind.LLM_FULL_RESPONSE_START))
    await task.push_frame(frame_llm_text("Hello there."))
    await task.push_frame(Frame(FrameKind.LLM_FULL_RESPONSE_END))
    await asyncio.sleep(0.1)

    assistant_messages = [m for m in context.messages if m.role == "assistant"]
    assert len(assistant_messages) == 1
    assert assistant_messages[0].content == "Hello there."

    await task.cancel()


@pytest.mark.asyncio
async def test_assistant_message_written_by_the_llm_is_not_duplicated():
    """A tool round records the assistant message in the LLM service."""
    from nitpick.context import LLMContext

    context = LLMContext("be helpful")
    aggregator = AssistantAggregator(context)

    class Capture:
        async def on_process_frame(self, processor, frame, direction):
            await processor.push_frame(frame, direction)

    task = PipelineTask([aggregator, FrameProcessor("capture", Capture())], PipelineParams())
    task.start()
    await asyncio.sleep(0.05)

    # The LLM service already wrote this message (it carried tool calls).
    await context.add_assistant_message("Hello there.")
    await task.push_frame(Frame(FrameKind.LLM_FULL_RESPONSE_START))
    await task.push_frame(frame_llm_text("Hello there."))
    await task.push_frame(Frame(FrameKind.LLM_FULL_RESPONSE_END))
    await asyncio.sleep(0.1)

    assistant_messages = [m for m in context.messages if m.role == "assistant"]
    assert len(assistant_messages) == 1

    await task.cancel()


# ---------------------------------------------------------------------------
# The room transport must subscribe before the pipeline boots.
#
# Subscribing on the START frame left a window in which arriving audio was
# silently dropped.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transport_receives_audio_that_arrives_before_the_pipeline_starts():
    from nitpick.transport import RoomTransport, TransportParams

    manager = RoomManager()
    participant = manager.register_agent_participant("Agent")
    manager.join_room(participant, "demo")

    transport = RoomTransport(manager, participant, params=TransportParams())

    browser = manager.register_participant(None, "browser")
    manager.join_room(browser, "demo")

    # The downstream processor holds the frame until the pipeline boots.
    downstream = FrameProcessor("downstream")
    transport.input().link(downstream)

    # Audio arrives before the pipeline is started.
    manager.emit_participant_audio(
        browser, audio=b"\x01\x02\x03\x04", sample_rate=16000, channels=1
    )
    await asyncio.sleep(0.05)

    assert downstream._input_q.qsize() == 1, (
        "the frame must be held by the downstream processor, not dropped"
    )


# ---------------------------------------------------------------------------
# A processor must not re-queue a frame into its own input queue.
#
# ``push_frame`` used to check "have I started?" and, if not, put the frame
# back into this processor's own input queue. The input loop then picked it
# up, called the handler, the handler pushed the frame again, and the two
# spun forever — a hang, not a dropped frame.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pushing_before_start_does_not_loop_forever():
    processor = FrameProcessor("solo")
    await processor.setup()

    pushed: list[Frame] = []

    class Handler:
        async def on_process_frame(self, processor, frame, direction):
            pushed.append(frame)
            # Pushing again is what used to spin.
            await processor.push_frame(frame, direction)

    processor._handler = Handler()

    await asyncio.wait_for(
        processor.push_frame(Frame(FrameKind.DATA, b"x"), DOWNSTREAM),
        timeout=2.0,
    )
    await asyncio.sleep(0.05)

    # The processor has no neighbour, so the frame goes nowhere — but it
    # must not spin.
    assert pushed == []


# ---------------------------------------------------------------------------
# The VAD processor must not confuse the framework's lifecycle flag with its
# own "the user is speaking" flag.
#
# Both were called ``_started``. The framework sets its flag when the
# START frame arrives, so the VAD's check ``not self._started`` was always
# False after boot and the VAD never emitted a turn boundary.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vad_emits_turn_boundaries():
    import numpy as np

    from nitpick.vad import EnergyVad, VadParams, VadProcessor

    def pcm(samples: int, amplitude: int) -> bytes:
        t = np.arange(samples) / 16_000.0
        wave = (np.sin(2 * np.pi * 220.0 * t) * amplitude).astype("<i2")
        return wave.tobytes()

    vad = VadProcessor(
        EnergyVad(16_000),
        params=VadParams(confidence=0.5, start_secs=0.1, stop_secs=0.1),
    )
    seen: list[FrameKind] = []

    class Capture:
        async def on_process_frame(self, processor, frame, direction):
            seen.append(frame.kind)
            await processor.push_frame(frame, direction)

    task = PipelineTask([vad, FrameProcessor("capture", Capture())], PipelineParams())
    task.start()
    await asyncio.sleep(0.05)

    await task.push_frame(
        frame_input_audio(AudioChunk(data=pcm(16_000, 6000), sample_rate=16_000)),
        DOWNSTREAM,
    )
    await asyncio.sleep(0.1)
    await task.push_frame(
        frame_input_audio(AudioChunk(data=pcm(16_000, 0), sample_rate=16_000)),
        DOWNSTREAM,
    )
    await asyncio.sleep(0.2)

    assert FrameKind.VAD_USER_STARTED_SPEAKING in seen
    assert FrameKind.VAD_USER_STOPPED_SPEAKING in seen

    await task.cancel()
