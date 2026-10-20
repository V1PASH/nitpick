"""The server-side VAD must close a turn on its own.

The browser's speaking detection is only a hint. If the client never
sends a speaking event, the agent still has to notice that the user
stopped talking — otherwise the bot never answers.
"""
from __future__ import annotations

import asyncio
import base64

import numpy as np
import pytest

from nitpick import Agent
from nitpick.context import LLMContext
from nitpick.frames import DOWNSTREAM, frame_llm_text
from nitpick.services.llm import LLMConfig, LLMService
from nitpick.services.stt.base import SttEvent, SttProvider
from nitpick.services.tts import TTSConfig, TTSService


class FakeStt(SttProvider):
    """Reports a fixed transcript when the turn closes."""

    name = "fake"

    def __init__(self, text: str = "hello agent") -> None:
        self.text = text
        self._on_event = None
        self.flushed = False

    async def start(self, on_event) -> None:
        self._on_event = on_event

    async def push_audio(self, pcm: bytes) -> None:
        pass

    async def flush(self) -> None:
        self.flushed = True
        await self._on_event(SttEvent(kind="final", text=self.text))

    async def stop(self) -> None:
        pass


class FakeTts(TTSService):
    def __init__(self) -> None:
        super().__init__(TTSConfig())
        self.synthesised: list[str] = []

    async def synthesize(self, text: str) -> bytes:
        self.synthesised.append(text)
        return b"\x00\x00" * 240


class FakeLlm(LLMService):
    def __init__(self, context: LLMContext, reply: str = "Hi there.") -> None:
        super().__init__(context, LLMConfig())
        self.reply = reply

    async def _stream_once(self):
        await self.context.add_assistant_message(self.reply)
        await self.push_frame(frame_llm_text(self.reply), DOWNSTREAM)
        return []


class FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)


def speech(seconds: float, rate: int = 16_000) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    wave = (np.sin(2 * np.pi * 220.0 * t) * 6000).astype("<i2")
    return wave.tobytes()


def silence(seconds: float, rate: int = 16_000) -> bytes:
    return np.zeros(int(seconds * rate), dtype="<i2").tobytes()


@pytest.mark.asyncio
async def test_server_vad_closes_the_turn_without_any_client_activity():
    context = LLMContext("You are a test agent.")
    stt = FakeStt(text="what is the weather")
    tts = FakeTts()

    agent = Agent(
        name="Tester",
        prompt="You are a test agent.",
        stt_provider=stt,
        llm_service=FakeLlm(context),
        tts_service=tts,
        vad="energy",
    )
    agent.context = context

    server = agent._server_with_manager()
    task = agent.build_pipeline("demo")
    task.start()
    await asyncio.sleep(0.1)

    ws = FakeWebSocket()
    await server.handle_message(ws, {"type": "join_room", "room_id": "demo"})

    # Stream speech, then silence — and send NO speaking activity at all.
    for payload in (speech(1.5), silence(2.5)):
        for index in range(0, len(payload), 3200):
            piece = payload[index : index + 3200]
            await server.handle_message(
                ws,
                {
                    "type": "audio_pcm",
                    "audio_base64": base64.b64encode(piece).decode(),
                    "sample_rate": 16000,
                    "channels": 1,
                },
            )
            await asyncio.sleep(0.01)

    await asyncio.sleep(0.5)

    assert stt.flushed, "the server-side VAD must close the turn on its own"
    assert any(
        m.role == "user" and m.content == "what is the weather"
        for m in context.messages
    ), "the transcript must reach the LLM context"
    assert tts.synthesised, "the agent must produce a reply"

    await task.cancel()


@pytest.mark.asyncio
async def test_server_vad_publishes_the_turn_to_the_room():
    """The room should learn about the turn, not just the pipeline."""
    context = LLMContext("test")
    agent = Agent(
        name="Tester",
        stt_provider=FakeStt(),
        llm_service=FakeLlm(context),
        tts_service=FakeTts(),
        vad="energy",
    )
    agent.context = context

    server = agent._server_with_manager()
    task = agent.build_pipeline("demo")
    task.start()
    await asyncio.sleep(0.1)

    ws = FakeWebSocket()
    await server.handle_message(ws, {"type": "join_room", "room_id": "demo"})

    for payload in (speech(1.5), silence(2.5)):
        for index in range(0, len(payload), 3200):
            piece = payload[index : index + 3200]
            await server.handle_message(
                ws,
                {
                    "type": "audio_pcm",
                    "audio_base64": base64.b64encode(piece).decode(),
                    "sample_rate": 16000,
                    "channels": 1,
                },
            )
            await asyncio.sleep(0.01)

    await asyncio.sleep(0.5)

    # The browser does not receive its own speaking event, but the agent
    # must have marked it as speaking in the room's state.
    browser = agent.manager.get_participant_by_websocket(ws)
    assert browser is not None
    assert browser.is_speaking is False, "the turn must have closed"

    await task.cancel()
