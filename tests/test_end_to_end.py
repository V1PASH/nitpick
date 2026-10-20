"""End-to-end test: a browser talks to an agent in a room.

Simulates the full loop — join, stream PCM, end the turn — and asserts the
agent produces a transcript, runs the LLM, and emits audio.
"""
from __future__ import annotations

import asyncio
import base64

import pytest

from nitpick import Agent
from nitpick.context import LLMContext
from nitpick.frames import DOWNSTREAM, frame_llm_text
from nitpick.services.llm import LLMConfig, LLMService
from nitpick.services.stt.base import SttEvent, SttProvider
from nitpick.services.tts import TTSConfig, TTSService


class FakeSttProvider(SttProvider):
    """Returns a fixed transcript for whatever audio it is given."""

    name = "fake"

    def __init__(self, text: str = "hello agent") -> None:
        self.text = text
        self._on_event = None
        self.pushed: list[bytes] = []

    async def start(self, on_event) -> None:
        self._on_event = on_event

    async def push_audio(self, pcm: bytes) -> None:
        self.pushed.append(pcm)

    async def flush(self) -> None:
        await self._on_event(SttEvent(kind="final", text=self.text))

    async def stop(self) -> None:
        pass


class FakeTtsService(TTSService):
    """Records what would have been synthesised."""

    def __init__(self) -> None:
        super().__init__(TTSConfig())
        self.synthesised: list[str] = []

    async def synthesize(self, text: str) -> bytes:
        self.synthesised.append(text)
        return b"\x00\x00" * 240  # 20 ms of silence at 24 kHz


class FakeLlmService(LLMService):
    """Answers without touching the network."""

    def __init__(self, context: LLMContext, reply: str = "It is sunny.") -> None:
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


@pytest.mark.asyncio
async def test_browser_talk_cycle():
    context = LLMContext("You are a test agent.")
    stt = FakeSttProvider(text="what is the weather")
    tts = FakeTtsService()

    agent = Agent(
        name="Tester",
        prompt="You are a test agent.",
        stt_provider=stt,
        llm_service=FakeLlmService(context),
        tts_service=tts,
    )
    agent.context = context

    server = agent._server_with_manager()
    task = agent.build_pipeline("demo")
    task.start()
    await asyncio.sleep(0.05)

    # A browser joins the room.
    ws = FakeWebSocket()
    await server.handle_message(ws, {"type": "join_room", "room_id": "demo"})

    # It streams some audio.
    pcm = base64.b64encode(b"\x00\x01" * 160).decode("ascii")
    await server.handle_message(
        ws,
        {"type": "audio_pcm", "audio_base64": pcm, "sample_rate": 16000, "channels": 1},
    )
    await asyncio.sleep(0.05)

    # The browser stops speaking — that closes the turn.
    await server.handle_message(
        ws, {"type": "participant_activity", "activity": "speaking_stopped"}
    )
    await asyncio.sleep(0.3)

    # The STT provider received the audio.
    assert stt.pushed, "the agent must receive the browser's audio"

    # The transcript reached the LLM context.
    messages = context.messages
    assert any(m.role == "user" and m.content == "what is the weather" for m in messages)

    # The agent produced audio.
    assert tts.synthesised, "the agent must synthesise a reply"
    assert "It is sunny." in " ".join(tts.synthesised)

    await task.cancel()


@pytest.mark.asyncio
async def test_agent_reply_is_published_to_the_room():
    """The agent's audio must reach the browser over the room."""
    context = LLMContext("You are a test agent.")
    agent = Agent(
        name="Tester",
        stt_provider=FakeSttProvider(text="hi"),
        llm_service=FakeLlmService(context, reply="Hello back."),
        tts_service=FakeTtsService(),
    )
    agent.context = context

    server = agent._server_with_manager()
    task = agent.build_pipeline("demo")
    task.start()
    await asyncio.sleep(0.05)

    ws = FakeWebSocket()
    await server.handle_message(ws, {"type": "join_room", "room_id": "demo"})

    await server.handle_message(
        ws, {"type": "participant_activity", "activity": "speaking_stopped"}
    )
    await asyncio.sleep(0.3)

    room_audio = [m for m in ws.sent if m.get("type") == "room_audio"]
    assert room_audio, "the agent's audio must be published to the room"
    assert room_audio[0]["mime_type"] == "audio/pcm"

    await task.cancel()
