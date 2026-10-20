"""Tests for billing, interruption handling, the SDK, and the Sarvam URL."""
from __future__ import annotations

import asyncio

import pytest

from nitpick.billing import MemoryBillingStorage, SessionBilling, TranscriptRole
from nitpick.frames import (
    DOWNSTREAM,
    Frame,
    FrameKind,
    FrameProcessor,
    frame_vad_started,
)
from nitpick.pipeline import PipelineParams, PipelineTask
from nitpick.processors.interruption import InterruptionHandler
from nitpick.services.stt.sarvam import SarvamSttProvider
from nitpick.transport import SpeakingGate


# ---------------------------------------------------------------------------
# Billing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_billing_records_without_blocking():
    collector = SessionBilling("session-1")
    storage = MemoryBillingStorage()
    collector.attach(storage)
    collector.start()

    collector.record_session_start({"user_id": "u-42"})
    collector.record_llm_usage(model="gpt-4o-mini", prompt_tokens=10, completion_tokens=5)
    collector.record_tts_usage(characters=42, model="gpt-4o-mini-tts")
    collector.record_stt_usage(provider="openai", audio_seconds=3.5)
    collector.record_transcript("hello", TranscriptRole.USER)

    await collector.close()

    kinds = [event.kind for event in storage.events]
    assert kinds == [
        "session_start",
        "llm_usage",
        "tts_usage",
        "stt_usage",
        "transcript",
    ]


@pytest.mark.asyncio
async def test_billing_records_the_token_total():
    collector = SessionBilling("session-2")
    storage = MemoryBillingStorage()
    collector.attach(storage)
    collector.start()

    collector.record_llm_usage(model="m", prompt_tokens=10, completion_tokens=5)
    await collector.close()

    event = storage.events[0]
    assert event.total_tokens == 15
    assert event.to_dict()["total_tokens"] == 15


@pytest.mark.asyncio
async def test_billing_survives_a_failing_storage():
    class Broken:
        def write(self, event):
            raise RuntimeError("disk on fire")

    collector = SessionBilling("session-3")
    collector.attach(Broken())
    collector.start()
    collector.record_session_start()
    # Must not raise.
    await collector.close()


# ---------------------------------------------------------------------------
# The speaking gate — client and server VAD coordination
# ---------------------------------------------------------------------------


def test_speaking_gate_emits_one_start_per_turn():
    gate = SpeakingGate()
    assert gate.claim_start() is True
    assert gate.claim_start() is False, "a second start must be suppressed"
    assert gate.claim_stop() is True
    assert gate.claim_stop() is False, "a second stop must be suppressed"


def test_speaking_gate_emits_a_stop_without_a_start():
    """A stop with no matching start still has to close the turn."""
    gate = SpeakingGate()
    assert gate.claim_stop() is True


# ---------------------------------------------------------------------------
# Interruption handling (barge-in)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_user_speech_interrupts_the_bot():
    handler = InterruptionHandler()
    interruptions: list[Frame] = []

    class Capture:
        async def on_process_frame(self, processor, frame, direction):
            if frame.kind is FrameKind.INTERRUPTION:
                interruptions.append(frame)
            await processor.push_frame(frame, direction)

    task = PipelineTask([handler, FrameProcessor("capture", Capture())], PipelineParams())
    task.start()
    await asyncio.sleep(0.05)

    # The bot starts talking.
    await task.push_frame(Frame(FrameKind.BOT_STARTED_SPEAKING), DOWNSTREAM)
    await asyncio.sleep(0.05)

    # The user starts speaking.
    await task.push_frame(frame_vad_started(), DOWNSTREAM)
    await asyncio.sleep(0.1)

    assert interruptions, "the user must interrupt the bot"
    assert handler.interruption_count == 1

    await task.cancel()


@pytest.mark.asyncio
async def test_user_speech_does_not_interrupt_a_silent_bot():
    handler = InterruptionHandler()
    interruptions: list[Frame] = []

    class Capture:
        async def on_process_frame(self, processor, frame, direction):
            if frame.kind is FrameKind.INTERRUPTION:
                interruptions.append(frame)
            await processor.push_frame(frame, direction)

    task = PipelineTask([handler, FrameProcessor("capture", Capture())], PipelineParams())
    task.start()
    await asyncio.sleep(0.05)

    await task.push_frame(frame_vad_started(), DOWNSTREAM)
    await asyncio.sleep(0.1)

    assert not interruptions, "there is nothing to interrupt"

    await task.cancel()


@pytest.mark.asyncio
async def test_interruptions_can_be_disabled():
    handler = InterruptionHandler(enabled=False)
    interruptions: list[Frame] = []

    class Capture:
        async def on_process_frame(self, processor, frame, direction):
            if frame.kind is FrameKind.INTERRUPTION:
                interruptions.append(frame)
            await processor.push_frame(frame, direction)

    task = PipelineTask([handler, FrameProcessor("capture", Capture())], PipelineParams())
    task.start()
    await asyncio.sleep(0.05)

    await task.push_frame(Frame(FrameKind.BOT_STARTED_SPEAKING), DOWNSTREAM)
    await task.push_frame(frame_vad_started(), DOWNSTREAM)
    await asyncio.sleep(0.1)

    assert not interruptions

    await task.cancel()


# ---------------------------------------------------------------------------
# The Sarvam provider
# ---------------------------------------------------------------------------


def test_sarvam_url_carries_the_right_parameters():
    provider = SarvamSttProvider(
        "key", model="saaras:v3", language="ml-IN", mode="transcribe"
    )
    url = provider.url()

    assert url.startswith("wss://api.sarvam.ai/speech-to-text/ws?")
    assert "model=saaras%3Av3" in url
    assert "sample_rate=16000" in url
    assert "input_audio_codec=pcm" in url
    assert "flush_signal=true" in url
    # The language parameter is hyphenated — Sarvam rejects the underscore.
    assert "language-code=ml-IN" in url
    assert "mode=transcribe" in url


def test_sarvam_translate_model_uses_the_translate_endpoint():
    provider = SarvamSttProvider("key", model="saaras:v2.5", language="ml-IN")
    url = provider.url()

    assert url.startswith("wss://api.sarvam.ai/speech-to-text-translate/ws?")
    # The translate endpoint takes no language code.
    assert "language-code" not in url


def test_sarvam_mode_is_only_sent_for_v3():
    provider = SarvamSttProvider("key", model="saarika:v2.5", mode="transcribe")
    assert "mode=" not in provider.url()
