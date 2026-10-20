"""Tests for the VAD: the state machine, the energy backend, and the processor."""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from nitpick.frames import (
    DOWNSTREAM,
    AudioChunk,
    FrameKind,
    FrameProcessor,
    frame_input_audio,
)
from nitpick.pipeline import PipelineParams, PipelineTask
from nitpick.vad import (
    EnergyVad,
    StateMachine,
    VadParams,
    VadProcessor,
    VadState,
    calculate_audio_volume,
)


def pcm(samples: np.ndarray) -> bytes:
    return samples.astype("<i2").tobytes()


def speech(samples: int, amplitude: int = 6000) -> bytes:
    t = np.arange(samples) / 16_000.0
    wave = (np.sin(2 * np.pi * 220.0 * t) * amplitude).astype(np.int16)
    return pcm(wave)


def silence(samples: int) -> bytes:
    return pcm(np.zeros(samples, dtype=np.int16))


# ---------------------------------------------------------------------------
# The state machine
# ---------------------------------------------------------------------------


def test_state_machine_opens_and_closes_a_turn():
    machine = StateMachine(16_000, VadParams(confidence=0.5, start_secs=0.2, stop_secs=0.8))

    # A window is 512 samples at 16 kHz — 32 ms. So 0.2 s of speech is
    # seven windows and 0.8 s of silence is twenty-five.
    for _ in range(machine.start_frames):
        machine.advance(0.9, speech(512))
    assert machine.state is VadState.SPEAKING

    for _ in range(machine.stop_frames):
        machine.advance(0.0, silence(512))
    assert machine.state is VadState.QUIET


def test_state_machine_debounces_a_short_noise_burst():
    machine = StateMachine(16_000, VadParams(confidence=0.5, start_secs=0.2, stop_secs=0.8))

    machine.advance(0.9, speech(512))
    assert machine.state is VadState.STARTING, "one window is not enough to open a turn"

    machine.advance(0.0, silence(512))
    assert machine.state is VadState.QUIET, "a brief burst must not open a turn"


def test_volume_of_silence_is_zero_and_loud_audio_is_high():
    assert calculate_audio_volume(silence(512)) < 0.01
    assert calculate_audio_volume(speech(512, amplitude=30000)) > 0.8


def test_empty_volume_is_zero():
    assert calculate_audio_volume(b"") == 0.0


# ---------------------------------------------------------------------------
# The energy backend
# ---------------------------------------------------------------------------


def test_energy_vad_ignores_silence():
    vad = EnergyVad(16_000)
    for _ in range(5):
        assert vad.is_speech(silence(512)) == 0.0


def test_energy_vad_detects_speech_above_the_noise_floor():
    vad = EnergyVad(16_000)
    # Seed the noise floor with quiet room tone.
    for _ in range(5):
        vad.is_speech(speech(512, amplitude=100))

    confident = vad.is_speech(speech(512, amplitude=8000))
    assert confident > 0.5, f"speech should be detected, got {confident}"


# ---------------------------------------------------------------------------
# The WebRTC VAD backend
# ---------------------------------------------------------------------------


def test_webrtc_vad_detects_speech():
    """The backend must work without pkg_resources.

    The ``webrtcvad`` wrapper package imports ``pkg_resources`` for a
    cosmetic version string, and recent setuptools releases no longer
    ship it — which Python 3.12+ virtualenvs do not install by default.
    nitpick calls the C extension directly to avoid that.
    """
    import importlib.util

    if importlib.util.find_spec("_webrtcvad") is None:
        pytest.skip("the webrtcvad C extension is not installed")

    from nitpick.vad import WebRtcVad

    vad = WebRtcVad(16_000)

    # The WebRTC VAD needs a moment to settle, so feed a few frames.
    for _ in range(3):
        vad.is_speech(silence(320))
    assert vad.is_speech(silence(320)) == 0.0

    for _ in range(3):
        assert vad.is_speech(speech(320, amplitude=12000)) == 1.0


def test_webrtc_vad_rejects_an_unsupported_sample_rate():
    import importlib.util

    if importlib.util.find_spec("_webrtcvad") is None:
        pytest.skip("the webrtcvad C extension is not installed")

    from nitpick.vad import WebRtcVad

    with pytest.raises(ValueError):
        WebRtcVad(44_100)


def test_webrtc_vad_needs_no_pkg_resources():
    """Importing nitpick's backend must not pull in pkg_resources."""
    import importlib.util
    import subprocess
    import sys

    if importlib.util.find_spec("_webrtcvad") is None:
        pytest.skip("the webrtcvad C extension is not installed")

    code = (
        "import sys;"
        "import nitpick.vad.backends as b;"
        "b.WebRtcVad(16000);"
        "print('pkg_resources' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False", (
        "the backend must not import pkg_resources"
    )


# ---------------------------------------------------------------------------
# The processor
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vad_processor_emits_turn_boundaries():
    vad = EnergyVad(16_000)
    processor = VadProcessor(vad, params=VadParams(confidence=0.5, start_secs=0.1, stop_secs=0.1))
    seen: list[FrameKind] = []

    class Capture:
        async def on_process_frame(self, processor, frame, direction):
            seen.append(frame.kind)
            await processor.push_frame(frame, direction)

    task = PipelineTask([processor, FrameProcessor("capture", Capture())], PipelineParams())
    task.start()
    await asyncio.sleep(0.05)

    # Loud audio, then silence — the turn opens on the speech and closes
    # on the silence.
    await task.push_frame(
        frame_input_audio(
            AudioChunk(data=speech(16_000), sample_rate=16_000)
        ),
        DOWNSTREAM,
    )
    await asyncio.sleep(0.1)

    await task.push_frame(
        frame_input_audio(
            AudioChunk(data=silence(16_000), sample_rate=16_000)
        ),
        DOWNSTREAM,
    )
    await asyncio.sleep(0.2)

    assert FrameKind.VAD_USER_STARTED_SPEAKING in seen
    assert FrameKind.VAD_USER_STOPPED_SPEAKING in seen

    await task.cancel()


@pytest.mark.asyncio
async def test_vad_processor_forwards_the_audio_it_analyses():
    """The STT downstream needs the same samples the VAD looked at."""
    vad = EnergyVad(16_000)
    processor = VadProcessor(vad)
    audio: list[bytes] = []

    class Capture:
        async def on_process_frame(self, processor, frame, direction):
            if frame.kind is FrameKind.INPUT_AUDIO_RAW:
                audio.append(frame.payload.data)
            await processor.push_frame(frame, direction)

    task = PipelineTask([processor, FrameProcessor("capture", Capture())], PipelineParams())
    task.start()
    await asyncio.sleep(0.05)

    chunk = speech(1024)
    await task.push_frame(
        frame_input_audio(AudioChunk(data=chunk, sample_rate=16_000)), DOWNSTREAM
    )
    await asyncio.sleep(0.1)

    assert audio, "the VAD must forward the audio it analysed"
    assert audio[0] == chunk

    await task.cancel()
