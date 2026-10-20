"""Voice Activity Detection.

Turn detection is the part of a voice pipeline that decides *when* the user
started and stopped talking. Getting it wrong is painful in both
directions: too eager and the agent interrupts itself mid-sentence; too
lax and the agent answers after a long pause, or never answers at all.

nitpick ships two backends behind one API:

``WebRtcVad``
    The default. A small, dependency-light energy/gamut model. Nothing to
    download, works offline straight after ``pip install``.

``SileroVad``
    A neural model (ONNX) that is markedly better at separating speech from
    noise. Requires the ``onnx`` extra and a model file.

Both feed the same state machine, so switching backends changes accuracy,
not behaviour.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


@dataclass
class VadParams:
    """Tunables for the VAD state machine.

    The defaults were tuned against real speech (browser microphone audio
    over Opus) rather than chosen arbitrarily:

    ``confidence``
        The speech probability above which a window counts as speech.

    ``start_secs``
        How much *unbroken* speech is needed before a turn opens. This
        matters more than it looks: after a browser's audio processing and
        Opus, neural confidence rarely stays above threshold for more than
        ~0.36 s, so a 0.4 s threshold meant turns often never opened and
        most of a speaker's speech never reached the STT.

    ``stop_secs``
        How much silence ends a turn. 0.8 s ends turns at sentence pauses
        rather than at every breath.

    ``min_volume``
        A loudness floor, so quiet noise never opens a turn.
    """

    confidence: float = 0.5
    start_secs: float = 0.2
    stop_secs: float = 0.8
    min_volume: float = 0.2


class VadState(str, Enum):
    """Where the VAD is in a turn."""

    QUIET = "quiet"
    STARTING = "starting"
    SPEAKING = "speaking"
    STOPPING = "stopping"


@runtime_checkable
class VadBackend(Protocol):
    """A speech probability source.

    Implementations answer one question: given a window of 16-bit mono PCM,
    how likely is it to contain speech?
    """

    #: The sample rate the backend expects.
    sample_rate: int

    def is_speech(self, window: bytes) -> float:
        """Return the speech probability for a window of PCM."""


def calculate_audio_volume(audio: bytes) -> float:
    """Normalised loudness of a PCM window, on a 0.0–1.0 scale.

    Uses dBFS rather than a full EBU R128 meter: silence maps to ~0.0,
    full scale to 1.0, and normal conversational speech to ~0.3–0.6.
    """
    if not audio:
        return 0.0

    count = len(audio) // 2
    if count == 0:
        return 0.0

    rms = _rms_samples(audio)
    if rms < 1e-9:
        return 0.0

    db = max(-60.0, min(0.0, 20.0 * math.log10(rms)))
    return (db + 60.0) / 60.0


def _rms_samples(audio: bytes) -> float:
    """Root-mean-square of 16-bit mono PCM, normalised to 0..1.

    Uses numpy when it is available (it is a core dependency) because this
    runs on every audio window.
    """
    import numpy as np

    samples = np.frombuffer(audio, dtype="<i2").astype(np.float32)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples * samples)) / 32768.0)


def exp_smoothing(current: float, previous: float, factor: float) -> float:
    """Exponential smoothing: ``prev + factor * (value - prev)``."""
    return current * factor + previous * (1.0 - factor)


class StateMachine:
    """Turn detection over a stream of speech probabilities.

    Four states: ``QUIET → STARTING → SPEAKING → STOPPING → QUIET``. The
    machine debounces both edges so a single cough does not open a turn and
    a pause between words does not close one.
    """

    SMOOTHING_FACTOR = 0.2

    def __init__(self, sample_rate: int, params: VadParams | None = None) -> None:
        self.params = params or VadParams()
        self.sample_rate = sample_rate

        # 512 samples at 16 kHz, 256 at 8 kHz — the window the backends use.
        self.frames_required = 512 if sample_rate == 16_000 else 256
        self.bytes_required = self.frames_required * 2

        frames_per_sec = self.frames_required / sample_rate
        self.start_frames = max(1, round(self.params.start_secs / frames_per_sec))
        self.stop_frames = max(1, round(self.params.stop_secs / frames_per_sec))

        self.starting_count = 0
        self.stopping_count = 0
        self.prev_volume = 0.0
        self._primed = False
        self._buffer = bytearray()
        self.state = VadState.QUIET

    # ------------------------------------------------------------------
    # Windowing
    # ------------------------------------------------------------------

    def next_window(self, chunk: bytes) -> bytes | None:
        """Buffer a chunk and take the next complete window, if any.

        Callers **must drain in a loop** — one call yields at most one
        window. A single call per audio frame keeps up only while frames are
        no wider than one window. Feed a wider frame and the excess stays
        buffered, growing without bound, so the VAD drifts ever further
        behind the live audio.
        """
        self._buffer.extend(chunk)
        if len(self._buffer) >= self.bytes_required:
            window = bytes(self._buffer[: self.bytes_required])
            del self._buffer[: self.bytes_required]
            return window
        return None

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------

    def advance(self, confidence: float, window: bytes) -> VadState:
        """Feed a speech probability and return the new state."""
        volume = exp_smoothing(
            calculate_audio_volume(window),
            self.prev_volume,
            self.SMOOTHING_FACTOR,
        )
        if not self._primed:
            # The smoother starts from silence, so its first output is a
            # fraction of the real level — which can push a loud frame
            # under the volume floor and lose it. Calibrate on the first
            # frame instead of smoothing down from zero.
            volume = calculate_audio_volume(window)
            self._primed = True
        self.prev_volume = volume

        speaking = (
            confidence >= self.params.confidence
            and volume >= self.params.min_volume
        )

        if speaking:
            if self.state is VadState.QUIET:
                self.state = VadState.STARTING
                self.starting_count = 1
            elif self.state is VadState.STARTING:
                self.starting_count += 1
            elif self.state is VadState.STOPPING:
                self.state = VadState.SPEAKING
                self.stopping_count = 0
        else:
            if self.state is VadState.STARTING:
                self.state = VadState.QUIET
                self.starting_count = 0
            elif self.state is VadState.SPEAKING:
                self.state = VadState.STOPPING
                self.stopping_count = 1
            elif self.state is VadState.STOPPING:
                self.stopping_count += 1

        if self.state is VadState.STARTING and self.starting_count >= self.start_frames:
            self.state = VadState.SPEAKING
            self.starting_count = 0

        if self.state is VadState.STOPPING and self.stopping_count >= self.stop_frames:
            self.state = VadState.QUIET
            self.stopping_count = 0

        return self.state
