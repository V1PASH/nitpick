"""VAD backends — the speech probability sources."""
from __future__ import annotations

import logging
import os
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

#: Where model files are cached when they are downloaded on first use.
CACHE_DIR_ENV = "NITPICK_CACHE_DIR"


def cache_dir() -> Path:
    """The directory nitpick caches downloaded models in."""
    override = os.getenv(CACHE_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".nitpick" / "cache"


class EnergyVad:
    """The default backend — energy detection with an adaptive noise floor.

    No dependencies, no model file, works on any Python. It tracks the
    background noise level and calls it speech when the signal rises
    meaningfully above that floor, which is enough for turn detection on a
    reasonably quiet microphone.

    The floor is learned from a history of recent frames rather than a
    single seed value, so a burst of noise does not poison it. Two
    absolute gates sit either side of the adaptive range: below
    ``min_rms`` nothing is speech, and above ``speech_rms`` everything is.

    For noisy environments, use the ``silero`` backend instead — a neural
    model separates speech from noise far better.

    Install with ``pip install nitpick[onnx]`` for that, or
    ``pip install nitpick[webrtcvad]`` for the WebRTC model.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        *,
        threshold: float = 3.0,
        min_rms: float = 0.004,
        speech_rms: float = 0.05,
        history_size: int = 100,
    ) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_rms = min_rms
        self.speech_rms = speech_rms
        self.history_size = history_size
        self._history: deque[float] = deque(maxlen=history_size)

    def is_speech(self, window: bytes) -> float:
        """Return a speech probability for the window."""
        rms = _rms(window)

        if rms <= 0.0:
            return 0.0

        # Absolute gate: quiet is never speech.
        if rms < self.min_rms:
            self._history.append(rms)
            return 0.0

        # Absolute gate: loud is always speech, whatever the noise floor
        # has drifted to.
        if rms >= self.speech_rms:
            self._history.append(rms)
            return 1.0

        self._history.append(rms)
        floor = self._noise_floor()
        if floor <= 0.0:
            return 0.0

        ratio = rms / floor
        if ratio < 1.0:
            return 0.0
        # Map the ratio onto 0..1, reaching 1.0 at the configured threshold.
        confidence = (ratio - 1.0) / max(self.threshold - 1.0, 0.001)
        return max(0.0, min(1.0, confidence))

    def _noise_floor(self) -> float:
        """The 10th percentile of recent frame energies.

        A low percentile is close to "the quietest thing recently heard",
        which is what a noise floor is. A percentile is far more robust
        than a single seed value: one burst of noise cannot raise it much.
        """
        if not self._history:
            return 0.0
        ordered = sorted(self._history)
        index = max(0, int(len(ordered) * 0.1) - 1)
        return max(ordered[index], self.min_rms)


def _rms(window: bytes) -> float:
    """Root-mean-square of a 16-bit mono PCM window, normalised to 0..1."""
    import numpy as np

    samples = np.frombuffer(window, dtype="<i2").astype(np.float32)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples * samples)) / 32768.0)


class WebRtcVad:
    """The WebRTC VAD backend — Google's model, packaged as a C extension.

    Install with ``pip install nitpick[webrtcvad]``. Building it needs a C
    compiler.

    The ``webrtcvad`` wrapper package is not used directly: it imports
    ``pkg_resources`` for a cosmetic ``__version__`` line, and recent
    setuptools releases no longer ship ``pkg_resources`` — which Python
    3.12+ virtualenvs do not install by default. The C extension underneath
    it (``_webrtcvad``) has no such dependency, so nitpick calls that
    directly and only falls back to the wrapper if the extension is
    missing.
    """

    def __init__(self, sample_rate: int = 16_000, aggressiveness: int = 2) -> None:
        if sample_rate not in (8_000, 16_000, 32_000, 48_000):
            raise ValueError(f"webrtcvad does not support {sample_rate} Hz")

        if not 0 <= aggressiveness <= 3:
            raise ValueError("aggressiveness must be between 0 and 3")

        self.sample_rate = sample_rate
        self._backend = _create_webrtc_vad(aggressiveness)

    def is_speech(self, window: bytes) -> float:
        """Return 1.0 when the window contains speech, else 0.0."""
        try:
            is_speech = self._backend.is_speech(window, self.sample_rate)
        except Exception as exc:  # noqa: BLE001 - a bad window must not kill the stream
            log.debug("webrtcvad rejected a window: %s", exc)
            return 0.0
        return 1.0 if is_speech else 0.0


def _create_webrtc_vad(aggressiveness: int) -> Any:
    """Build a WebRTC VAD instance, preferring the bare C extension."""
    try:
        import _webrtcvad
    except ImportError:
        _webrtcvad = None  # type: ignore[assignment]

    if _webrtcvad is not None:
        return _NativeWebRtcVad(_webrtcvad, aggressiveness)

    # The extension is missing — fall back to the wrapper package, which
    # needs pkg_resources.
    import importlib.util

    if importlib.util.find_spec("webrtcvad") is None:
        raise RuntimeError(
            "The webrtcvad backend needs the 'webrtcvad' package. "
            "Install it with: pip install nitpick[webrtcvad]"
        )
    try:
        import webrtcvad
    except ImportError as exc:
        raise RuntimeError(
            "The webrtcvad backend could not be imported "
            f"({exc}). Install 'setuptools' for pkg_resources, or use the "
            "default 'energy' backend instead."
        ) from exc
    return _WrapperWebRtcVad(webrtcvad.Vad(aggressiveness))


class _NativeWebRtcVad:
    """Drives the ``_webrtcvad`` C extension directly."""

    def __init__(self, extension: Any, aggressiveness: int) -> None:
        self._extension = extension
        self._vad = extension.create()
        extension.init(self._vad)
        extension.set_mode(self._vad, aggressiveness)

    def is_speech(self, window: bytes, sample_rate: int) -> bool:
        frames = len(window) // 2
        if frames == 0:
            return False
        return bool(self._extension.process(self._vad, sample_rate, window, frames))


class _WrapperWebRtcVad:
    """Drives the ``webrtcvad`` wrapper package."""

    def __init__(self, vad: Any) -> None:
        self._vad = vad

    def is_speech(self, window: bytes, sample_rate: int) -> bool:
        return bool(self._vad.is_speech(window, sample_rate))


class SileroVad:
    """A neural backend, powered by the Silero VAD ONNX model.

    Noticeably better than WebRTC VAD at separating speech from noise —
    worth the extra dependency if your users are in noisy places.

    Requires the ``onnx`` extra::

        pip install nitpick[onnx]

    The model is looked for in this order:

    1. ``weights_path`` if given
    2. ``silero.onnx`` in the nitpick cache directory
    3. downloaded once into the cache directory
    """

    #: Where the model is fetched from if it is not already present.
    URL = (
        "https://raw.githubusercontent.com/snakers4/silero-vad/master/"
        "src/silero_vad/data/silero_vad.onnx"
    )

    def __init__(
        self,
        sample_rate: int = 16_000,
        *,
        weights_path: str | os.PathLike[str] | None = None,
    ) -> None:
        import importlib.util

        if importlib.util.find_spec("onnxruntime") is None:
            raise RuntimeError(
                "The Silero backend needs the 'onnx' extra. "
                "Install it with: pip install nitpick[onnx]"
            )

        if sample_rate not in (8_000, 16_000):
            raise ValueError("Silero supports 8 kHz and 16 kHz")

        self.sample_rate = sample_rate
        # 512 samples at 16 kHz, 256 at 8 kHz.
        self.num_samples = 512 if sample_rate == 16_000 else 256
        # The model wants the tail of the previous chunk prepended, so the
        # first syllable is not lost.
        self.context_size = 64 if sample_rate == 16_000 else 32

        self._path = self._resolve_weights(weights_path)
        self._session = self._load(self._path)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(self.context_size, dtype=np.float32)

    # ------------------------------------------------------------------

    def _resolve_weights(self, weights_path: str | os.PathLike[str] | None) -> Path:
        if weights_path is not None:
            path = Path(weights_path).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Silero model not found at {path}")
            return path

        cached = cache_dir() / "silero.onnx"
        if cached.exists():
            return cached

        cached.parent.mkdir(parents=True, exist_ok=True)
        log.info("Downloading the Silero VAD model to %s", cached)
        self._download(cached)
        return cached

    @staticmethod
    def _download(destination: Path) -> None:
        import urllib.request

        temporary = destination.with_suffix(".part")
        with urllib.request.urlopen(SileroVad.URL, timeout=120) as response:
            temporary.write_bytes(response.read())
        temporary.replace(destination)

    @staticmethod
    def _load(path: Path) -> Any:
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.log_severity_level = 3
        return onnxruntime.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )

    # ------------------------------------------------------------------

    def is_speech(self, window: bytes) -> float:
        """Return the model's speech probability for the window."""
        import numpy as np

        samples = np.frombuffer(window, dtype="<i2").astype(np.float32) / 32768.0
        if samples.size < self.num_samples:
            samples = np.pad(samples, (0, self.num_samples - samples.size))
        elif samples.size > self.num_samples:
            samples = samples[: self.num_samples]

        # Prepend the tail of the previous chunk: the model uses it as
        # context, and dropping it costs the first syllable of every turn.
        with_context = np.concatenate([self._context, samples])
        frame = with_context.reshape(1, -1)

        input_feed = {
            "input": frame,
            "state": self._state,
            "sr": np.array(self.sample_rate, dtype=np.int64),
        }

        try:
            outputs = self._session.run(None, input_feed)
        except Exception as exc:  # noqa: BLE001
            log.debug("Silero inference failed: %s", exc)
            return 0.0

        # The model returns the new state; keep it for the next call.
        for output in outputs:
            if getattr(output, "shape", None) == (2, 1, 128):
                self._state = output
                break

        # Carry the last context_size samples into the next call.
        self._context = samples[-self.context_size :].copy()

        probability = float(np.asarray(outputs[0]).reshape(-1)[0])
        return max(0.0, min(1.0, probability))
