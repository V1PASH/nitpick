"""Frame payloads and the :class:`Frame` envelope."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

from .kinds import FrameKind, UNINTERRUPTIBLE_KINDS

_frame_ids = itertools.count(1)


def next_frame_id() -> int:
    """Return a process-unique frame id."""
    return next(_frame_ids)


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------


@dataclass
class StartPayload:
    """Parameters every processor receives with the ``START`` frame."""

    allow_interruptions: bool = False
    enable_metrics: bool = False
    enable_usage_metrics: bool = False
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class ErrorPayload:
    """A non-fatal or fatal error raised by a processor."""

    error: str
    fatal: bool = False
    processor_name: str | None = None


@dataclass
class Transcription:
    """A transcript produced by an STT provider."""

    text: str
    user_id: str = "user"
    timestamp: str = ""
    language: str | None = None
    finalized: bool = False

    def with_language(self, language: str) -> "Transcription":
        self.language = language
        return self

    def finalized_(self) -> "Transcription":
        self.finalized = True
        return self


@dataclass
class AudioChunk:
    """A chunk of raw PCM audio."""

    data: bytes
    sample_rate: int
    num_channels: int = 1
    num_frames: int = 0
    transport_source: str | None = None

    def __post_init__(self) -> None:
        if not self.num_frames and self.num_channels:
            self.num_frames = len(self.data) // (self.num_channels * 2)


@dataclass
class VadStarted:
    """The user started speaking, as attested by VAD."""

    start_secs: float = 0.0
    timestamp: float = 0.0


@dataclass
class VadStopped:
    """The user stopped speaking, as attested by VAD.

    ``transcript`` is bundled *into* this frame on purpose.

    nitpick moves system frames on a separate lane from data frames, and
    system frames always overtake queued data. A transcription and a
    "user stopped" signal emitted as two separate frames will therefore be
    **reordered** — the stop arrives first, the aggregator sees a turn close
    with no text, and the agent silently never answers.

    Carrying the transcript inside the stop frame makes the ordering
    structural instead of probabilistic: one frame cannot be reordered
    against itself.
    """

    stop_secs: float = 0.0
    timestamp: float = 0.0
    transcript: Transcription | None = None


@dataclass
class FunctionCall:
    """A tool invocation requested by the model."""

    id: str
    function_name: str
    arguments: str = ""


@dataclass
class FunctionCallResult:
    """The summarised result of a tool invocation — this is what the LLM sees."""

    id: str
    function_name: str
    result: str


@dataclass
class FunctionCallRawResult:
    """The full structured result of a tool invocation.

    The LLM never sees this frame; downstream processors (logging, UI) do.
    """

    id: str
    function_name: str
    raw_data: Any = None


# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------


@dataclass
class Frame:
    """A typed message travelling through the pipeline."""

    kind: FrameKind
    payload: Any = None
    id: int = field(default_factory=next_frame_id)
    sibling_id: int | None = None

    # ---- Classification ----------------------------------------------------

    @property
    def is_system(self) -> bool:
        return self.kind.is_system

    @property
    def is_uninterruptible(self) -> bool:
        """True for frames that must survive an interruption drain."""
        return self.kind in UNINTERRUPTIBLE_KINDS

    # ---- Derivation --------------------------------------------------------

    def with_new_id(self) -> "Frame":
        return Frame(kind=self.kind, payload=self.payload, sibling_id=self.sibling_id)

    def with_sibling(self, sibling_id: int) -> "Frame":
        self.sibling_id = sibling_id
        return self

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Frame(kind={self.kind.value}, id={self.id}, payload={self.payload!r})"


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------


def frame_start(payload: StartPayload | None = None) -> Frame:
    return Frame(FrameKind.START, payload or StartPayload())


def frame_cancel(reason: str | None = None) -> Frame:
    return Frame(FrameKind.CANCEL, reason)


def frame_stop(reason: str | None = None) -> Frame:
    return Frame(FrameKind.STOP, reason)


def frame_end(reason: str | None = None) -> Frame:
    return Frame(FrameKind.END, reason)


def frame_error(message: str, fatal: bool = False, processor_name: str | None = None) -> Frame:
    return Frame(FrameKind.ERROR, ErrorPayload(message, fatal, processor_name))


def frame_interruption() -> Frame:
    return Frame(FrameKind.INTERRUPTION)


def frame_heartbeat(timestamp: float) -> Frame:
    return Frame(FrameKind.HEARTBEAT, timestamp)


def frame_transcription(transcription: Transcription) -> Frame:
    return Frame(FrameKind.TRANSCRIPTION, transcription)


def frame_llm_text(text: str) -> Frame:
    return Frame(FrameKind.LLM_TEXT, text)


def frame_input_audio(chunk: AudioChunk) -> Frame:
    return Frame(FrameKind.INPUT_AUDIO_RAW, chunk)


def frame_output_audio(chunk: AudioChunk) -> Frame:
    return Frame(FrameKind.OUTPUT_AUDIO_RAW, chunk)


def frame_vad_started(start_secs: float = 0.0, timestamp: float = 0.0) -> Frame:
    return Frame(FrameKind.VAD_USER_STARTED_SPEAKING, VadStarted(start_secs, timestamp))


def frame_vad_stopped(
    stop_secs: float = 0.0,
    timestamp: float = 0.0,
    transcript: Transcription | None = None,
) -> Frame:
    return Frame(FrameKind.VAD_USER_STOPPED_SPEAKING, VadStopped(stop_secs, timestamp, transcript))


def frame_client_vad_started(timestamp: float = 0.0) -> Frame:
    return Frame(FrameKind.CLIENT_VAD_USER_STARTED_SPEAKING, timestamp)


def frame_client_vad_stopped(timestamp: float = 0.0) -> Frame:
    return Frame(FrameKind.CLIENT_VAD_USER_STOPPED_SPEAKING, timestamp)


def frame_llm_full_response_start() -> Frame:
    return Frame(FrameKind.LLM_FULL_RESPONSE_START)


def frame_llm_full_response_end() -> Frame:
    return Frame(FrameKind.LLM_FULL_RESPONSE_END)


def frame_function_call_start() -> Frame:
    return Frame(FrameKind.FUNCTION_CALL_START)


def frame_function_call_end() -> Frame:
    return Frame(FrameKind.FUNCTION_CALL_END)


def frame_function_call_in_progress(call: FunctionCall) -> Frame:
    return Frame(FrameKind.FUNCTION_CALL_IN_PROGRESS, call)


def frame_function_call_result(result: FunctionCallResult) -> Frame:
    return Frame(FrameKind.FUNCTION_CALL_RESULT, result)


def frame_function_call_raw_result(result: FunctionCallRawResult) -> Frame:
    return Frame(FrameKind.FUNCTION_CALL_RAW_RESULT, result)
