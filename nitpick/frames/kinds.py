"""Flat frame taxonomy.

Every frame that flows through a nitpick pipeline carries a :class:`FrameKind`.
Keeping the taxonomy flat (rather than a class hierarchy) means dispatch is a
single attribute comparison instead of an ``isinstance`` chain, and every frame
kind is discoverable from one enum.

The three families mirror the ones Pipecat and rustvani use:

``SYSTEM``
    Lifecycle and control-of-the-stream frames — start, cancel, interruption,
    speaking signals, VAD signals, raw audio input.

``CONTROL``
    Boundaries around a logical unit of work — the start/end of an LLM
    response, the start/end of a tool-call round.

``DATA``
    Payloads — transcriptions, LLM text, audio output, tool-call results.
"""
from __future__ import annotations

from enum import Enum


class FrameKind(str, Enum):
    """Every kind of frame nitpick understands."""

    # ---- System: lifecycle -------------------------------------------------
    START = "start"
    CANCEL = "cancel"
    ERROR = "error"
    INTERRUPTION = "interruption"
    STOP = "stop"
    END_TASK = "end_task"
    CANCEL_TASK = "cancel_task"
    STOP_TASK = "stop_task"
    INTERRUPTION_TASK = "interruption_task"

    # ---- System: speaking signals -----------------------------------------
    BOT_SPEAKING = "bot_speaking"
    USER_SPEAKING = "user_speaking"
    BOT_STARTED_SPEAKING = "bot_started_speaking"
    BOT_STOPPED_SPEAKING = "bot_stopped_speaking"
    USER_STARTED_SPEAKING = "user_started_speaking"
    USER_STOPPED_SPEAKING = "user_stopped_speaking"
    VAD_USER_STARTED_SPEAKING = "vad_user_started_speaking"
    VAD_USER_STOPPED_SPEAKING = "vad_user_stopped_speaking"
    CLIENT_VAD_USER_STARTED_SPEAKING = "client_vad_user_started_speaking"
    CLIENT_VAD_USER_STOPPED_SPEAKING = "client_vad_user_stopped_speaking"

    # ---- System: audio input ----------------------------------------------
    INPUT_AUDIO_RAW = "input_audio_raw"

    # ---- System: processor control -----------------------------------------
    PAUSE_PROCESSOR = "pause_processor"
    RESUME_PROCESSOR = "resume_processor"

    # ---- System: health ----------------------------------------------------
    HEARTBEAT = "heartbeat"

    # ---- Control: pipeline lifecycle --------------------------------------
    END = "end"

    # ---- Control: LLM response boundaries ----------------------------------
    LLM_FULL_RESPONSE_START = "llm_full_response_start"
    LLM_FULL_RESPONSE_END = "llm_full_response_end"

    # ---- Control: function-call boundaries ---------------------------------
    FUNCTION_CALL_START = "function_call_start"
    FUNCTION_CALL_END = "function_call_end"

    # ---- Data ---------------------------------------------------------------
    DATA = "data"
    TRANSCRIPTION = "transcription"
    LLM_TEXT = "llm_text"
    LLM_CONTEXT = "llm_context"
    OUTPUT_AUDIO_RAW = "output_audio_raw"
    FUNCTION_CALL_IN_PROGRESS = "function_call_in_progress"
    FUNCTION_CALL_RESULT = "function_call_result"
    FUNCTION_CALL_RAW_RESULT = "function_call_raw_result"

    # ---- Helpers ------------------------------------------------------------

    @property
    def is_system(self) -> bool:
        """True for lifecycle/control frames that must not be queued behind data."""
        return self in _SYSTEM_KINDS

    @property
    def is_control(self) -> bool:
        """True for frames that mark a boundary rather than carrying a payload."""
        return self in _CONTROL_KINDS

    @property
    def is_data(self) -> bool:
        """True for payload frames."""
        return self in _DATA_KINDS

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


_SYSTEM_KINDS = frozenset(
    {
        FrameKind.START,
        FrameKind.CANCEL,
        FrameKind.ERROR,
        FrameKind.INTERRUPTION,
        FrameKind.STOP,
        FrameKind.END_TASK,
        FrameKind.CANCEL_TASK,
        FrameKind.STOP_TASK,
        FrameKind.INTERRUPTION_TASK,
        FrameKind.BOT_SPEAKING,
        FrameKind.USER_SPEAKING,
        FrameKind.BOT_STARTED_SPEAKING,
        FrameKind.BOT_STOPPED_SPEAKING,
        FrameKind.USER_STARTED_SPEAKING,
        FrameKind.USER_STOPPED_SPEAKING,
        FrameKind.VAD_USER_STARTED_SPEAKING,
        FrameKind.VAD_USER_STOPPED_SPEAKING,
        FrameKind.CLIENT_VAD_USER_STARTED_SPEAKING,
        FrameKind.CLIENT_VAD_USER_STOPPED_SPEAKING,
        FrameKind.INPUT_AUDIO_RAW,
        FrameKind.PAUSE_PROCESSOR,
        FrameKind.RESUME_PROCESSOR,
        FrameKind.HEARTBEAT,
    }
)

_CONTROL_KINDS = frozenset(
    {
        FrameKind.END,
        FrameKind.LLM_FULL_RESPONSE_START,
        FrameKind.LLM_FULL_RESPONSE_END,
        FrameKind.FUNCTION_CALL_START,
        FrameKind.FUNCTION_CALL_END,
    }
)

_DATA_KINDS = frozenset(
    {
        FrameKind.DATA,
        FrameKind.TRANSCRIPTION,
        FrameKind.LLM_TEXT,
        FrameKind.LLM_CONTEXT,
        FrameKind.OUTPUT_AUDIO_RAW,
        FrameKind.FUNCTION_CALL_IN_PROGRESS,
        FrameKind.FUNCTION_CALL_RESULT,
        FrameKind.FUNCTION_CALL_RAW_RESULT,
    }
)

#: Frames that must survive an interruption drain. Interrupting the bot is
#: important; interrupting a shutdown is a bug.
UNINTERRUPTIBLE_KINDS = frozenset(
    {
        FrameKind.END,
        FrameKind.END_TASK,
        FrameKind.CANCEL_TASK,
        FrameKind.STOP_TASK,
    }
)
