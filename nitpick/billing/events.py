"""Billing events — what a session costs.

The event vocabulary is deliberately provider-agnostic. Each service
reports what it consumed; the storage layer decides what that is worth.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TranscriptRole(str, Enum):
    """Who said a transcript line."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass(frozen=True)
class TranscriptEntry:
    """One line of the conversation transcript."""

    text: str
    role: TranscriptRole
    timestamp: float = field(default_factory=time.time)
    turn_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BillingEvent:
    """A single metered event.

    This is a tagged union flattened into one dataclass. Every variant
    carries ``session_id`` so events can be interleaved from concurrent
    sessions without confusion.
    """

    session_id: str
    kind: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Session lifecycle
    started_at: float | None = None
    ended_at: float | None = None
    finish_reason: str | None = None

    # LLM usage
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float | None = None

    # TTS usage
    characters: int | None = None
    characters_by_model: dict[str, int] | None = None

    # STT usage
    provider: str | None = None
    audio_seconds: float | None = None

    # Transcript
    transcript: TranscriptEntry | None = None

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable view, with ``None`` fields dropped."""
        payload: dict[str, Any] = {
            "session_id": self.session_id,
            "kind": self.kind,
            "timestamp": self.timestamp,
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        for name in (
            "started_at",
            "ended_at",
            "finish_reason",
            "model",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "latency_ms",
            "characters",
            "characters_by_model",
            "provider",
            "audio_seconds",
        ):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        if self.transcript is not None:
            payload["transcript"] = {
                "text": self.transcript.text,
                "role": self.transcript.role.value,
                "timestamp": self.transcript.timestamp,
                "turn_id": self.transcript.turn_id,
                **({"metadata": self.transcript.metadata} if self.transcript.metadata else {}),
            }
        return payload
