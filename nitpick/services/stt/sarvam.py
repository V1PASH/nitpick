"""Sarvam AI speech-to-text.

Sarvam's streaming API is the strongest option for Indian languages —
Malayalam, Hindi, Tamil and more — and its WebSocket protocol suits a
voice pipeline well: audio goes up continuously, a ``flush`` signal closes
the segment.

Protocol summary
----------------
* Endpoint: ``wss://api.sarvam.ai/speech-to-text/ws``
* Auth: the ``api-subscription-key`` header (lowercase — Sarvam's SDK
  uses that exact spelling)
* Query: ``model``, ``sample_rate``, ``input_audio_codec``,
  ``flush_signal=true``, and optionally ``language-code``, ``mode``,
  ``prompt``
* Flush: the text message ``{"type": "flush"}``
* Results: ``{"type": "data", "data": {"transcript": ..., "language_code": ...}}``

Sarvam's streaming API returns finals only.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any
from urllib.parse import quote

from .base import EventCallback, SttEvent, SttProvider

log = logging.getLogger(__name__)

BASE_URL = "wss://api.sarvam.ai"
STT_PATH = "/speech-to-text/ws"
TRANSLATE_PATH = "/speech-to-text-translate/ws"

#: Models that use the legacy translation endpoint.
TRANSLATE_MODELS = frozenset({"saaras:v2.5"})
#: Models that accept the ``mode`` parameter.
MODE_MODELS = frozenset({"saaras:v3"})


class SarvamSttProvider(SttProvider):
    """Streaming transcription via Sarvam AI."""

    name = "sarvam"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = "saaras:v3",
        language: str | None = None,
        mode: str | None = "transcribe",
        prompt: str | None = None,
        sample_rate: int = 16_000,
        encoding: str = "pcm",
        high_vad_sensitivity: bool = False,
    ) -> None:
        self.api_key = api_key or os.getenv("SARVAM_API_KEY") or os.getenv(
            "SARVAM_API_SUBSCRIPTION_KEY"
        )
        self.model = model
        self.language = language
        self.mode = mode
        self.prompt = prompt
        self.sample_rate = sample_rate
        self.encoding = encoding
        self.high_vad_sensitivity = high_vad_sensitivity

        self._socket: Any = None
        self._on_event: EventCallback | None = None
        self._reader: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # URL construction
    # ------------------------------------------------------------------

    @property
    def path(self) -> str:
        if self.model in TRANSLATE_MODELS:
            return TRANSLATE_PATH
        return STT_PATH

    def url(self) -> str:
        """The full WebSocket URL, query string included."""
        params = [
            f"model={quote(self.model, safe='')}",
            f"sample_rate={self.sample_rate}",
            f"input_audio_codec={quote(self.encoding, safe='')}",
            "flush_signal=true",
        ]

        # The language parameter is hyphenated: ``language-code``.
        if self.language and self.model not in TRANSLATE_MODELS:
            params.append(f"language-code={quote(self.language, safe='')}")

        if self.mode and self.model in MODE_MODELS:
            params.append(f"mode={quote(self.mode, safe='')}")

        if self.prompt:
            params.append(f"prompt={quote(self.prompt, safe='')}")

        if self.high_vad_sensitivity:
            params.append("high_vad_sensitivity=true")

        return f"{BASE_URL}{self.path}?{'&'.join(params)}"

    # ------------------------------------------------------------------
    # Provider lifecycle
    # ------------------------------------------------------------------

    async def start(self, on_event: EventCallback) -> None:
        import websockets

        if not self.api_key:
            raise RuntimeError("SARVAM_API_KEY is not set")

        self._on_event = on_event
        headers = {"api-subscription-key": self.api_key}
        self._socket = await websockets.connect(self.url(), additional_headers=headers)
        self._reader = asyncio.create_task(self._read_loop(), name="sarvam-stt-reader")

    async def push_audio(self, pcm: bytes) -> None:
        if self._socket is None:
            return
        await self._socket.send(pcm)

    async def flush(self) -> None:
        """Close the current segment."""
        if self._socket is None:
            return
        await self._socket.send(json.dumps({"type": "flush"}))

    async def stop(self) -> None:
        if self._reader is not None:
            self._reader.cancel()
            self._reader = None
        if self._socket is not None:
            await self._socket.close()
            self._socket = None

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    async def _read_loop(self) -> None:
        assert self._socket is not None
        try:
            async for raw in self._socket:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                await self._handle_message(raw)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.exception("Sarvam STT reader failed")

    async def _handle_message(self, raw: str) -> None:
        try:
            envelope = json.loads(raw)
        except json.JSONDecodeError:
            log.debug("Sarvam STT: unparseable message")
            return

        if not isinstance(envelope, dict):
            return

        message_type = envelope.get("type")
        if message_type != "data":
            return

        data = envelope.get("data")
        if not isinstance(data, dict):
            return

        text = data.get("transcript")
        language = data.get("language_code") or self.language

        audio_ms = None
        metrics = data.get("metrics")
        if isinstance(metrics, dict):
            duration = metrics.get("audio_duration")
            if isinstance(duration, (int, float)):
                audio_ms = float(duration) * 1000.0

        if isinstance(text, str) and text.strip():
            await self._emit(
                SttEvent(kind="final", text=text, language=language, is_final=True, audio_ms=audio_ms)
            )
        else:
            # An empty transcript still answers the flush — it must close the
            # turn rather than leaving the gate waiting on its timeout.
            await self._emit(SttEvent(kind="final", text="", language=language, audio_ms=audio_ms))

    async def _emit(self, event: SttEvent) -> None:
        if self._on_event is None:
            return
        result = self._on_event(event)
        if asyncio.iscoroutine(result):
            await result
