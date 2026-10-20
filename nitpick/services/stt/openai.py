"""OpenAI speech-to-text.

Uses the batch ``/v1/audio/transcriptions`` endpoint: PCM is accumulated for
the turn and transcribed when the turn ends. This is the most portable
option — it works with a plain ``OPENAI_API_KEY`` and no streaming plumbing.
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import wave

from .base import EventCallback, SttEvent, SttProvider

log = logging.getLogger(__name__)


class OpenAISttProvider(SttProvider):
    """Batch transcription via OpenAI."""

    name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = "gpt-4o-mini-transcribe",
        language: str | None = None,
        sample_rate: int = 16_000,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.language = language
        self.sample_rate = sample_rate
        self.timeout = timeout
        self._buffer = bytearray()
        self._on_event: EventCallback | None = None

    async def start(self, on_event: EventCallback) -> None:
        self._on_event = on_event
        self._buffer.clear()

    async def push_audio(self, pcm: bytes) -> None:
        self._buffer.extend(pcm)

    async def flush(self) -> None:
        if not self._buffer:
            await self._emit(SttEvent(kind="final", text=""))
            return
        payload = bytes(self._buffer)
        self._buffer.clear()
        try:
            text = await asyncio.to_thread(self._transcribe_sync, payload)
        except Exception as exc:  # noqa: BLE001 - surfaced as an error event
            log.warning("OpenAI STT failed: %s", exc)
            await self._emit(SttEvent(kind="error", error=str(exc)))
            return
        await self._emit(SttEvent(kind="final", text=text))

    async def stop(self) -> None:
        self._buffer.clear()

    # ------------------------------------------------------------------

    def _transcribe_sync(self, pcm: bytes) -> str:
        import httpx

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(pcm)
        buffer.seek(0)

        files = {"file": ("audio.wav", buffer.read(), "audio/wav")}
        data: dict[str, str] = {"model": self.model}
        if self.language:
            data["language"] = self.language

        response = httpx.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            files=files,
            data=data,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return str(response.json().get("text", "")).strip()

    async def _emit(self, event: SttEvent) -> None:
        if self._on_event is None:
            return
        result = self._on_event(event)
        if asyncio.iscoroutine(result):
            await result
