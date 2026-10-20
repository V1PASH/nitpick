"""Deepgram streaming speech-to-text.

Opens a WebSocket to Deepgram's ``listen`` endpoint and streams PCM as it
arrives. Interim results are forwarded so the UI can show live captions.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from .base import EventCallback, SttEvent, SttProvider

log = logging.getLogger(__name__)


class DeepgramSttProvider(SttProvider):
    """Streaming transcription via Deepgram."""

    name = "deepgram"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = "nova-3",
        language: str | None = None,
        sample_rate: int = 16_000,
        interim_results: bool = True,
    ) -> None:
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY")
        self.model = model
        self.language = language
        self.sample_rate = sample_rate
        self.interim_results = interim_results
        self._socket: Any = None
        self._on_event: EventCallback | None = None
        self._reader: asyncio.Task | None = None

    async def start(self, on_event: EventCallback) -> None:
        import websockets

        if not self.api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is not set")

        self._on_event = on_event
        query = [
            f"model={self.model}",
            "encoding=linear16",
            "sample_rate={self.sample_rate}",
            "channels=1",
            f"interim_results={'true' if self.interim_results else 'false'}",
            "smart_format=true",
        ]
        if self.language:
            query.append(f"language={self.language}")

        url = f"wss://api.deepgram.com/v1/listen?{'&'.join(query)}"
        headers = {"Authorization": f"Token {self.api_key}"}
        self._socket = await websockets.connect(url, additional_headers=headers)
        self._reader = asyncio.create_task(self._read_loop(), name="deepgram-reader")

    async def push_audio(self, pcm: bytes) -> None:
        if self._socket is None:
            return
        await self._socket.send(pcm)

    async def flush(self) -> None:
        if self._socket is None:
            return
        await self._socket.send(json.dumps({"type": "Finalize"}))

    async def stop(self) -> None:
        if self._reader is not None:
            self._reader.cancel()
            self._reader = None
        if self._socket is not None:
            await self._socket.close()
            self._socket = None

    # ------------------------------------------------------------------

    async def _read_loop(self) -> None:
        assert self._socket is not None
        try:
            async for raw in self._socket:
                message = json.loads(raw)
                await self._handle_message(message)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.exception("Deepgram reader failed")

    async def _handle_message(self, message: dict) -> None:
        if message.get("type") == "Results":
            channel = (message.get("channel") or {}).get("alternatives") or []
            if not channel:
                return
            alternative = channel[0]
            text = str(alternative.get("transcript", "")).strip()
            if not text:
                return
            is_final = bool(message.get("is_final"))
            await self._emit(
                SttEvent(
                    kind="final" if is_final else "partial",
                    text=text,
                    language=alternative.get("language"),
                    is_final=is_final,
                )
            )

    async def _emit(self, event: SttEvent) -> None:
        if self._on_event is None:
            return
        result = self._on_event(event)
        if asyncio.iscoroutine(result):
            await result
