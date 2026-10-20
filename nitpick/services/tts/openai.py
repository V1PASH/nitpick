"""Text-to-speech.

The service buffers LLM text into sentences and synthesises each sentence as
it completes, so the first audio reaches the user while the model is still
generating the rest of the reply.

Any OpenAI-compatible TTS endpoint works. The response format is PCM, so the
audio goes straight into the output transport without decoding.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...frames import (
    DOWNSTREAM,
    AudioChunk,
    Frame,
    FrameKind,
    FrameProcessor,
    frame_output_audio,
)

if TYPE_CHECKING:  # pragma: no cover
    from ...billing import BillingCollector

log = logging.getLogger(__name__)

#: Abbreviations that end in a period but do not end a sentence.
_ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.",
    "etc.", "e.g.", "i.e.", "inc.", "ltd.", "co.", "corp.", "fig.", "no.",
    "approx.", "dept.", "est.", "min.", "max.",
}

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class TTSConfig:
    """Configuration for :class:`TTSService`."""

    api_key: str | None = None
    model: str = "gpt-4o-mini-tts"
    voice: str = "alloy"
    base_url: str = "https://api.openai.com/v1"
    sample_rate: int = 24_000
    timeout: float = 60.0
    #: Audio format requested from the provider. ``pcm`` is raw 16-bit.
    response_format: str = "pcm"

    def resolved_key(self) -> str:
        key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        return key


class TTSService(FrameProcessor):
    """Turns LLM text into audio frames."""

    def __init__(
        self,
        config: TTSConfig | None = None,
        *,
        name: str = "TTSService",
        billing: "BillingCollector | None" = None,
    ) -> None:
        super().__init__(name, _TTSHandler(self))
        self.config = config or TTSConfig()
        self._billing = billing
        self._buffer = ""
        self._speaking = False

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    async def synthesize(self, text: str) -> bytes:
        """Synthesise one chunk of text to raw PCM."""
        import httpx

        payload = {
            "model": self.config.model,
            "voice": self.config.voice,
            "input": text,
            "response_format": self.config.response_format,
        }
        url = f"{self.config.base_url.rstrip('/')}/audio/speech"
        headers = {
            "Authorization": f"Bearer {self.config.resolved_key()}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.content

    async def _emit_audio(self, text: str) -> None:
        text = text.strip()
        if not text:
            return

        if not self._speaking:
            self._speaking = True
            await self.push_frame(Frame(FrameKind.BOT_STARTED_SPEAKING), DOWNSTREAM)

        try:
            pcm = await self.synthesize(text)
        except Exception as exc:  # noqa: BLE001 - a TTS failure must not kill the turn
            log.warning("TTS synthesis failed: %s", exc)
            await self.push_error(f"TTS error: {exc}", fatal=False)
            return

        if self._billing is not None:
            self._billing.record_tts_usage(
                characters=len(text), model=self.config.model
            )

        chunk = AudioChunk(
            data=pcm,
            sample_rate=self.config.sample_rate,
            num_channels=1,
        )
        await self.push_frame(frame_output_audio(chunk), DOWNSTREAM)

    async def _finish_speaking(self) -> None:
        if not self._speaking:
            return
        self._speaking = False
        await self.push_frame(Frame(FrameKind.BOT_STOPPED_SPEAKING), DOWNSTREAM)


class _TTSHandler:
    """Buffers LLM text and synthesises complete sentences."""

    def __init__(self, service: TTSService) -> None:
        self._service = service

    async def on_process_frame(
        self, processor: FrameProcessor, frame: Frame, direction: str
    ) -> None:
        service = self._service
        kind = frame.kind

        if kind is FrameKind.LLM_TEXT:
            text = frame.payload if isinstance(frame.payload, str) else ""
            service._buffer += text
            sentences, service._buffer = _split_sentences(service._buffer)
            for sentence in sentences:
                await service._emit_audio(sentence)
            await processor.push_frame(frame, direction)
            return

        if kind is FrameKind.LLM_FULL_RESPONSE_END:
            remaining = service._buffer.strip()
            service._buffer = ""
            if remaining:
                await service._emit_audio(remaining)
            await service._finish_speaking()
            await processor.push_frame(frame, direction)
            return

        if kind is FrameKind.INTERRUPTION:
            service._buffer = ""
            await service._finish_speaking()
            await processor.push_frame(frame, direction)
            return

        await processor.push_frame(frame, direction)


def _split_sentences(text: str) -> tuple[list[str], str]:
    """Split text into complete sentences plus a remainder.

    Returns ``(sentences, remainder)`` where ``remainder`` is the trailing
    partial sentence that may still grow. Abbreviations are handled so
    ``Mr. Smith`` does not split the sentence in half.
    """
    if not text:
        return [], ""

    boundaries: list[int] = []
    index = 0
    length = len(text)

    while index < length:
        char = text[index]
        if char in ".!?":
            # A boundary needs whitespace (or end of text) after the mark.
            next_index = index + 1
            if next_index >= length or text[next_index].isspace():
                if char == "." and _ends_with_abbreviation(text[: next_index]):
                    index = next_index
                    continue
                boundaries.append(next_index)
        index += 1

    if not boundaries:
        return [], text

    # Every boundary is a real sentence end: the text after the last one is
    # the incomplete tail that may still grow.
    cut = boundaries[-1]
    sentences = [part.strip() for part in _cut(text, boundaries)]
    sentences = [sentence for sentence in sentences if sentence]
    remainder = text[cut:].lstrip() if cut < length else ""
    return sentences, remainder


def _cut(text: str, boundaries: list[int]) -> list[str]:
    """Cut ``text`` at the given end offsets."""
    parts: list[str] = []
    start = 0
    for boundary in boundaries:
        parts.append(text[start:boundary])
        start = boundary
    return parts


def _ends_with_abbreviation(text: str) -> bool:
    """True if the text ends with a known abbreviation like ``Mr.``."""
    stripped = text.rstrip()
    if not stripped.endswith("."):
        return False
    words = stripped.split()
    if not words:
        return False
    return words[-1].lower() in _ABBREVIATIONS
