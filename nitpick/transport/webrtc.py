"""The agent as a native WebRTC peer.

With this transport the agent is a real WebRTC endpoint: the browser's
``RTCPeerConnection`` negotiates with it directly, and audio travels as
Opus over SRTP. Nothing is relayed through the signaling socket, so the
latency is a network hop rather than a websocket hop.

It is optional — ``pip install nitpick[webrtc]`` pulls in ``aiortc``.

Two tracks are involved:

* **inbound** — the browser's microphone, decoded to 16 kHz mono PCM and
  pushed into the pipeline.
* **outbound** — the agent's TTS audio, encoded as Opus and sent to the
  browser.

Signaling still goes over the ordinary nitpick WebSocket protocol: the
browser sends ``webrtc_offer`` and ``ice_candidate`` messages, the server
answers them, and this transport does the media part.
"""
from __future__ import annotations

import asyncio
import fractions
import logging
from typing import Any

from ..frames import (
    DOWNSTREAM,
    AudioChunk,
    Frame,
    FrameProcessor,
    frame_input_audio,
)

log = logging.getLogger(__name__)

#: Opus in WebRTC always runs at 48 kHz.
WEBRTC_SAMPLE_RATE = 48_000


def _require_aiortc() -> Any:
    try:
        import aiortc  # noqa: F401
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "The WebRTC transport needs the 'webrtc' extra. "
            "Install it with: pip install nitpick[webrtc]"
        ) from exc
    return aiortc


class WebRtcTransport:
    """Bridges a WebRTC peer connection to a pipeline."""

    def __init__(
        self,
        *,
        ice_servers: list[dict[str, Any]] | None = None,
        input_sample_rate: int = 16_000,
        output_sample_rate: int = 24_000,
        audio_enabled: bool = True,
    ) -> None:
        _require_aiortc()
        from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection

        self.ice_servers = ice_servers or [{"urls": "stun:stun.l.google.com:19302"}]
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.audio_enabled = audio_enabled

        # aiortc wants typed configuration objects, not plain dicts. The
        # public API stays dict-shaped because that is what the browser's
        # RTCPeerConnection uses.
        servers = [
            server
            if isinstance(server, RTCIceServer)
            else RTCIceServer(
                urls=server["urls"],
                username=server.get("username"),
                credential=server.get("credential"),
            )
            for server in self.ice_servers
        ]

        self._pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=servers))
        self._inbound_queue: asyncio.Queue = asyncio.Queue()
        self._outbound_queue: asyncio.Queue = asyncio.Queue()
        self._input = FrameProcessor("WebRtcTransport.input", _WebRtcInput(self))
        self._output = FrameProcessor("WebRtcTransport.output", _WebRtcOutput(self))
        self._remote_track: Any = None
        self._reader_task: asyncio.Task | None = None

        # The 'track' event fires while the remote description is applied,
        # so the handler has to be registered before negotiation starts.
        @self._pc.on("track")
        def on_track(track: Any) -> None:  # pragma: no cover - event driven
            if track.kind != "audio":
                return
            self._remote_track = track
            self._reader_task = asyncio.create_task(
                self._read_remote(track), name="nitpick-webrtc-reader"
            )

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def input(self) -> FrameProcessor:
        return self._input

    def output(self) -> FrameProcessor:
        return self._output

    @property
    def peer_connection(self) -> Any:
        return self._pc

    # ------------------------------------------------------------------
    # Signaling
    # ------------------------------------------------------------------

    async def answer(self, offer: dict[str, Any]) -> dict[str, Any]:
        """Answer a remote offer and return the local description."""
        from aiortc import RTCSessionDescription

        await self._pc.setRemoteDescription(
            RTCSessionDescription(sdp=offer["sdp"], type=offer["type"])
        )

        if self.audio_enabled:
            self._attach_tracks()

        answer = await self._pc.createAnswer()
        await self._pc.setLocalDescription(answer)
        return {"sdp": self._pc.localDescription.sdp, "type": self._pc.localDescription.type}

    async def add_candidate(self, candidate: dict[str, Any]) -> None:
        """Add a remote ICE candidate."""
        from aiortc import RTCIceCandidate

        if not candidate:
            return
        try:
            await self._pc.addIceCandidate(
                RTCIceCandidate(
                    candidateFoundation=candidate.get("foundation"),
                    candidate=candidate.get("candidate"),
                    port=candidate.get("port"),
                    priority=candidate.get("priority"),
                    protocol=candidate.get("protocol"),
                    type=candidate.get("type"),
                    relatedAddress=candidate.get("relatedAddress"),
                    relatedPort=candidate.get("relatedPort"),
                    tcpType=candidate.get("tcpType"),
                    sdpMid=candidate.get("sdpMid"),
                    sdpMLineIndex=candidate.get("sdpMLineIndex"),
                )
            )
        except Exception as exc:  # noqa: BLE001 - a bad candidate is not fatal
            log.debug("Ignoring an ICE candidate: %s", exc)

    async def close(self) -> None:
        """Tear the peer connection down."""
        if self._reader_task is not None:
            self._reader_task.cancel()
            self._reader_task = None
        await self._pc.close()

    # ------------------------------------------------------------------
    # Media
    # ------------------------------------------------------------------

    def _attach_tracks(self) -> None:
        outbound = _build_outbound_track_class()(self._outbound_queue)
        self._pc.addTrack(outbound)

    async def _read_remote(self, track: Any) -> None:
        """Pull frames from the peer and push them into the pipeline."""
        import numpy as np

        while True:
            try:
                frame = await track.recv()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - the stream ended
                return

            try:
                samples = frame.to_ndarray()
            except Exception:  # noqa: BLE001
                continue

            if samples.ndim == 2:
                samples = samples.mean(axis=0)
            elif samples.ndim != 1:
                continue

            # aiortc delivers 48 kHz; resample to the pipeline's rate.
            if frame.sample_rate and frame.sample_rate != self.input_sample_rate:
                samples = _resample(
                    samples.astype(np.float32),
                    frame.sample_rate,
                    self.input_sample_rate,
                )

            pcm = (samples * 32767.0).astype(np.int16).tobytes()
            chunk = AudioChunk(
                data=pcm,
                sample_rate=self.input_sample_rate,
                num_channels=1,
            )
            await self._input.push_frame(frame_input_audio(chunk), DOWNSTREAM)

    async def publish_audio(self, chunk: AudioChunk) -> None:
        """Queue output audio for the peer."""
        await self._outbound_queue.put(chunk)


def _resample(samples: Any, from_rate: int, to_rate: int) -> Any:
    """Linear resample a float32 array."""
    import numpy as np

    if from_rate == to_rate or samples.size == 0:
        return samples
    ratio = from_rate / to_rate
    out_length = max(1, int(samples.size / ratio))
    positions = np.arange(out_length) * ratio
    left = np.floor(positions).astype(np.int64)
    frac = positions - left
    right = np.minimum(left + 1, samples.size - 1)
    return (samples[left] * (1 - frac) + samples[right] * frac).astype(np.float32)


def _build_outbound_track_class() -> Any:
    """Build the outbound track class.

    It has to subclass ``AudioStreamTrack`` — aiortc checks the type — so
    the class is created when ``aiortc`` is importable rather than at
    module import time, keeping ``aiortc`` an optional dependency.
    """
    from aiortc.mediastreams import AudioStreamTrack

    class OutboundTrack(AudioStreamTrack):
        """A WebRTC track that plays whatever the pipeline produces."""

        def __init__(self, queue: asyncio.Queue) -> None:
            super().__init__()
            self._queue = queue
            self._timestamp = 0

        async def recv(self) -> Any:
            import av
            import numpy as np

            chunk = await self._queue.get()
            samples = np.frombuffer(chunk.data, dtype=np.int16)

            # Opus in WebRTC always runs at 48 kHz, so resample if the
            # pipeline produces something else.
            if chunk.sample_rate and chunk.sample_rate != WEBRTC_SAMPLE_RATE:
                samples = _resample_int16(
                    samples, chunk.sample_rate, WEBRTC_SAMPLE_RATE
                )

            frame = av.AudioFrame.from_ndarray(
                samples.reshape(1, -1), format="s16", layout="mono"
            )
            frame.sample_rate = WEBRTC_SAMPLE_RATE
            frame.pts = self._timestamp
            frame.time_base = fractions.Fraction(1, WEBRTC_SAMPLE_RATE)
            self._timestamp += samples.size
            return frame

    return OutboundTrack


def _resample_int16(samples: Any, from_rate: int, to_rate: int) -> Any:
    """Linear resample an int16 array."""
    import numpy as np

    if from_rate == to_rate or samples.size == 0:
        return samples
    floats = samples.astype(np.float32)
    resampled = _resample(floats, from_rate, to_rate)
    return np.clip(resampled, -32768, 32767).astype(np.int16)


class _WebRtcInput:
    def __init__(self, transport: WebRtcTransport) -> None:
        self._transport = transport

    async def on_process_frame(
        self, processor: FrameProcessor, frame: Frame, direction: str
    ) -> None:
        await processor.push_frame(frame, direction)


class _WebRtcOutput:
    def __init__(self, transport: WebRtcTransport) -> None:
        self._transport = transport

    async def on_process_frame(
        self, processor: FrameProcessor, frame: Frame, direction: str
    ) -> None:
        if frame.kind.value == "output_audio_raw":
            payload = frame.payload
            if isinstance(payload, AudioChunk):
                await self._transport.publish_audio(payload)
        await processor.push_frame(frame, direction)
