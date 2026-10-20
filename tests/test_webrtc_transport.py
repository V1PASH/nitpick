"""Tests for the native WebRTC transport.

These connect two real ``aiortc`` peers over a loopback connection and
check that audio actually flows in both directions.
"""
from __future__ import annotations

import asyncio
import fractions

import pytest

pytest.importorskip("aiortc", reason="the WebRTC transport needs aiortc")

import numpy as np  # noqa: E402
from aiortc import RTCPeerConnection, RTCSessionDescription  # noqa: E402
from aiortc.mediastreams import AudioStreamTrack  # noqa: E402

from nitpick.frames import (  # noqa: E402
    DOWNSTREAM,
    AudioChunk,
    FrameKind,
    FrameProcessor,
    frame_start,
)
from nitpick.transport.webrtc import WebRtcTransport  # noqa: E402


class _ToneTrack(AudioStreamTrack):
    """Emits a continuous tone so the other side has something to decode."""

    def __init__(self, sample_rate: int = 48_000) -> None:
        super().__init__()
        self._timestamp = 0
        self._sample_rate = sample_rate

    async def recv(self):
        import av

        samples = np.zeros((1, 960), dtype=np.int16)
        frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
        frame.sample_rate = self._sample_rate
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, self._sample_rate)
        self._timestamp += samples.shape[1]
        # Pace the track at real time.
        await asyncio.sleep(0.02)
        return frame


@pytest.mark.asyncio
async def test_webrtc_transport_negotiates_and_moves_audio():
    agent = WebRtcTransport()
    browser = RTCPeerConnection()

    try:
        # The browser offers; the agent answers.
        browser.addTrack(_ToneTrack())
        offer = await browser.createOffer()
        await browser.setLocalDescription(offer)

        answer = await agent.answer(
            {"sdp": browser.localDescription.sdp, "type": browser.localDescription.type}
        )
        await browser.setRemoteDescription(
            RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
        )

        # Wait for the connection to come up.
        for _ in range(100):
            if browser.connectionState == "connected":
                break
            await asyncio.sleep(0.05)
        assert browser.connectionState == "connected"

        # The agent's input processor should receive audio from the peer.
        # The transport pushes into the head of the pipeline, so the capture
        # processor has to be linked after it.
        received: list[AudioChunk] = []

        class Capture:
            async def on_process_frame(self, processor, frame, direction):
                if frame.kind is FrameKind.INPUT_AUDIO_RAW:
                    received.append(frame.payload)
                await processor.push_frame(frame, direction)

        capture = FrameProcessor("capture", Capture())
        agent.input().link(capture)

        await agent.input().setup()
        await capture.setup()
        # START boots both processors: the input's process lane and the
        # capture's.
        await agent.input().push_frame(frame_start(), DOWNSTREAM)
        await asyncio.sleep(0.05)

        for _ in range(60):
            if received:
                break
            await asyncio.sleep(0.05)

        assert received, "the agent must receive audio from the peer"
        assert received[0].sample_rate == 16_000
        assert len(received[0].data) > 0

    finally:
        await browser.close()
        await agent.close()


@pytest.mark.asyncio
async def test_webrtc_transport_publishes_audio_to_the_peer():
    agent = WebRtcTransport()
    browser = RTCPeerConnection()

    try:
        received: list = []

        @browser.on("track")
        def on_track(track):
            received.append(track)

        # A sendrecv transceiver: the browser sends a tone and receives the
        # agent's audio on the same m-line.
        browser.addTrack(_ToneTrack())
        offer = await browser.createOffer()
        await browser.setLocalDescription(offer)

        answer = await agent.answer(
            {"sdp": browser.localDescription.sdp, "type": browser.localDescription.type}
        )
        await browser.setRemoteDescription(
            RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
        )

        for _ in range(100):
            if browser.connectionState == "connected":
                break
            await asyncio.sleep(0.05)

        # Push a stream of audio into the agent's output. The browser's
        # jitter buffer needs a few packets before it starts delivering
        # frames, so publish a realistic stream rather than a single chunk.
        pcm = (np.sin(np.arange(960) * 0.1) * 8000).astype(np.int16).tobytes()
        for _ in range(20):
            await agent.publish_audio(
                AudioChunk(data=pcm, sample_rate=24_000, num_channels=1)
            )
            await asyncio.sleep(0.02)

        # The browser receives the agent's track on the shared m-line.
        for _ in range(60):
            if received:
                break
            await asyncio.sleep(0.05)

        assert received, "the browser must receive a track from the agent"

        frame = await asyncio.wait_for(received[0].recv(), timeout=10)
        assert frame.sample_rate == 48_000

    finally:
        await browser.close()
        await agent.close()
