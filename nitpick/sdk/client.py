"""The nitpick SDK — drive an agent room from your own code.

The SDK speaks the same WebSocket protocol as the browser client, so
anything you can do in the browser you can do from Python::

    from nitpick.sdk import RoomClient

    async with RoomClient("ws://localhost:8000/ws", name="My App") as client:
        await client.join_room("demo")
        await client.publish_pcm(pcm_bytes, sample_rate=16000)
        async for message in client.messages():
            if message["type"] == "room_audio":
                play(message["audio_base64"])
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections import deque
from typing import Any, AsyncIterator

log = logging.getLogger(__name__)


class SDKError(RuntimeError):
    """Raised when a signaling call fails."""


class RoomClient:
    """An async WebSocket client for a nitpick room.

    Parameters
    ----------
    websocket_url:
        The signaling URL, e.g. ``ws://localhost:8000/ws``.
    name:
        The display name shown in the room.
    """

    def __init__(self, websocket_url: str, *, name: str = "SDK User") -> None:
        self.websocket_url = websocket_url
        self.name = name
        self.participant_id: str | None = None
        self._websocket: Any | None = None
        self._incoming: asyncio.Queue = asyncio.Queue()
        self._reader: asyncio.Task | None = None
        self._pending: deque[dict[str, Any]] = deque()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> dict[str, Any]:
        """Open the connection and wait for the server's greeting."""
        if self._websocket is not None:
            return {"type": "connected", "participant_id": self.participant_id}

        try:
            import websockets
        except ImportError as exc:  # pragma: no cover - import guard
            raise SDKError("the 'websockets' package is required") from exc

        self._websocket = await websockets.connect(self.websocket_url)
        connected = await self._recv_until({"connected"})
        self.participant_id = connected.get("participant_id")
        self._reader = asyncio.create_task(self._read_loop(), name="nitpick-sdk-reader")
        return connected

    async def close(self) -> None:
        """Close the connection."""
        if self._reader is not None:
            self._reader.cancel()
            self._reader = None
        if self._websocket is not None:
            await self._websocket.close()
            self._websocket = None
        self.participant_id = None
        self._pending.clear()

    async def __aenter__(self) -> "RoomClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Room operations
    # ------------------------------------------------------------------

    async def join_room(
        self, room_id: str, *, max_participants: int = 10
    ) -> dict[str, Any]:
        """Join an existing room."""
        return await self._send_and_expect(
            {
                "type": "join_room",
                "room_id": room_id,
                "name": self.name,
                "max_participants": max_participants,
            },
            expected={"joined_room", "error"},
        )

    async def create_room(
        self, room_id: str, *, max_participants: int = 10
    ) -> dict[str, Any]:
        """Create a room and join it."""
        return await self._send_and_expect(
            {
                "type": "create_room",
                "room_id": room_id,
                "name": self.name,
                "max_participants": max_participants,
            },
            expected={"joined_room", "error"},
        )

    async def leave_room(self) -> dict[str, Any]:
        """Leave the current room."""
        return await self._send_and_expect(
            {"type": "leave_room"}, expected={"left_room", "error"}
        )

    async def publish_pcm(
        self,
        pcm: bytes,
        *,
        sample_rate: int = 16_000,
        channels: int = 1,
    ) -> dict[str, Any]:
        """Stream a chunk of raw PCM to the room.

        This is the path an agent listens on.
        """
        if not pcm:
            raise SDKError("pcm must not be empty")
        payload = {
            "type": "audio_pcm",
            "audio_base64": base64.b64encode(pcm).decode("ascii"),
            "sample_rate": sample_rate,
            "channels": channels,
        }
        await self.send(payload)
        return payload

    async def publish_audio(
        self, audio: bytes, *, mime_type: str = "audio/pcm"
    ) -> dict[str, Any]:
        """Fan out an opaque audio blob to the room."""
        if not audio:
            raise SDKError("audio must not be empty")
        payload = {
            "type": "publish_audio",
            "audio_base64": base64.b64encode(audio).decode("ascii"),
            "mime_type": mime_type,
        }
        await self.send(payload)
        return payload

    async def set_speaking(self, speaking: bool) -> None:
        """Report a client-side speaking transition."""
        await self.send(
            {
                "type": "participant_activity",
                "activity": "speaking_started" if speaking else "speaking_stopped",
            }
        )

    async def send_transcript(self, text: str, *, is_final: bool = True) -> None:
        """Report a transcript obtained elsewhere."""
        await self.send(
            {
                "type": "stt_transcript",
                "text": text,
                "is_final": is_final,
                "source": "nitpick_sdk",
            }
        )

    # ------------------------------------------------------------------
    # WebRTC signalling
    # ------------------------------------------------------------------

    async def send_offer(self, target_participant_id: str, sdp: Any) -> None:
        """Relay a WebRTC offer to one peer."""
        await self.send(
            {
                "type": "webrtc_offer",
                "target_participant_id": target_participant_id,
                "sdp": sdp,
            }
        )

    async def send_answer(self, target_participant_id: str, sdp: Any) -> None:
        """Relay a WebRTC answer to one peer."""
        await self.send(
            {
                "type": "webrtc_answer",
                "target_participant_id": target_participant_id,
                "sdp": sdp,
            }
        )

    async def send_ice_candidate(
        self, target_participant_id: str, candidate: Any
    ) -> None:
        """Relay an ICE candidate to one peer."""
        await self.send(
            {
                "type": "ice_candidate",
                "target_participant_id": target_participant_id,
                "candidate": candidate,
            }
        )

    # ------------------------------------------------------------------
    # Receiving
    # ------------------------------------------------------------------

    async def messages(self) -> AsyncIterator[dict[str, Any]]:
        """Yield messages as they arrive."""
        while True:
            message = await self.recv()
            if message is None:
                return
            yield message

    async def recv(self) -> dict[str, Any] | None:
        """Receive the next message. Returns ``None`` once disconnected."""
        if self._pending:
            return self._pending.popleft()
        return await self._incoming.get()

    async def send(self, payload: dict[str, Any]) -> None:
        """Send a raw message."""
        if self._websocket is None:
            raise SDKError("call connect() before sending")
        await self._websocket.send(json.dumps(payload))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _read_loop(self) -> None:
        try:
            while True:
                raw = await self._websocket.recv()
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(message, dict):
                    continue
                await self._incoming.put(message)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - the connection simply ended
            pass
        finally:
            await self._incoming.put(None)  # type: ignore[arg-type]

    async def _send_and_expect(
        self, payload: dict[str, Any], *, expected: set[str]
    ) -> dict[str, Any]:
        await self.send(payload)
        message = await self._recv_until(expected)
        if message.get("type") == "error":
            raise SDKError(f"{message.get('code')}: {message.get('message')}")
        return message

    async def _recv_until(self, expected: set[str]) -> dict[str, Any]:
        """Wait for a message of one of the expected types."""
        while True:
            message = await self.recv()
            if message is None:
                raise SDKError("the connection closed while waiting for a reply")
            message_type = message.get("type")
            if isinstance(message_type, str) and message_type in expected:
                return message
            self._pending.append(message)
