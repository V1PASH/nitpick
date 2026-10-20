"""The nitpick signaling server.

A thin protocol adapter over :class:`~nitpick.rooms.RoomManager`. It speaks
a small, flat JSON protocol over a single WebSocket:

**Client → server**

=========================  ====================================================
``join_room``              join an existing room
``create_room``            create a room, then join it
``leave_room``             leave the current room
``participant_activity``   a speaking edge (``speaking_started``/``stopped``)
``stt_transcript``         a transcript from the browser's speech recogniser
``stt_audio_chunk``        a batch audio chunk (WebM/Opus) for server-side STT
``audio_pcm``              **raw PCM frames** — the streaming path
``publish_audio``          base64 audio to fan out to the room
``webrtc_offer``           SDP relay to one peer
``webrtc_answer``          SDP relay to one peer
``ice_candidate``          ICE relay to one peer
=========================  ====================================================

**Server → client**

``connected``, ``joined_room``, ``left_room``, ``room_event``,
``room_audio``, ``audio_published``, ``webrtc_offer``, ``webrtc_answer``,
``ice_candidate``, ``error``.

The server never touches media for the browser-to-browser path — WebRTC is
native in the browser and the server only relays SDP and ICE.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from ..rooms import RoomControlModule, RoomManager

log = logging.getLogger(__name__)

MAX_ROOM_AUDIO_CHUNK_BYTES = 1024 * 1024
MAX_STT_AUDIO_CHUNK_BYTES = 5 * 1024 * 1024


class SignalingConfig:
    """Environment-driven configuration for the signaling server."""

    def __init__(self) -> None:
        self.reload()

    def reload(self) -> None:
        self.room_create_api_key = os.getenv("ROOM_CREATE_API_KEY") or None
        self.room_create_api_secret = os.getenv("ROOM_CREATE_API_SECRET") or None
        self.require_create_room_auth = _env_bool(
            "REQUIRE_CREATE_ROOM_AUTH",
            default=bool(self.room_create_api_key and self.room_create_api_secret),
        )
        self.room_create_token_ttl_seconds = _env_int(
            "ROOM_CREATE_TOKEN_TTL_SECONDS", 300
        )
        self.require_explicit_room_create = _env_bool(
            "REQUIRE_EXPLICIT_ROOM_CREATE", default=False
        )
        self.max_room_participants = _env_int("MAX_ROOM_PARTICIPANTS", 100)

    def to_dict(self) -> dict[str, Any]:
        return {
            "require_create_room_auth": self.require_create_room_auth,
            "require_explicit_room_create": self.require_explicit_room_create,
            "max_room_participants": self.max_room_participants,
        }


def _env_bool(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class ProtocolError(Exception):
    """A client protocol violation. Carries a machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class SignalingServer:
    """Owns the room manager and implements the WebSocket protocol."""

    def __init__(
        self,
        manager: RoomManager | None = None,
        *,
        control: RoomControlModule | None = None,
        audio_output_sample_rate: int = 24_000,
    ) -> None:
        self.manager = manager or RoomManager()
        self.control = control or RoomControlModule(self.manager)
        self.config = SignalingConfig()
        self.control.max_participants_limit = self.config.max_room_participants
        #: The sample rate of the audio the agent publishes, so the browser
        #: can play it without guessing.
        self.audio_output_sample_rate = audio_output_sample_rate

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    async def handle_message(
        self, websocket: Any, payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Process one client message. Returns a response to send, if any.

        Errors are returned as ``error`` payloads rather than raised, so the
        transport layer never has to guess at a protocol code.
        """
        try:
            return await self._dispatch(websocket, payload)
        except ProtocolError as exc:
            return {"type": "error", "code": exc.code, "message": exc.message}

    async def _dispatch(
        self, websocket: Any, payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        message_type = payload.get("type")
        if not isinstance(message_type, str) or not message_type:
            raise ProtocolError("missing_type", "Message must carry a string 'type'")

        handler = _HANDLERS.get(message_type)
        if handler is None:
            raise ProtocolError(
                "unsupported_type", f"Unsupported message type: {message_type}"
            )
        return await handler(self, websocket, payload)

    # ---- join / create / leave ------------------------------------------

    async def _handle_join(self, websocket: Any, payload: dict[str, Any]) -> dict[str, Any]:
        room_id = payload.get("room_id")
        if not isinstance(room_id, str) or not room_id.strip():
            raise ProtocolError("invalid_room_id", "room_id is required")

        participant = self.manager.get_participant_by_websocket(websocket)
        if participant is None:
            participant = self.manager.register_participant(
                websocket, name=str(payload.get("name") or "Anonymous")
            )

        create_if_missing = not self.config.require_explicit_room_create
        try:
            room, created = self.control.join_participant(
                participant,
                room_name=room_id,
                max_participants=payload.get("max_participants"),
                create_if_missing=create_if_missing,
            )
        except Exception as exc:  # noqa: BLE001 - mapped to a protocol error
            self.manager.unregister_participant_by_id(participant.participant_id)
            raise ProtocolError("room_not_found", str(exc)) from exc

        others = [p.to_dict() for p in room.participants if p.participant_id != participant.participant_id]
        return {
            "type": "joined_room",
            "room": room.to_dict(include_participants=False),
            "self": participant.to_dict(),
            "participants": others,
            "room_created": created,
            "join_source": "join_room",
        }

    async def _handle_create(self, websocket: Any, payload: dict[str, Any]) -> dict[str, Any]:
        room_id = payload.get("room_id")
        if not isinstance(room_id, str) or not room_id.strip():
            raise ProtocolError("invalid_room_id", "room_id is required")

        self._verify_create_room_auth(room_id, payload)

        participant = self.manager.get_participant_by_websocket(websocket)
        if participant is None:
            participant = self.manager.register_participant(
                websocket, name=str(payload.get("name") or "Anonymous")
            )

        try:
            room, _created = self.control.join_participant(
                participant,
                room_name=room_id,
                max_participants=payload.get("max_participants"),
                create_if_missing=True,
            )
        except Exception as exc:  # noqa: BLE001
            self.manager.unregister_participant_by_id(participant.participant_id)
            raise ProtocolError("room_already_exists", str(exc)) from exc

        others = [p.to_dict() for p in room.participants if p.participant_id != participant.participant_id]
        return {
            "type": "joined_room",
            "room": room.to_dict(include_participants=False),
            "self": participant.to_dict(),
            "participants": others,
            "room_created": True,
            "join_source": "create_room",
        }

    async def _handle_leave(self, websocket: Any, payload: dict[str, Any]) -> dict[str, Any]:
        participant = self.manager.get_participant_by_websocket(websocket)
        if participant is None or participant.room_id is None:
            raise ProtocolError("not_in_room", "Participant is not in a room")
        room_id = participant.room_id
        room = self.manager.get_room(room_id)
        existed = room is not None
        self.manager.leave_room(participant)
        return {
            "type": "left_room",
            "room_id": room_id,
            "room_exists": self.manager.has_room(room_id),
            "room_deleted": existed and not self.manager.has_room(room_id),
        }

    # ---- activity -------------------------------------------------------

    async def _handle_activity(
        self, websocket: Any, payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        participant = self.manager.get_participant_by_websocket(websocket)
        if participant is None:
            raise ProtocolError("not_in_room", "Participant is not in a room")
        activity = payload.get("activity")
        if activity == "speaking_started":
            self.manager.update_participant_speaking(participant, True)
        elif activity == "speaking_stopped":
            self.manager.update_participant_speaking(participant, False)
        else:
            raise ProtocolError("invalid_activity", f"Unknown activity: {activity}")
        return None

    # ---- transcripts ----------------------------------------------------

    async def _handle_stt_transcript(
        self, websocket: Any, payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        participant = self.manager.get_participant_by_websocket(websocket)
        if participant is None:
            raise ProtocolError("not_in_room", "Participant is not in a room")
        text = payload.get("text")
        if not isinstance(text, str):
            raise ProtocolError("invalid_stt_payload", "text must be a string")
        normalized = " ".join(text.strip().split())
        if not normalized:
            return None
        self.manager.emit_participant_transcript(
            participant,
            transcript_text=normalized,
            transcript_is_final=bool(payload.get("is_final", False)),
            transcript_source=str(payload.get("source") or "browser_web_speech"),
        )
        return None

    # ---- audio ----------------------------------------------------------

    async def _handle_audio_pcm(
        self, websocket: Any, payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        """The streaming path: raw PCM frames forwarded to the agent.

        Unlike ``stt_audio_chunk`` this is *not* a batch format — it is a
        steady stream of fixed-size frames, which is what a streaming STT
        provider needs.
        """
        participant = self.manager.get_participant_by_websocket(websocket)
        if participant is None:
            raise ProtocolError("not_in_room", "Participant is not in a room")

        audio_b64 = payload.get("audio_base64")
        if not isinstance(audio_b64, str) or not audio_b64:
            raise ProtocolError("invalid_audio_payload", "audio_base64 is required")
        try:
            raw = base64.b64decode(audio_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ProtocolError("invalid_audio_payload", "audio_base64 is not valid base64") from exc
        if not raw:
            raise ProtocolError("invalid_audio_payload", "audio chunk is empty")
        if len(raw) > MAX_ROOM_AUDIO_CHUNK_BYTES:
            raise ProtocolError("audio_chunk_too_large", "audio chunk exceeds the size limit")

        sample_rate = int(payload.get("sample_rate") or 16_000)
        channels = int(payload.get("channels") or 1)

        # Hand the frame to the agent through the room event bus so in-process
        # agents (and out-of-process SDK clients) both receive it.
        self.manager.emit_participant_audio(
            participant,
            audio=raw,
            sample_rate=sample_rate,
            channels=channels,
        )
        return None

    async def _handle_publish_audio(
        self, websocket: Any, payload: dict[str, Any]
    ) -> dict[str, Any]:
        participant = self.manager.get_participant_by_websocket(websocket)
        if participant is None or participant.room_id is None:
            raise ProtocolError("not_in_room", "Participant is not in a room")

        audio_b64 = payload.get("audio_base64")
        if not isinstance(audio_b64, str) or not audio_b64:
            raise ProtocolError("invalid_audio_payload", "audio_base64 is required")
        try:
            raw = base64.b64decode(audio_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ProtocolError("invalid_audio_payload", "audio_base64 is not valid base64") from exc
        if not raw:
            raise ProtocolError("invalid_audio_payload", "audio chunk is empty")
        if len(raw) > MAX_ROOM_AUDIO_CHUNK_BYTES:
            raise ProtocolError("audio_chunk_too_large", "audio chunk exceeds the size limit")

        room_audio = {
            "type": "room_audio",
            "room_id": participant.room_id,
            "from_participant_id": participant.participant_id,
            "mime_type": str(payload.get("mime_type") or "audio/pcm"),
            "audio_base64": audio_b64,
            "audio_bytes": len(raw),
            "created_at": time.time(),
        }
        sequence = payload.get("sequence")
        if sequence is not None:
            room_audio["sequence"] = sequence

        await self.manager.broadcast_message_to_room(
            room_id=participant.room_id,
            payload=room_audio,
            exclude_participant_id=participant.participant_id,
        )
        return {
            "type": "audio_published",
            "room_id": participant.room_id,
            "audio_bytes": len(raw),
            "sequence": sequence,
        }

    # ---- WebRTC relay ----------------------------------------------------

    async def _handle_webrtc_relay(
        self, websocket: Any, payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Relay SDP/ICE to a single peer.

        The server never parses the SDP or the candidate — it copies the
        opaque blob and adds routing metadata.
        """
        sender = self.manager.get_participant_by_websocket(websocket)
        if sender is None or sender.room_id is None:
            raise ProtocolError("not_in_room", "Participant is not in a room")

        target_id = payload.get("target_participant_id")
        if not isinstance(target_id, str) or not target_id:
            raise ProtocolError("invalid_target", "target_participant_id is required")

        target = self.manager.get_participant(target_id)
        if target is None or target.websocket is None:
            raise ProtocolError("target_not_found", "Target participant not found")
        if target.room_id != sender.room_id:
            raise ProtocolError("target_not_in_room", "Target is in a different room")

        relay: dict[str, Any] = {
            "type": payload["type"],
            "from_participant_id": sender.participant_id,
            "room_id": sender.room_id,
        }
        if "sdp" in payload:
            relay["sdp"] = payload["sdp"]
        if "candidate" in payload:
            relay["candidate"] = payload["candidate"]

        await self.manager._send_to_websocket(target.websocket, relay)
        return None

    # ---- auth ------------------------------------------------------------

    def _verify_create_room_auth(
        self, room_id: str, payload: dict[str, Any]
    ) -> None:
        if not self.config.require_create_room_auth:
            return

        api_key = self.config.room_create_api_key
        api_secret = self.config.room_create_api_secret
        if not api_key or not api_secret:
            raise ProtocolError(
                "create_room_auth_not_configured",
                "create_room authentication is required but not configured",
            )

        token = payload.get("api_token")
        api_ts = payload.get("api_ts")
        if not isinstance(token, str) or not isinstance(api_ts, int):
            raise ProtocolError(
                "invalid_create_room_auth", "api_token and api_ts are required"
            )

        skew = abs(int(time.time()) - api_ts)
        if skew > self.config.room_create_token_ttl_seconds:
            raise ProtocolError("expired_create_room_token", "Token has expired")

        expected = hmac.new(
            api_secret.encode("utf-8"),
            f"{api_key}:{room_id}:{api_ts}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected, token):
            raise ProtocolError("unauthorized_create_room", "Invalid create-room token")


_HANDLERS = {
    "join_room": SignalingServer._handle_join,
    "create_room": SignalingServer._handle_create,
    "leave_room": SignalingServer._handle_leave,
    "participant_activity": SignalingServer._handle_activity,
    "stt_transcript": SignalingServer._handle_stt_transcript,
    "audio_pcm": SignalingServer._handle_audio_pcm,
    "publish_audio": SignalingServer._handle_publish_audio,
    "webrtc_offer": SignalingServer._handle_webrtc_relay,
    "webrtc_answer": SignalingServer._handle_webrtc_relay,
    "ice_candidate": SignalingServer._handle_webrtc_relay,
}


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------


def create_app(server: SignalingServer | None = None) -> Any:
    """Build the FastAPI application.

    The app exposes ``/ws`` (signaling), ``/health``, and ``/`` (the browser
    client).
    """
    # NOTE: FastAPI, WebSocket and the response classes are imported at
    # module scope on purpose. This module uses ``from __future__ import
    # annotations``, so FastAPI resolves the endpoint's annotations by name
    # from the module globals. A function-local import would leave
    # ``WebSocket`` unresolvable and FastAPI would reject the connection
    # with a policy-violation close.
    signaling = server or SignalingServer()

    app = FastAPI(title="nitpick", version="0.1.0")
    app.state.signaling = signaling

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "stats": signaling.manager.stats(),
                "config": signaling.config.to_dict(),
            }
        )

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        html = _load_web_client()
        if html is None:
            return HTMLResponse("<h1>nitpick</h1><p>Browser client not installed.</p>")
        return HTMLResponse(html)

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        participant = signaling.manager.register_participant(websocket)
        await websocket.send_json(
            {
                "type": "connected",
                "participant_id": participant.participant_id,
                "stt": {"provider": "pcm", "transport": "audio_pcm"},
                "audio": {"output_sample_rate": signaling.audio_output_sample_rate},
            }
        )
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "code": "invalid_json",
                            "message": "Payload is not valid JSON",
                        }
                    )
                    continue
                if not isinstance(payload, dict):
                    await websocket.send_json(
                        {
                            "type": "error",
                            "code": "invalid_payload",
                            "message": "Payload must be a JSON object",
                        }
                    )
                    continue
                response = await signaling.handle_message(websocket, payload)
                if response is not None:
                    await websocket.send_json(response)
        except WebSocketDisconnect:
            pass
        finally:
            signaling.manager.unregister_participant_by_id(participant.participant_id)

    return app


def _load_web_client() -> str | None:
    path = Path(__file__).resolve().parent.parent / "web" / "index.html"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None
