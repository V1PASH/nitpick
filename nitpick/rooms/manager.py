"""The room manager — room state plus the event bus.

Everything in nitpick that needs to know "what happened in this room" goes
through :class:`RoomManager`. The signaling layer is just a protocol adapter
on top of it, and the agent runtime is just another consumer.

Design notes
------------
* Listeners receive a plain ``dict``, not an event object, so a listener
  written against one version keeps working when fields are added.
* Async listeners are scheduled as **tracked** tasks. A fire-and-forget
  ``create_task`` with no registry leaks tasks and swallows exceptions; the
  registry keeps a strong reference and the callback removes it on completion.
* Sends to participants are isolated: one dead socket never aborts a
  broadcast to the rest of the room.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from enum import Enum
from typing import Any, Awaitable, Callable

from .model import Participant, ParticipantRole, ParticipantState, Room

log = logging.getLogger(__name__)


class RoomEventType(str, Enum):
    """Every event the manager emits."""

    ROOM_CREATED = "room_created"
    ROOM_DESTROYED = "room_destroyed"
    PARTICIPANT_JOINED = "participant_joined"
    PARTICIPANT_LEFT = "participant_left"
    PARTICIPANT_SPEAKING_STARTED = "participant_speaking_started"
    PARTICIPANT_SPEAKING_STOPPED = "participant_speaking_stopped"
    PARTICIPANT_TRANSCRIPT = "participant_transcript"
    PARTICIPANT_AUDIO = "participant_audio"


EventListener = Callable[[dict[str, Any]], Any]


class RoomManager:
    """Owns rooms, participants, and the room event bus."""

    def __init__(self, *, event_history_size: int = 200) -> None:
        self._rooms: dict[str, Room] = {}
        self._participants: dict[str, Participant] = {}
        self._ws_index: dict[int, str] = {}
        self._listeners: list[EventListener] = []
        self._event_history: list[dict[str, Any]] = []
        self._event_history_size = event_history_size
        self._listener_tasks: set[asyncio.Task[Any]] = set()

    # ------------------------------------------------------------------
    # Event bus
    # ------------------------------------------------------------------

    def add_event_listener(self, listener: EventListener) -> None:
        if listener not in self._listeners:
            self._listeners.append(listener)

    def remove_event_listener(self, listener: EventListener) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    def list_events(self) -> list[dict[str, Any]]:
        return list(self._event_history)

    def _emit_event(
        self,
        event_type: RoomEventType,
        *,
        room_id: str | None = None,
        participant: Participant | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        event: dict[str, Any] = {
            "event_id": f"evt-{uuid.uuid4().hex[:12]}",
            "event_type": event_type.value,
            "room_id": room_id,
            "participant_id": participant.participant_id if participant else None,
            "participant_name": participant.name if participant else None,
            "participant_role": participant.role.value if participant else None,
            "is_speaking": participant.is_speaking if participant else None,
            "participant_count": (
                self._rooms[room_id].participant_count
                if room_id and room_id in self._rooms
                else 0
            ),
            "created_at": time.time(),
        }
        event.update(extra)

        self._event_history.append(event)
        if len(self._event_history) > self._event_history_size:
            del self._event_history[: len(self._event_history) - self._event_history_size]

        for listener in list(self._listeners):
            try:
                result = listener(event)
            except Exception:  # noqa: BLE001 - one bad listener must not break the bus
                log.exception("Room event listener failed")
                continue
            if asyncio.iscoroutine(result):
                task = asyncio.create_task(result)
                self._listener_tasks.add(task)
                task.add_done_callback(self._listener_tasks.discard)

        if event_type in _BROADCAST_EVENTS:
            self._schedule_broadcast(event)

        return event

    def _schedule_broadcast(self, event: dict[str, Any]) -> None:
        """Fan the event out to the room's websockets.

        Guarded because the manager can be driven from synchronous code
        (tests, scripts) where there is no running event loop to schedule
        the coroutine on.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        asyncio.create_task(self._broadcast_room_event(event))

    # ------------------------------------------------------------------
    # Rooms
    # ------------------------------------------------------------------

    def create_room(
        self, room_id: str, *, max_participants: int = 10, created_by: str | None = None
    ) -> Room:
        room = Room(
            room_id=room_id,
            max_participants=max_participants,
            created_by=created_by,
        )
        self._rooms[room_id] = room
        self._emit_event(RoomEventType.ROOM_CREATED, room_id=room_id)
        return room

    def has_room(self, room_id: str) -> bool:
        return room_id in self._rooms

    def get_room(self, room_id: str) -> Room | None:
        return self._rooms.get(room_id)

    def list_rooms(self) -> list[Room]:
        return list(self._rooms.values())

    def close_room(self, room_id: str) -> Room | None:
        room = self._rooms.pop(room_id, None)
        if room is None:
            return None
        for participant in room.participants:
            self._participants.pop(participant.participant_id, None)
            self._ws_index.pop(id(participant.websocket), None)
        self._emit_event(RoomEventType.ROOM_DESTROYED, room_id=room_id)
        return room

    # ------------------------------------------------------------------
    # Participants
    # ------------------------------------------------------------------

    def register_participant(
        self, websocket: Any, name: str = "Anonymous", *, role: ParticipantRole = ParticipantRole.LOCAL
    ) -> Participant:
        participant = Participant(
            websocket=websocket, name=name, role=role, state=ParticipantState.CONNECTING
        )
        self._participants[participant.participant_id] = participant
        if websocket is not None:
            self._ws_index[id(websocket)] = participant.participant_id
        return participant

    def register_agent_participant(
        self, name: str = "Agent", *, participant_id: str | None = None
    ) -> Participant:
        participant = Participant(
            websocket=None,
            name=name,
            role=ParticipantRole.AGENT,
            state=ParticipantState.CONNECTING,
        )
        if participant_id:
            participant.participant_id = participant_id
        self._participants[participant.participant_id] = participant
        return participant

    def unregister_participant_by_id(self, participant_id: str) -> Participant | None:
        participant = self._participants.pop(participant_id, None)
        if participant is None:
            return None
        if participant.websocket is not None:
            self._ws_index.pop(id(participant.websocket), None)
        if participant.room_id:
            room = self._rooms.get(participant.room_id)
            if room is not None and participant.participant_id in room._participants:
                room.remove_participant(participant.participant_id)
                self._emit_event(
                    RoomEventType.PARTICIPANT_LEFT,
                    room_id=room.room_id,
                    participant=participant,
                )
                if room.is_empty:
                    self.close_room(room.room_id)
        participant.state = ParticipantState.DISCONNECTED
        return participant

    def unregister_participant(self, websocket: Any) -> Participant | None:
        participant_id = self._ws_index.get(id(websocket))
        if participant_id is None:
            return None
        return self.unregister_participant_by_id(participant_id)

    def get_participant(self, participant_id: str) -> Participant | None:
        return self._participants.get(participant_id)

    def get_participant_by_websocket(self, websocket: Any) -> Participant | None:
        """Look up the participant behind a websocket."""
        if websocket is None:
            return None
        participant_id = self._ws_index.get(id(websocket))
        if participant_id is None:
            return None
        return self._participants.get(participant_id)

    # ------------------------------------------------------------------
    # Membership
    # ------------------------------------------------------------------

    def join_room(self, participant: Participant, room_id: str) -> Room:
        room = self._rooms.get(room_id)
        if room is None:
            room = self.create_room(room_id)
        room.add_participant(participant)
        self._emit_event(
            RoomEventType.PARTICIPANT_JOINED,
            room_id=room.room_id,
            participant=participant,
        )
        return room

    def leave_room(self, participant: Participant) -> str | None:
        room_id = participant.room_id
        if room_id is None:
            return None
        room = self._rooms.get(room_id)
        if room is None:
            return None
        if participant.participant_id in room._participants:
            room.remove_participant(participant.participant_id)
        self._emit_event(
            RoomEventType.PARTICIPANT_LEFT,
            room_id=room_id,
            participant=participant,
        )
        if room.is_empty:
            self.close_room(room_id)
        return room_id

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def update_participant_speaking(
        self, participant: Participant, speaking: bool
    ) -> dict[str, Any] | None:
        """Emit a speaking edge.

        Starting is edge-triggered — a repeated ``speaking_started`` is
        suppressed. Stopping is **not**: a stop closes the user's turn, so it
        is always emitted even if no matching start was seen. Dropping it
        would strand the turn open and the agent would never answer.
        """
        if speaking and participant.is_speaking:
            return None
        participant.is_speaking = speaking
        event_type = (
            RoomEventType.PARTICIPANT_SPEAKING_STARTED
            if speaking
            else RoomEventType.PARTICIPANT_SPEAKING_STOPPED
        )
        return self._emit_event(
            event_type, room_id=participant.room_id, participant=participant
        )

    def emit_participant_transcript(
        self,
        participant: Participant,
        *,
        transcript_text: str,
        transcript_is_final: bool = True,
        transcript_source: str = "agent_llm",
    ) -> dict[str, Any]:
        return self._emit_event(
            RoomEventType.PARTICIPANT_TRANSCRIPT,
            room_id=participant.room_id,
            participant=participant,
            transcript_text=transcript_text,
            transcript_is_final=transcript_is_final,
            transcript_source=transcript_source,
        )

    def emit_participant_audio(
        self,
        participant: Participant,
        *,
        audio: bytes,
        sample_rate: int,
        channels: int = 1,
    ) -> dict[str, Any]:
        """Publish a raw PCM frame from a participant.

        Audio events are **not** fanned out over the websocket — that would
        re-encode every frame as base64 for every peer. They are delivered to
        in-process listeners only (the agent runtime), which is where the
        STT pipeline lives.
        """
        return self._emit_event(
            RoomEventType.PARTICIPANT_AUDIO,
            room_id=participant.room_id,
            participant=participant,
            audio=audio,
            sample_rate=sample_rate,
            channels=channels,
        )

    # ------------------------------------------------------------------
    # Broadcasting
    # ------------------------------------------------------------------

    async def broadcast_message_to_room(
        self,
        *,
        room_id: str,
        payload: dict[str, Any],
        exclude_participant_id: str | None = None,
    ) -> None:
        """Send a payload to every participant in a room.

        Sends are isolated: a failure to one recipient is logged and the
        remaining recipients still get the message.
        """
        room = self._rooms.get(room_id)
        if room is None:
            return
        for participant in room.participants:
            if participant.websocket is None:
                continue
            if participant.participant_id == exclude_participant_id:
                continue
            await self._send_to_websocket(participant.websocket, payload)

    async def _broadcast_room_event(self, event: dict[str, Any]) -> None:
        room_id = event.get("room_id")
        if not room_id:
            return
        room = self._rooms.get(room_id)
        if room is None:
            return
        envelope = {"type": "room_event", "event": event}
        origin = event.get("participant_id")
        for participant in room.participants:
            if participant.websocket is None:
                continue
            if participant.participant_id == origin:
                continue
            await self._send_to_websocket(participant.websocket, envelope)

    @staticmethod
    async def _send_to_websocket(websocket: Any, payload: dict[str, Any]) -> None:
        """Send JSON to a websocket, tolerating different client shapes."""
        try:
            if hasattr(websocket, "send_json"):
                result = websocket.send_json(payload)
            elif hasattr(websocket, "send_text"):
                import json

                result = websocket.send_text(json.dumps(payload))
            elif hasattr(websocket, "send"):
                import json

                result = websocket.send(json.dumps(payload))
            else:
                return
            if asyncio.iscoroutine(result):
                await result
        except Exception:  # noqa: BLE001 - a dead socket must not break the room
            log.warning("Failed to send %s to participant", payload.get("type"))

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        return {
            "rooms": len(self._rooms),
            "participants": len(self._participants),
            "event_listeners": len(self._listeners),
            "events_buffered": len(self._event_history),
        }


_BROADCAST_EVENTS = frozenset(
    {
        RoomEventType.PARTICIPANT_JOINED,
        RoomEventType.PARTICIPANT_LEFT,
        RoomEventType.PARTICIPANT_SPEAKING_STARTED,
        RoomEventType.PARTICIPANT_SPEAKING_STOPPED,
        RoomEventType.PARTICIPANT_TRANSCRIPT,
    }
)
