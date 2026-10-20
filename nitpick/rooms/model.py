"""Room and participant data model."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


def new_room_id() -> str:
    return f"r-{uuid.uuid4().hex[:12]}"


def new_participant_id() -> str:
    return f"p-{uuid.uuid4().hex[:12]}"


class ParticipantRole(str, Enum):
    """What a participant is allowed to do."""

    LOCAL = "local"
    REMOTE = "remote"
    AGENT = "agent"


class ParticipantState(str, Enum):
    """Where a participant is in its lifecycle."""

    CONNECTING = "connecting"
    IN_ROOM = "in_room"
    DISCONNECTED = "disconnected"


@dataclass
class Participant:
    """One member of a room — a browser tab, a phone, or an agent."""

    websocket: Any | None = None
    name: str = "Anonymous"
    role: ParticipantRole = ParticipantRole.LOCAL
    participant_id: str = field(default_factory=new_participant_id)
    room_id: Optional[str] = None
    state: ParticipantState = ParticipantState.CONNECTING
    is_speaking: bool = False
    created_at: float = field(default_factory=time.time)

    @property
    def is_agent(self) -> bool:
        return self.role is ParticipantRole.AGENT

    def to_dict(self) -> dict[str, Any]:
        return {
            "participant_id": self.participant_id,
            "name": self.name,
            "role": self.role.value,
            "room_id": self.room_id,
            "state": self.state.value,
            "is_speaking": self.is_speaking,
            "created_at": self.created_at,
        }


class RoomError(Exception):
    """Base class for room errors."""


class RoomFullError(RoomError):
    """The room has reached its participant limit."""


class ParticipantNotInRoomError(RoomError):
    """The participant is not a member of the room."""


class DuplicateParticipantError(RoomError):
    """The participant is already a member of the room."""


class RoomNotFoundError(RoomError):
    """No room with that id exists."""


@dataclass
class Room:
    """A named group of participants that can exchange media and events."""

    room_id: str = field(default_factory=new_room_id)
    max_participants: int = 10
    created_at: float = field(default_factory=time.time)
    created_by: Optional[str] = None
    _participants: dict[str, Participant] = field(default_factory=dict)

    def add_participant(self, participant: Participant) -> None:
        if len(self._participants) >= self.max_participants:
            raise RoomFullError(
                f"room {self.room_id} is full ({self.max_participants})"
            )
        if participant.participant_id in self._participants:
            raise DuplicateParticipantError(participant.participant_id)
        participant.room_id = self.room_id
        participant.state = ParticipantState.IN_ROOM
        self._participants[participant.participant_id] = participant

    def remove_participant(self, participant_id: str) -> Participant:
        if participant_id not in self._participants:
            raise ParticipantNotInRoomError(participant_id)
        participant = self._participants.pop(participant_id)
        participant.state = ParticipantState.DISCONNECTED
        participant.room_id = None
        return participant

    def get_participant(self, participant_id: str) -> Participant | None:
        return self._participants.get(participant_id)

    @property
    def participants(self) -> list[Participant]:
        return list(self._participants.values())

    @property
    def participant_count(self) -> int:
        return len(self._participants)

    @property
    def is_empty(self) -> bool:
        return not self._participants

    @property
    def is_full(self) -> bool:
        return len(self._participants) >= self.max_participants

    def to_dict(self, *, include_participants: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "room_id": self.room_id,
            "max_participants": self.max_participants,
            "participant_count": self.participant_count,
            "created_at": self.created_at,
            "created_by": self.created_by,
        }
        if include_participants:
            payload["participants"] = [p.to_dict() for p in self.participants]
        return payload
