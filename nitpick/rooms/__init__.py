"""Rooms, participants, and the room event bus."""
from __future__ import annotations

from .control import (
    ParticipantNotFoundError,
    ParticipantNotInRoomControlError,
    ParticipantRemovalResult,
    RoomAlreadyExistsError,
    RoomCapacityError,
    RoomCloseResult,
    RoomControlError,
    RoomControlModule,
)
from .manager import RoomEventType, RoomManager
from .model import (
    DuplicateParticipantError,
    Participant,
    ParticipantRole,
    ParticipantState,
    Room,
    RoomError,
    RoomFullError,
    RoomNotFoundError,
    new_participant_id,
    new_room_id,
)

__all__ = [
    "DuplicateParticipantError",
    "Participant",
    "ParticipantNotFoundError",
    "ParticipantNotInRoomControlError",
    "ParticipantRemovalResult",
    "ParticipantRole",
    "ParticipantState",
    "Room",
    "RoomAlreadyExistsError",
    "RoomCapacityError",
    "RoomCloseResult",
    "RoomControlError",
    "RoomControlModule",
    "RoomError",
    "RoomEventType",
    "RoomFullError",
    "RoomManager",
    "RoomNotFoundError",
    "new_participant_id",
    "new_room_id",
]
