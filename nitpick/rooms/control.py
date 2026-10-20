"""Validation and administration facade over :class:`RoomManager`.

The manager holds state; this module holds the *rules*. Keeping them apart
means the rules can be unit-tested without touching the manager, and the
manager stays free of policy.
"""
from __future__ import annotations

from dataclasses import dataclass

from .manager import RoomManager
from .model import Participant, Room, RoomNotFoundError


@dataclass(frozen=True)
class RoomCloseResult:
    room_id: str
    removed_participant_ids: tuple[str, ...]


@dataclass(frozen=True)
class ParticipantRemovalResult:
    participant_id: str
    room_id: str | None
    room_deleted: bool


class RoomControlError(ValueError):
    """Base class for control-layer errors."""


class RoomAlreadyExistsError(RoomControlError):
    """A room with that id already exists."""


class RoomCapacityError(RoomControlError):
    """The requested capacity is out of bounds."""


class ParticipantNotFoundError(RoomControlError):
    """No participant with that id exists."""


class ParticipantNotInRoomControlError(RoomControlError):
    """The participant exists but is not in that room."""


class RoomControlModule:
    """Validated room operations."""

    DEFAULT_MAX_PARTICIPANTS = 10
    DEFAULT_MAX_PARTICIPANTS_LIMIT = 100
    DEFAULT_MAX_ROOM_NAME_LENGTH = 64

    def __init__(
        self,
        manager: RoomManager,
        *,
        default_max_participants: int = DEFAULT_MAX_PARTICIPANTS,
        max_participants_limit: int = DEFAULT_MAX_PARTICIPANTS_LIMIT,
        max_room_name_length: int = DEFAULT_MAX_ROOM_NAME_LENGTH,
    ) -> None:
        self._manager = manager
        self._default_max_participants = default_max_participants
        self._max_participants_limit = max_participants_limit
        self._max_room_name_length = max_room_name_length

    @property
    def manager(self) -> RoomManager:
        return self._manager

    @property
    def max_room_name_length(self) -> int:
        """The longest a room id may be."""
        return self._max_room_name_length

    @property
    def max_participants_limit(self) -> int:
        return self._max_participants_limit

    @max_participants_limit.setter
    def max_participants_limit(self, value: int) -> None:
        self._max_participants_limit = max(1, int(value))

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_room_name(self, room_name: str) -> str:
        if not isinstance(room_name, str):
            raise RoomControlError("room_name must be a string")
        normalized = room_name.strip()
        if not normalized:
            raise RoomControlError("room_name must not be empty")
        if len(normalized) > self._max_room_name_length:
            raise RoomControlError(
                f"room_name must be at most {self._max_room_name_length} characters"
            )
        return normalized

    def validate_max_participants(self, value: int | None) -> int:
        if value is None:
            return self._default_max_participants
        if isinstance(value, bool) or not isinstance(value, int):
            raise RoomControlError("max_participants must be an integer")
        if value < 1:
            raise RoomControlError("max_participants must be at least 1")
        if value > self._max_participants_limit:
            raise RoomCapacityError(
                f"max_participants must be at most {self._max_participants_limit}"
            )
        return value

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def create_room(
        self,
        room_name: str,
        *,
        creator_id: str | None = None,
        max_participants: int | None = None,
    ) -> Room:
        room_id = self.validate_room_name(room_name)
        capacity = self.validate_max_participants(max_participants)
        if self._manager.has_room(room_id):
            raise RoomAlreadyExistsError(room_id)
        return self._manager.create_room(
            room_id, max_participants=capacity, created_by=creator_id
        )

    def join_participant(
        self,
        participant: Participant,
        *,
        room_name: str,
        max_participants: int | None = None,
        create_if_missing: bool = True,
    ) -> tuple[Room, bool]:
        """Join a participant to a room. Returns ``(room, created)``."""
        room_id = self.validate_room_name(room_name)
        capacity = self.validate_max_participants(max_participants)

        created = False
        room = self._manager.get_room(room_id)
        if room is None:
            if not create_if_missing:
                raise RoomNotFoundError(room_id)
            room = self._manager.create_room(
                room_id, max_participants=capacity, created_by=participant.participant_id
            )
            created = True
        # An existing room keeps its own capacity. A client must not be
        # able to raise the limit — or lower it below the current
        # occupancy — just by joining. Use update_room_capacity() for that.

        self._manager.join_room(participant, room_id)
        return room, created

    def close_room(self, room_name: str) -> RoomCloseResult:
        room_id = self.validate_room_name(room_name)
        room = self._manager.get_room(room_id)
        if room is None:
            raise RoomNotFoundError(room_id)
        removed = tuple(p.participant_id for p in room.participants)
        self._manager.close_room(room_id)
        return RoomCloseResult(room_id=room_id, removed_participant_ids=removed)

    def remove_participant(self, participant_id: str) -> ParticipantRemovalResult:
        participant = self._manager.get_participant(participant_id)
        if participant is None:
            raise ParticipantNotFoundError(participant_id)
        room_id = participant.room_id
        self._manager.unregister_participant_by_id(participant_id)
        room_deleted = bool(room_id) and not self._manager.has_room(room_id)
        return ParticipantRemovalResult(
            participant_id=participant_id, room_id=room_id, room_deleted=room_deleted
        )

    def update_room_capacity(self, room_name: str, max_participants: int) -> Room:
        room_id = self.validate_room_name(room_name)
        capacity = self.validate_max_participants(max_participants)
        room = self._manager.get_room(room_id)
        if room is None:
            raise RoomNotFoundError(room_id)
        if capacity < room.participant_count:
            raise RoomCapacityError(
                f"cannot shrink capacity to {capacity} below {room.participant_count} "
                "current participants"
            )
        room.max_participants = capacity
        return room

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_rooms(self) -> list[Room]:
        return self._manager.list_rooms()

    def list_room_participants(self, room_name: str) -> list[Participant]:
        room_id = self.validate_room_name(room_name)
        room = self._manager.get_room(room_id)
        if room is None:
            raise RoomNotFoundError(room_id)
        return room.participants
