"""Tests for the rooms, the event bus, and the signaling protocol."""
from __future__ import annotations

import asyncio

import pytest

from nitpick.rooms import RoomControlModule, RoomManager
from nitpick.signaling.server import SignalingServer


class FakeWebSocket:
    """Captures everything the server sends."""

    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)

    def messages_of_type(self, message_type: str) -> list[dict]:
        return [m for m in self.sent if m.get("type") == message_type]


def test_room_capacity_is_enforced():
    manager = RoomManager()
    control = RoomControlModule(manager)
    control.create_room("demo", max_participants=2)

    first = manager.register_participant(None, "a")
    second = manager.register_participant(None, "b")
    manager.join_room(first, "demo")
    manager.join_room(second, "demo")

    third = manager.register_participant(None, "c")
    with pytest.raises(Exception):
        manager.join_room(third, "demo")


def test_speaking_events_are_edge_triggered():
    manager = RoomManager()
    participant = manager.register_participant(None, "a")
    manager.join_room(participant, "demo")

    first = manager.update_participant_speaking(participant, True)
    second = manager.update_participant_speaking(participant, True)

    assert first is not None
    assert second is None, "a repeated speaking state must not emit an event"


@pytest.mark.asyncio
async def test_events_reach_listeners():
    manager = RoomManager()
    received: list[dict] = []
    manager.add_event_listener(received.append)

    participant = manager.register_participant(None, "a")
    manager.join_room(participant, "demo")
    manager.emit_participant_transcript(participant, transcript_text="hello")

    assert any(e["event_type"] == "participant_transcript" for e in received)
    assert any(e["event_type"] == "participant_joined" for e in received)


@pytest.mark.asyncio
async def test_async_listener_exceptions_do_not_break_the_bus():
    manager = RoomManager()
    calls: list[str] = []

    async def bad_listener(event: dict) -> None:
        calls.append("bad")
        raise RuntimeError("boom")

    def good_listener(event: dict) -> None:
        calls.append("good")

    manager.add_event_listener(bad_listener)
    manager.add_event_listener(good_listener)

    participant = manager.register_participant(None, "a")
    manager.join_room(participant, "demo")

    await asyncio.sleep(0.05)
    assert "good" in calls


@pytest.mark.asyncio
async def test_join_room_returns_the_room_and_participants():
    server = SignalingServer()
    ws = FakeWebSocket()

    response = await server.handle_message(
        ws, {"type": "join_room", "room_id": "demo", "name": "Alice"}
    )

    assert response is not None
    assert response["type"] == "joined_room"
    assert response["room"]["room_id"] == "demo"
    assert response["self"]["name"] == "Alice"


@pytest.mark.asyncio
async def test_unknown_message_type_is_an_error_not_a_crash():
    server = SignalingServer()
    ws = FakeWebSocket()

    response = await server.handle_message(ws, {"type": "nonsense"})

    assert response["type"] == "error"
    assert response["code"] == "unsupported_type"


@pytest.mark.asyncio
async def test_pcm_audio_reaches_in_process_listeners():
    """The streaming path must deliver raw PCM to the agent."""
    server = SignalingServer()
    ws = FakeWebSocket()
    received: list[dict] = []
    server.manager.add_event_listener(received.append)

    await server.handle_message(ws, {"type": "join_room", "room_id": "demo"})

    import base64

    pcm = base64.b64encode(b"\x00\x01" * 160).decode("ascii")
    response = await server.handle_message(
        ws,
        {
            "type": "audio_pcm",
            "audio_base64": pcm,
            "sample_rate": 16000,
            "channels": 1,
        },
    )

    assert response is None
    audio_events = [e for e in received if e["event_type"] == "participant_audio"]
    assert len(audio_events) == 1
    assert audio_events[0]["sample_rate"] == 16000
    assert len(audio_events[0]["audio"]) == 320


@pytest.mark.asyncio
async def test_joining_a_full_room_reports_room_full():
    """A full room must say so, not claim the room does not exist."""
    server = SignalingServer()
    server.control.create_room("demo", max_participants=1)

    first = FakeWebSocket()
    await server.handle_message(first, {"type": "join_room", "room_id": "demo"})

    second = FakeWebSocket()
    response = await server.handle_message(
        second, {"type": "join_room", "room_id": "demo"}
    )

    assert response["type"] == "error"
    assert response["code"] == "room_full"


def test_joining_does_not_change_an_existing_rooms_capacity():
    """A client must not be able to resize a room by joining it."""
    manager = RoomManager()
    control = RoomControlModule(manager)
    control.create_room("demo", max_participants=2)

    participant = manager.register_participant(None, "a")
    control.join_participant(participant, room_name="demo")

    # A second joiner asks for a much larger room.
    other = manager.register_participant(None, "b")
    control.join_participant(other, room_name="demo", max_participants=50)

    room = manager.get_room("demo")
    assert room is not None
    assert room.max_participants == 2, "the room's capacity must be unchanged"
    assert room.participant_count == 2


@pytest.mark.asyncio
async def test_an_invalid_capacity_is_reported_precisely():
    server = SignalingServer()
    ws = FakeWebSocket()

    response = await server.handle_message(
        ws,
        {"type": "join_room", "room_id": "demo", "max_participants": "not-a-number"},
    )

    assert response["type"] == "error"
    assert response["code"] == "invalid_max_participants"


@pytest.mark.asyncio
async def test_an_over_long_room_id_is_rejected():
    server = SignalingServer()
    ws = FakeWebSocket()

    response = await server.handle_message(
        ws, {"type": "join_room", "room_id": "x" * 200}
    )

    assert response["type"] == "error"
    assert response["code"] == "invalid_room_id"


@pytest.mark.asyncio
async def test_stt_audio_chunk_is_not_a_supported_type():
    """The batch chunk path is not implemented — it must say so clearly."""
    server = SignalingServer()
    ws = FakeWebSocket()

    response = await server.handle_message(
        ws, {"type": "stt_audio_chunk", "audio_base64": "AAAA"}
    )

    assert response["type"] == "error"
    assert response["code"] == "unsupported_type"


@pytest.mark.asyncio
async def test_webrtc_relay_targets_one_peer_only():
    server = SignalingServer()
    sender_ws = FakeWebSocket()
    target_ws = FakeWebSocket()
    other_ws = FakeWebSocket()

    await server.handle_message(sender_ws, {"type": "join_room", "room_id": "demo"})
    await server.handle_message(target_ws, {"type": "join_room", "room_id": "demo"})
    await server.handle_message(other_ws, {"type": "join_room", "room_id": "demo"})

    target_id = server.manager.get_participant_by_websocket(target_ws).participant_id
    sender_id = server.manager.get_participant_by_websocket(sender_ws).participant_id

    response = await server.handle_message(
        sender_ws,
        {"type": "webrtc_offer", "target_participant_id": target_id, "sdp": {"sdp": "v=0"}},
    )

    assert response is None
    relayed = target_ws.messages_of_type("webrtc_offer")
    assert len(relayed) == 1
    assert relayed[0]["from_participant_id"] == sender_id
    assert not other_ws.messages_of_type("webrtc_offer")


@pytest.mark.asyncio
async def test_webrtc_relay_rejects_a_peer_in_another_room():
    server = SignalingServer()
    sender_ws = FakeWebSocket()
    target_ws = FakeWebSocket()

    await server.handle_message(sender_ws, {"type": "join_room", "room_id": "room-a"})
    await server.handle_message(target_ws, {"type": "join_room", "room_id": "room-b"})

    target_id = server.manager.get_participant_by_websocket(target_ws).participant_id
    response = await server.handle_message(
        sender_ws,
        {"type": "ice_candidate", "target_participant_id": target_id, "candidate": {}},
    )

    assert response["type"] == "error"
    assert response["code"] == "target_not_in_room"
