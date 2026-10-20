# The nitpick SDK

The SDK is how you drive a nitpick room from your own code — a Python
service, a test harness, a second agent, or anything that speaks the
signaling protocol.

For how the framework itself works, see [architecture.md](architecture.md).

---

## What the SDK is

nitpick is a **room-based** system. A room holds participants; participants
exchange audio and events. The agent is a participant, not a special case.

That gives you two ways to build on it:

**In-process (the `Agent` facade).** The agent runs inside the server
process, registered directly with the `RoomManager`. This is the path for
"build an agent and run it locally":

```python
from nitpick import Agent

agent = Agent(name="Nitpick", prompt="You are friendly.")
agent.run(room_id="demo", port=8000)
```

**Out-of-process (the SDK).** A separate process connects over the
signaling WebSocket, joins a room, and talks to the agent as a peer. Use
this when the agent is a service, when you want to test from outside the
process, or when you want several clients.

```python
from nitpick import RoomClient

async with RoomClient("ws://localhost:8000/ws", name="My App") as client:
    await client.join_room("demo")
    await client.publish_pcm(pcm, sample_rate=16000)
```

---

## The client

### `RoomClient`

```python
RoomClient(websocket_url: str, *, name: str = "SDK User")
```

| Method | Purpose |
|---|---|
| `connect()` | Open the socket and wait for the server's greeting |
| `close()` | Close the socket |
| `join_room(room_id, *, max_participants=10)` | Join an existing room |
| `create_room(room_id, *, max_participants=10)` | Create a room, then join it |
| `leave_room()` | Leave the current room |
| `publish_pcm(pcm, *, sample_rate=16000, channels=1)` | Stream raw PCM to the room |
| `publish_audio(audio, *, mime_type="audio/pcm")` | Fan out an opaque audio blob |
| `set_speaking(speaking)` | Report a client-side speaking transition |
| `send_transcript(text, *, is_final=True)` | Report a transcript obtained elsewhere |
| `send_offer(target, sdp)` | Relay a WebRTC offer |
| `send_answer(target, sdp)` | Relay a WebRTC answer |
| `send_ice_candidate(target, candidate)` | Relay an ICE candidate |
| `recv()` | Receive the next message (`None` once disconnected) |
| `messages()` | Async-iterate messages |
| `send(payload)` | Send a raw message |

It is an async context manager, so `connect`/`close` are handled for you:

```python
async with RoomClient(url, name="Tester") as client:
    ...
```

### A worked example

Driving the agent from a script — send some audio, read the reply:

```python
import asyncio
import base64
import wave

from nitpick import RoomClient


def read_wav(path: str) -> tuple[bytes, int]:
    with wave.open(path, "rb") as wav:
        assert wav.getsampwidth() == 2 and wav.getnchannels() == 1
        return wav.readframes(wav.getnframes()), wav.getframerate()


async def main() -> None:
    pcm, rate = read_wav("question.wav")

    async with RoomClient("ws://localhost:8000/ws", name="Script") as client:
        await client.join_room("demo")

        # Stream 20 ms frames.
        frame = rate * 2 // 100 * 2
        for offset in range(0, len(pcm), frame):
            await client.publish_pcm(pcm[offset : offset + frame], sample_rate=rate)
            await asyncio.sleep(0.02)

        # Tell the agent the turn is over.
        await client.set_speaking(False)

        # Collect the reply.
        async for message in client.messages():
            if message["type"] == "room_audio":
                print("got", message["audio_bytes"], "bytes of audio")
            elif message["type"] == "room_interrupt":
                print("interrupted")


asyncio.run(main())
```

---

## The room model

### Rooms

A room is a named group of participants with a capacity limit.

```python
room = Room(room_id="demo", max_participants=10)
```

`RoomControlModule` is the validated facade over the manager — it holds no
state of its own, only rules:

```python
control = RoomControlModule(manager)
control.create_room("demo", max_participants=10)
control.join_participant(participant, room_name="demo")
control.list_room_participants("demo")
control.close_room("demo")
```

Errors are typed: `RoomAlreadyExistsError`, `RoomCapacityError`,
`ParticipantNotFoundError`, `ParticipantNotInRoomControlError`,
`RoomNotFoundError`.

### Participants

```python
@dataclass
class Participant:
    participant_id: str
    name: str
    role: ParticipantRole      # local | remote | agent
    room_id: str | None
    state: ParticipantState    # connecting | in_room | disconnected
    is_speaking: bool
    created_at: float
```

A participant is addressed by its `participant_id`, which is what the
WebRTC relay messages use as `target_participant_id`.

### The agent is a participant

The agent joins a room with `role="agent"`. It is addressable, it receives
room events, and it publishes audio like anyone else. Nothing about the
agent is special to the room layer.

---

## The event bus

`RoomManager` is the transport-agnostic core. The signaling server is a
protocol adapter on top of it; the agent runtime is another consumer.

```python
manager = RoomManager()
manager.add_event_listener(lambda event: print(event))
```

Listeners receive a **plain dict**, not an event object, so adding a field
never breaks a listener.

### Event types

| `event_type` | Meaning |
|---|---|
| `room_created` | A room was created |
| `room_destroyed` | A room was destroyed (it emptied) |
| `participant_joined` | Someone joined |
| `participant_left` | Someone left |
| `participant_speaking_started` | A participant started speaking |
| `participant_speaking_stopped` | A participant stopped speaking |
| `participant_transcript` | A participant produced a transcript |
| `participant_audio` | Raw PCM from a participant (in-process only) |

### Event shape

```json
{
  "event_id": "evt-1a2b3c4d5e6f",
  "event_type": "participant_transcript",
  "room_id": "demo",
  "participant_id": "p-9f8e7d6c5b4a",
  "participant_name": "Tester",
  "participant_role": "local",
  "is_speaking": false,
  "participant_count": 2,
  "created_at": 1763000000.123,
  "transcript_text": "hello",
  "transcript_is_final": true,
  "transcript_source": "nitpick_sdk"
}
```

`transcript_text`, `transcript_is_final` and `transcript_source` are
present only on transcript events. `audio`, `sample_rate` and `channels` are
present only on audio events.

### Broadcast rules

The five participant events are pushed to the room over the WebSocket as a
single envelope:

```json
{ "type": "room_event", "event": { ... } }
```

They go to everyone **except** the participant the event is about — a
participant already knows their own state.

`room_created` and `room_destroyed` stay in the event history and reach
in-process listeners only.

`participant_audio` is never broadcast over the WebSocket. It is
delivered to in-process listeners only; broadcasting every audio frame as
base64 to every peer would be wasteful.

### Event history

The manager keeps the last 200 events:

```python
manager.list_events()   # a list of plain dicts, oldest first
manager.stats()         # {"rooms": 1, "participants": 2, ...}
```

---

## The signaling protocol

One WebSocket. Flat JSON objects. Every message has a `type`.

### Endpoint

```
ws://<host>:<port>/ws
```

### Client → server

#### `join_room`

```json
{ "type": "join_room", "room_id": "demo", "name": "Tester", "max_participants": 10 }
```

`room_id` is required. `name` defaults to `"Anonymous"`. The room is created
if it does not exist, unless `REQUIRE_EXPLICIT_ROOM_CREATE=true`.

#### `create_room`

```json
{ "type": "create_room", "room_id": "demo", "name": "Tester", "max_participants": 10 }
```

Creates the room, then joins it. Fails with `room_already_exists` if the
room exists. Accepts optional HMAC authentication fields — see
[Authentication](#authentication).

#### `leave_room`

```json
{ "type": "leave_room" }
```

#### `participant_activity`

```json
{ "type": "participant_activity", "activity": "speaking_started" }
```

`activity` is `speaking_started` or `speaking_stopped`. This is the
client-side VAD reporting a turn boundary. The server's own VAD
coordinates with it, so sending both is safe — exactly one turn boundary
is emitted per utterance.

#### `audio_pcm`

```json
{
  "type": "audio_pcm",
  "audio_base64": "...",
  "sample_rate": 16000,
  "channels": 1
}
```

**The streaming path.** Raw 16-bit little-endian mono PCM, in whatever
frame size you like (20 ms is typical). This is what the agent's STT
service listens on.

#### `stt_transcript`

```json
{ "type": "stt_transcript", "text": "hello", "is_final": true, "source": "browser_web_speech" }
```

For clients that transcribe locally (the browser's Web Speech API) and
send text rather than audio.

#### `publish_audio`

```json
{ "type": "publish_audio", "audio_base64": "...", "mime_type": "audio/pcm", "sequence": 12 }
```

Fans an opaque audio blob out to the room. `sequence` is optional.

#### `webrtc_offer` / `webrtc_answer` / `ice_candidate`

```json
{ "type": "webrtc_offer", "target_participant_id": "p-1234", "sdp": { ... } }
{ "type": "ice_candidate", "target_participant_id": "p-1234", "candidate": { ... } }
```

Relayed to a single peer. The server never parses the SDP or the candidate
— it copies the opaque blob and adds routing metadata. The target must be
in the same room.

### Server → client

#### `connected`

Sent immediately after the socket opens.

```json
{
  "type": "connected",
  "participant_id": "p-9f8e7d6c5b4a",
  "stt": { "provider": "pcm", "transport": "audio_pcm" },
  "audio": { "output_sample_rate": 24000 }
}
```

`audio.output_sample_rate` tells the client what rate the agent's audio
arrives at, so it can play it without guessing.

#### `joined_room`

```json
{
  "type": "joined_room",
  "room": { "room_id": "demo", "max_participants": 10, "participant_count": 2, "created_at": 1763000000.0, "created_by": "p-1234" },
  "self": { "participant_id": "p-9f8e...", "name": "Tester", "role": "local", "room_id": "demo", "state": "in_room", "is_speaking": false, "created_at": 1763000000.0 },
  "participants": [ ... other participants ... ],
  "room_created": true,
  "join_source": "create_room"
}
```

`participants` lists the *other* members. `join_source` is `create_room` or
`join_room`.

#### `left_room`

```json
{ "type": "left_room", "room_id": "demo", "room_exists": false, "room_deleted": true }
```

#### `room_event`

```json
{ "type": "room_event", "event": { ... } }
```

See [Event shape](#event-shape).

#### `room_audio`

```json
{
  "type": "room_audio",
  "room_id": "demo",
  "from_participant_id": "p-9f8e...",
  "mime_type": "audio/pcm",
  "audio_base64": "...",
  "audio_bytes": 9600,
  "created_at": 1763000000.5
}
```

Audio published into the room — this is how the agent's reply reaches the
browser.

#### `room_interrupt`

```json
{ "type": "room_interrupt", "room_id": "demo", "from_participant_id": "p-9f8e...", "created_at": 1763000000.5 }
```

The user interrupted. Clients should stop whatever audio they have queued.

#### `audio_published`

```json
{ "type": "audio_published", "room_id": "demo", "audio_bytes": 9600, "sequence": null }
```

Acknowledges a `publish_audio` to the publisher only.

#### `webrtc_offer` / `webrtc_answer` / `ice_candidate`

```json
{ "type": "webrtc_offer", "from_participant_id": "p-1234", "room_id": "demo", "sdp": { ... } }
```

Relayed from a peer, with the sender identified.

#### `error`

```json
{ "type": "error", "code": "room_not_found", "message": "room 'demo' does not exist" }
```

`code` is stable and machine-readable; `message` is for humans.

### Error codes

| Code | Cause |
|---|---|
| `invalid_json` | The payload was not valid JSON |
| `invalid_payload` | The payload was not a JSON object |
| `missing_type` | No `type` field |
| `unsupported_type` | Unknown message type |
| `invalid_room_id` | `room_id` missing or not a string |
| `invalid_max_participants` | Capacity out of bounds or not an integer |
| `room_already_exists` | `create_room` on an existing room |
| `room_not_found` | Joining a room that does not exist |
| `room_full` | The room is at capacity |
| `not_in_room` | The message needs a room membership first |
| `invalid_activity` | Unknown `activity` value |
| `invalid_stt_payload` | `text` was not a string |
| `invalid_audio_payload` | Missing, empty or non-base64 audio |
| `audio_chunk_too_large` | Audio exceeded 1 MiB |
| `invalid_target` | `target_participant_id` missing |
| `target_not_found` | No such participant |
| `target_not_in_room` | The target is in a different room |
| `create_room_auth_not_configured` | Auth required but not configured |
| `invalid_create_room_auth` | `api_token` / `api_ts` missing |
| `unauthorized_create_room` | The token did not match |
| `expired_create_room_token` | The token was outside the TTL |

---

## Authentication

`create_room` can require an HMAC-SHA256 signature. The client computes:

```
token = HMAC_SHA256(api_secret, f"{api_key}:{room_id}:{api_ts}")
```

where `api_ts` is a unix timestamp. The server accepts it if the clock skew
is within `ROOM_CREATE_TOKEN_TTL_SECONDS` (default 300) and the signature
matches.

```json
{
  "type": "create_room",
  "room_id": "demo",
  "api_key": "your-key",
  "api_ts": 1763000000,
  "api_token": "the-hex-digest"
}
```

Enabled by setting both `ROOM_CREATE_API_KEY` and `ROOM_CREATE_API_SECRET`.
`join_room` is never authenticated — it is only meaningful for rooms that
already exist.

---

## Writing your own client

The SDK is a convenience, not a requirement. The protocol is small enough
to implement in any language.

The minimum a client must do:

1. Connect to `/ws`.
2. Read the `connected` message; keep `participant_id`.
3. Send `join_room`.
4. Read the `joined_room` message.
5. Send `audio_pcm` frames while the user talks.
6. Send `participant_activity` with `speaking_stopped` when they stop.
7. Read `room_audio` messages and play them.
8. On `room_interrupt`, stop what you are playing.

In the browser, add WebRTC on top: on `room_event` with
`event_type == "participant_joined"`, create an `RTCPeerConnection`, make an
offer, and send it as `webrtc_offer` to that participant. Answer incoming
offers and relay ICE candidates.

The browser client shipped with nitpick does exactly this — see
`nitpick/web/index.html` for a complete, working reference implementation.

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `ROOM_CREATE_API_KEY` | unset | Key accepted for `create_room` |
| `ROOM_CREATE_API_SECRET` | unset | HMAC secret |
| `REQUIRE_CREATE_ROOM_AUTH` | `true` when both are set | Require signed tokens |
| `ROOM_CREATE_TOKEN_TTL_SECONDS` | `300` | Accepted clock skew |
| `REQUIRE_EXPLICIT_ROOM_CREATE` | `false` | Require rooms to be created first |
| `MAX_ROOM_PARTICIPANTS` | `100` | Upper bound on client-requested capacity |

---

## Extending the SDK

`RoomClient` is a plain class. To add behaviour, subclass it:

```python
class VoiceClient(RoomClient):
    async def on_transcript(self, text: str, is_final: bool) -> None:
        print("heard:", text)

    async def messages(self):
        async for message in super().messages():
            if message.get("type") == "room_event":
                event = message["event"]
                if event.get("event_type") == "participant_transcript":
                    await self.on_transcript(
                        event["transcript_text"], event["transcript_is_final"]
                    )
            yield message
```

Or skip the SDK entirely and use the `RoomManager` in-process, which gives
you typed events instead of dicts and no serialisation at all.
