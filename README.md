# nitpick

**A Python framework for building voice agents.** Compose an agent from STT,
LLM and TTS services, run it locally, and talk to it in the browser over
WebRTC.

```python
from nitpick import Agent, tool

@tool("get_weather", "Look up the weather for a city.")
def get_weather(city: str) -> str:
    return f"It is 28°C in {city}."

agent = Agent(
    name="Nitpick",
    prompt="You are a friendly voice assistant.",
    stt="openai",
    llm="openai:gpt-4o-mini",
    tts="openai:alloy",
    tools=[get_weather],
)

agent.run(room_id="demo", port=8000)
```

Open <http://localhost:8000> and talk to it.

---

## Why

Most voice-agent frameworks make you choose between a rich pipeline you have
to wire up by hand and a black box you cannot extend. nitpick gives you the
pipeline, but makes the common case one expression.

Three ideas shape the framework:

**1. Frames move through a linked list of processors.**
Every component — VAD, STT, LLM, TTS, transport, the pipeline itself — is the
same type: a `FrameProcessor` with a handler. Adding a service means writing
one handler, not learning a new abstraction.

**2. System frames overtake data frames.**
nitpick keeps two lanes per processor. Lifecycle frames (interruption,
cancel, "the user stopped speaking") are processed on a separate lane and
always jump ahead of queued data. An in-flight LLM call can never delay a
barge-in.

**3. The transcript and the turn-close are one frame.**
This is the detail that separates a framework that works from one that
mostly works. Because system frames overtake data frames, emitting a
transcript and a "user stopped speaking" signal as two separate frames means
the stop can arrive first — and an agent that sees a turn close with no text
**silently never answers**. nitpick makes the turn-close frame *carry* the
transcript, so the ordering is structural rather than probabilistic.

```python
@dataclass
class VadStopped:
    stop_secs: float
    timestamp: float
    transcript: Transcription | None = None   # bundled, not raced
```

---

## Install

```bash
git clone https://github.com/V1PASH/nitpick.git
cd nitpick
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Python 3.10+. Runtime dependencies are `fastapi`, `uvicorn`, `websockets` and
`httpx`.

Set the keys for the services you use:

```bash
export OPENAI_API_KEY=sk-...      # STT, LLM and TTS
export DEEPGRAM_API_KEY=...       # optional: streaming STT
```

---

## Running an agent

### From Python

```python
from nitpick import Agent

Agent(name="Assistant", prompt="You are helpful.").run(port=8000)
```

### From the CLI

```bash
nitpick serve --room demo --port 8000 --name Assistant
nitpick version
```

The agent serves a browser client at `/`, a health check at `/health`, and the
signaling socket at `/ws`.

---

## Architecture

```
                      ┌──────────── PipelineTask ────────────┐
  browser  ──ws──►   │  RoomTransport.input                  │
                      │        ↓                              │
                      │  SttService      ← turn gate          │
                      │        ↓                              │
                      │  UserAggregator  ← triggers the LLM   │
                      │        ↓                              │
                      │  LLMService      ← streaming + tools   │
                      │        ↓                              │
                      │  AssistantAggregator                  │
                      │        ↓                              │
                      │  TTSService      ← sentence buffering  │
                      │        ↓                              │
                      │  RoomTransport.output                 │
                      └───────────────────────────────────────┘
```

### Packages

| Package | What lives there |
|---|---|
| `nitpick.frames` | `Frame`, `FrameKind`, `FrameProcessor`, the two-lane queues |
| `nitpick.pipeline` | `Pipeline`, `PipelineTask`, `PipelineParams` |
| `nitpick.rooms` | `Room`, `Participant`, `RoomManager`, `RoomControlModule` |
| `nitpick.signaling` | The FastAPI signaling server and the WebSocket protocol |
| `nitpick.transport` | `RoomTransport` (agent side), `ChannelTransport` (testing) |
| `nitpick.services.stt` | `SttProvider` protocol, `SttService`, OpenAI + Deepgram |
| `nitpick.services.llm` | `LLMService` — streaming, tool calls, tool loop |
| `nitpick.services.tts` | `TTSService` — sentence buffering, synthesis |
| `nitpick.processors` | The user and assistant aggregators |
| `nitpick.agent` | The `Agent` facade — the API you actually use |

---

## Adding a service

An STT provider answers four questions and nothing else. Everything else —
turn detection, buffering, the frame plumbing, the ordering guarantee — is
handled by `SttService`:

```python
from nitpick.services.stt import SttProvider, SttService

class MyProvider(SttProvider):
    name = "myprovider"
    sample_rate = 16_000

    async def start(self, on_event): ...
    async def push_audio(self, pcm: bytes): ...
    async def flush(self): ...
    async def stop(self): ...

stt = SttService(MyProvider())
```

`on_event` receives `SttEvent(kind="partial"|"final"|"error", text=...)`.
Report a `final` and the turn closes with the transcript attached.

---

## Testing a pipeline

`ChannelTransport` drives a pipeline through plain queues — no server, no
network:

```python
from nitpick import ChannelTransport, PipelineParams, PipelineTask

transport = ChannelTransport()
task = PipelineTask([transport.input(), stt, transport.output()], PipelineParams())
task.start()
```

---

## The signaling protocol

One WebSocket, flat JSON messages.

**Client → server**

| Type | Purpose |
|---|---|
| `join_room` / `create_room` / `leave_room` | Room membership |
| `participant_activity` | `speaking_started` / `speaking_stopped` — turn boundaries |
| `audio_pcm` | **Raw PCM frames** — the streaming path to the agent |
| `stt_transcript` | A transcript from the browser's speech recogniser |
| `stt_audio_chunk` | A batch WebM/Opus chunk for server-side STT |
| `publish_audio` | Base64 audio fanned out to the room |
| `webrtc_offer` / `webrtc_answer` / `ice_candidate` | SDP and ICE relay |

**Server → client**

`connected`, `joined_room`, `left_room`, `room_event`, `room_audio`,
`audio_published`, `webrtc_offer`, `webrtc_answer`, `ice_candidate`, `error`.

WebRTC is native in the browser. The server never touches media for a
browser-to-browser call — it relays SDP and ICE to a single peer and checks
that both peers are in the same room.

---

## Tests

```bash
pytest
```

The suite covers the two-lane frame ordering, the transcript/turn-close
guarantee, the turn gate's release timeout, the room event bus, the
signaling protocol, and a full browser-to-agent turn.

---

## Status

Alpha. The pipeline core, rooms, signaling, STT/LLM/TTS services and the
browser client all work end to end. Not yet built: server-side VAD (turn
detection currently comes from the browser), barge-in interruption handling,
per-session billing, and an `aiortc` transport so the agent can be a native
WebRTC peer rather than relaying PCM over the signaling socket.

## License

BSD-2-Clause.
