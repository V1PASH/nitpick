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

Python 3.10+. The core install pulls in `fastapi`, `uvicorn`, `websockets`,
`httpx` and `numpy` — nothing else, and no model downloads.

Set the keys for the services you use:

```bash
export OPENAI_API_KEY=sk-...      # STT, LLM and TTS
export DEEPGRAM_API_KEY=...       # optional: streaming STT
export SARVAM_API_KEY=...         # optional: Sarvam STT (Indian languages)
```

### Optional extras

| Extra | Adds | Why |
|---|---|---|
| `pip install nitpick[onnx]` | Silero VAD | A neural VAD that handles noisy rooms |
| `pip install nitpick[webrtcvad]` | WebRTC VAD | Google's model; needs a C compiler to build |
| `pip install nitpick[webrtc]` | `aiortc` | The agent as a native WebRTC peer |
| `pip install nitpick[all]` | all of the above | |

The default VAD backend is `energy` — pure Python, no dependencies, no model
file. It works straight after `pip install`.

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

### From your own code — the SDK

```python
from nitpick import RoomClient

async with RoomClient("ws://localhost:8000/ws", name="My App") as client:
    await client.join_room("demo")
    await client.publish_pcm(pcm, sample_rate=16000)
    async for message in client.messages():
        ...
```

---

## Architecture

```
                      ┌──────────── PipelineTask ────────────┐
  browser  ──ws──►   │  RoomTransport.input                  │
                      │        ↓                              │
                      │  VadProcessor      ← turn boundaries  │
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
                      │  InterruptionHandler ← barge-in        │
                      │        ↓                              │
                      │  RoomTransport.output                 │
                      └───────────────────────────────────────┘
```

### Packages

| Package | What lives there |
|---|---|
| `nitpick.frames` | `Frame`, `FrameKind`, `FrameProcessor`, the two-lane queues |
| `nitpick.pipeline` | `Pipeline`, `PipelineTask`, `PipelineParams` |
| `nitpick.vad` | `StateMachine`, the `Energy`/`WebRtc`/`Silero` backends, `VadProcessor` |
| `nitpick.rooms` | `Room`, `Participant`, `RoomManager`, `RoomControlModule` |
| `nitpick.signaling` | The FastAPI signaling server and the WebSocket protocol |
| `nitpick.transport` | `RoomTransport`, `ChannelTransport`, `WebRtcTransport` |
| `nitpick.services.stt` | `SttProvider` protocol, `SttService`, OpenAI + Deepgram + Sarvam |
| `nitpick.services.llm` | `LLMService` — streaming, tool calls, tool loop |
| `nitpick.services.tts` | `TTSService` — sentence buffering, synthesis |
| `nitpick.processors` | The user, assistant and interruption processors |
| `nitpick.billing` | Per-session usage and transcript events |
| `nitpick.sdk` | `RoomClient` — drive a room from your own code |
| `nitpick.agent` | The `Agent` facade — the API you actually use |

---

## Turn detection

The VAD is a four-state machine — `QUIET → STARTING → SPEAKING → STOPPING` —
that debounces both edges, so a single cough does not open a turn and a
pause between words does not close one.

Three backends behind one API:

| Backend | Install | Notes |
|---|---|---|
| `energy` *(default)* | — | Pure Python, adaptive noise floor. No dependencies. |
| `webrtc` | `nitpick[webrtcvad]` | Google's model. Needs a C compiler. |
| `silero` | `nitpick[onnx]` | Neural. Best in noisy rooms; the model is downloaded once. |

```python
agent = Agent(vad="silero")     # or "energy", "webrtc", or None
```

**Client and server VAD coordination.** The browser reports its own speaking
edges, and the server has its own VAD. A shared `SpeakingGate` ensures
exactly one turn boundary per utterance — whichever side detects speech
first wins, and the other becomes a no-op.

---

## Barge-in

When the user starts speaking while the bot is still talking, the bot stops.
The `InterruptionHandler` watches for that, broadcasts an interruption, and:

* the LLM service aborts its in-flight request,
* the TTS service drops its buffer,
* the room gets a `room_interrupt` message so the browser stops playback.

```python
agent = Agent(allow_interruptions=True)   # the default
```

---

## Usage tracking

`SessionBilling` records what a session consumed — LLM tokens, TTS
characters, STT duration, and the full transcript.

```python
from nitpick import LogBillingStorage, SessionBilling

billing = SessionBilling("session-42")
billing.attach(LogBillingStorage("usage.jsonl"))
billing.start()
```

Recording is a single `put_nowait` onto an unbounded queue drained by a
background task, so it never blocks and never drops. Cost tracking costs
the audio path nothing.

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
`room_interrupt`, `audio_published`, `webrtc_offer`, `webrtc_answer`,
`ice_candidate`, `error`.

WebRTC is native in the browser. The server never touches media for a
browser-to-browser call — it relays SDP and ICE to a single peer and checks
that both peers are in the same room.

### The agent as a native WebRTC peer

With `pip install nitpick[webrtc]`, `WebRtcTransport` makes the agent a real
WebRTC endpoint: the browser negotiates with it directly and audio travels
as Opus over SRTP, with nothing relayed through the signaling socket.

---

## Tests

```bash
pytest
```

56 tests covering the two-lane frame ordering, the transcript/turn-close
guarantee, the turn gate's release timeout, the VAD state machine and its
drain loop, client/server VAD coordination, barge-in, billing, the room
event bus, the signaling protocol, the SDK, and a full browser-to-agent turn.

## License

BSD-2-Clause.
