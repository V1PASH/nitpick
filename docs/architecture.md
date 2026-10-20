# Architecture

How nitpick works, from the frame up.

This document explains the design and the reasoning behind it. For the wire
protocol and the client SDK, see [sdk.md](sdk.md).

---

## The mental model

nitpick is one idea applied three times.

A **frame** is a typed message. A **processor** is a node that consumes frames
and pushes them onward. A **pipeline** is a chain of processors.

Every component in the system — the VAD, the STT service, the LLM, the TTS,
the transport, the pipeline itself — is the same type: a `FrameProcessor` with
a handler. There is no separate "streaming service" interface and no separate
"transport" interface. Adding a component means writing one handler.

```
browser ──▶ RoomTransport.input ──▶ VadProcessor ──▶ SttService
            ──▶ UserAggregator ──▶ LLMService ──▶ AssistantAggregator
            ──▶ TTSService ──▶ InterruptionHandler ──▶ RoomTransport.output
                                                                       │
            ◀─────────────────────────────────────────────────────────┘
```

---

## Layer 1 — Frames

`src/frames/`

A frame is a flat enum with a typed payload, not a class hierarchy:

```python
class FrameKind(str, Enum):
    START = "start"
    VAD_USER_STARTED_SPEAKING = "vad_user_started_speaking"
    TRANSCRIPTION = "transcription"
    LLM_TEXT = "llm_text"
    OUTPUT_AUDIO_RAW = "output_audio_raw"
    ...

@dataclass
class Frame:
    kind: FrameKind
    payload: Any = None
    id: int = field(default_factory=next_frame_id)
    sibling_id: int | None = None
```

Three families, distinguished by `FrameKind.is_system` / `.is_control` /
`.is_data`:

| Family | Examples | Role |
|---|---|---|
| **System** | `start`, `cancel`, `interruption`, `vad_user_started_speaking`, `input_audio_raw` | Lifecycle and control-of-the-stream |
| **Control** | `end`, `llm_full_response_start`, `function_call_start` | Boundaries around a unit of work |
| **Data** | `transcription`, `llm_text`, `output_audio_raw`, `function_call_result` | Payloads |

**Why flat rather than a hierarchy.** Dispatch is one attribute comparison
instead of an `isinstance` chain, and every frame kind is discoverable from a
single enum. Payloads stay typed through dedicated dataclasses
(`Transcription`, `AudioChunk`, `VadStopped`, `FunctionCall`).

**`id` and `sibling_id`.** Every frame gets a process-unique id. When a
frame is broadcast in both directions, the two copies carry each other's id
in `sibling_id`, so a consumer that sees both can tell they are the same
logical event.

---

## Layer 2 — The processor

`src/frames/processor.py`

This is the most important file in the framework.

### The two lanes

Each processor owns **two queues** and **two tasks**:

```
                  ┌─────────────────────────────┐
   queue_frame ──▶│  input lane (priority queue)│
                  └──────────────┬──────────────┘
                                 │
              system frame? ─────┴─────► yes ──▶ process INLINE
                                 │                    (on the input task)
                                 └──► no ──▶ process queue
                                                │
                  ┌─────────────────────────────┐▼──────┐
                  │  process lane (FIFO)               │
                  └────────────────────────────────────┘
```

The input lane is a `PriorityQueue`: system frames (priority 0) overtake
data frames (priority 1). The input task pulls from it and:

* **system frame** → processes it **inline**, on the input task
* **data frame** → forwards it to the process queue

The process task drains the process queue one frame at a time.

### Why this matters

Because system frames are processed inline on the input task, they **leap
the entire data queue**. An `INTERRUPTION` frame does not wait behind the
audio chunks already buffered. An in-flight LLM or TTS handler cannot delay
a barge-in.

Without the split — a single queue, everything in order — the user would
have to wait for the current audio backlog to drain before the bot could be
interrupted. On a 20 ms frame cadence with a slow TTS, that is audible.

The cost is one extra queue and one extra task per processor. That is the
whole price of responsive interruption.

### Frame delivery

`push_frame` hands a frame to the *neighbour's* queue:

```python
async def push_frame(self, frame, direction=DOWNSTREAM):
    target = self._next if direction == DOWNSTREAM else self.previous
    if target is None:
        return
    await target.queue_frame(frame, direction)
```

Frames are never dropped for being early. If the neighbour has not booted
yet, the frame simply waits in its queue. The `START` frame is the first
thing that flows through a pipeline, so anything queued behind it is
delivered in order once the pipeline is up.

> A note for anyone modifying this: an earlier version re-queued a frame
> into the processor's *own* input queue when it had not started. The input
> loop then picked it up, called the handler, the handler pushed it again,
> and the two spun forever. The neighbour's queue is the only safe place to
> park a frame.

### Processors link into a list

```python
self._next: FrameProcessor | None   # strong
self._prev: WeakRef | None          # weak, so a pipeline is not a cycle
```

`Pipeline` links them and is itself a `FrameProcessor`, so pipelines nest.

---

## Layer 3 — The pipeline

`src/pipeline/`

`PipelineTask` is the lifecycle wrapper. It bookends the user's processors
with two sentinels:

```
[TaskSource] ──▶ user processors ──▶ [TaskSink]
```

**TaskSource** is the upstream boundary. Task-control frames that escape
upstream are converted into real lifecycle frames and pushed back
downstream:

| Escaping frame | Becomes |
|---|---|
| `end_task` | `end` |
| `cancel_task` | `cancel` |
| `stop_task` | `stop` |
| `interruption_task` | a broadcast interruption |

**TaskSink** is the downstream boundary. It watches for `end`, `stop` and
`cancel` leaving the pipeline and records the finish reason.

`PipelineTask` also owns:

* the `START` frame that boots every processor
* a **push channel** — an `asyncio.Queue` that external code (a transport,
  an SDK client) pushes frames into
* optional heartbeats and an idle timeout
* graceful cleanup of every processor on exit

---

## The turn-close guarantee

This is the single most important behaviour in the framework, and the reason
nitpick exists as something other than a Pipecat clone.

### The failure mode

A voice turn produces two things:

1. a **transcript** — data
2. a **"the user stopped speaking"** signal — system

System frames overtake data frames. So if those are two separate frames, the
stop can arrive *first*:

```
  STT service ──▶ transcript      ──────────────────────────▶ LLM
                └─▶ user_stopped  ──▶ LLM      ← arrives first, turn closes
                                                     with no text
```

The aggregator sees a turn close with no transcript, has nothing to send the
LLM, and **the agent silently never answers**. There is no error, no log
line, no crash. The conversation just stops.

This is not a hypothetical. It is the single most common way a voice agent
"works" in testing and fails in production, because the race window depends
on timing.

### The fix

The transcript rides *inside* the stop frame:

```python
@dataclass
class VadStopped:
    stop_secs: float
    timestamp: float
    transcript: Transcription | None = None
```

One frame cannot be reordered against itself. The ordering becomes
structural rather than probabilistic.

### The machinery

`SttService` implements this with a **turn gate**:

1. The stop frame arrives → the service **stashes** it and asks the provider
   to finalise.
2. The provider returns a final transcript → the service emits **one** frame
   carrying both.
3. If the provider never answers, a **release timeout** (5 s by default)
   releases the stop frame without a transcript, so the turn is never left
   hanging.

`UserAggregator` then reads the transcript from the stop frame:

```python
if kind is FrameKind.VAD_USER_STOPPED_SPEAKING:
    stop = frame.payload
    await aggregator._close_turn(stop)   # stop.transcript holds the text
```

Because the aggregator triggers the LLM *from the stop frame*, it cannot
possibly run before the text arrives.

---

## A full turn, traced

```
browser                     server                                agent pipeline
───────                     ──────                                ──────────────

audio_pcm ─────────────────▶ room event
                            │
                            ├─▶ RoomTransport.input
                            │      └─▶ INPUT_AUDIO_RAW frame
                            │             │
                            │             ├─▶ VadProcessor
                            │             │      confidence 0.91 → SPEAKING
                            │             │      └─▶ VAD_USER_STARTED_SPEAKING
                            │             │
                            │             └─▶ SttService ──▶ provider.push_audio()
                            │
                            └─▶ ... 2.5 s of silence ...
                                   VadProcessor: QUIET
                                   └─▶ VAD_USER_STOPPED_SPEAKING
                                          │
                                          └─▶ SttService: stash the stop
                                                 provider.flush()
                                                 provider → "what is the weather"
                                                 └─▶ VAD_USER_STOPPED_SPEAKING
                                                        {transcript: "what is the weather"}
                                                       │
                                                       └─▶ UserAggregator
                                                              context.add_user_message(...)
                                                              └─▶ LLM_CONTEXT frame
                                                                     │
                                                                     └─▶ LLMService
                                                                            streams tokens
                                                                            └─▶ LLM_TEXT frames
                                                                                   │
                                                                                   └─▶ AssistantAggregator
                                                                                          context.add_assistant_message(...)
                                                                                          │
                                                                                   ┌──────┴──────┐
                                                                                   │             │
                                                                              LLM_FULL_    LLM_TEXT
                                                                              RESPONSE_END
                                                                                   │
                                                                                   └─▶ TTSService
                                                                                          sentence buffered
                                                                                          └─▶ OUTPUT_AUDIO_RAW
                                                                                                 │
                                                                                       RoomTransport.output
                                                                                                 │
room_audio ◀────────────────────────────────────────────────────────────────────────────────┘
```

Note the two lanes in action: the `VAD_USER_STOPPED_SPEAKING` frame is a
system frame and overtakes any queued data, but it carries its own
transcript so nothing is lost by that overtaking.

---

## Turn detection

`src/vad/`

### The state machine

```
 QUIET ──speech──▶ STARTING ──start_secs──▶ SPEAKING
   ▲                   │                        │
   │                   └───silence──────────────┤
   │                                            ▼
   └────────stop_secs────────────────────── STOPPING
```

Both edges are debounced. `start_secs` (default 0.2 s) means a turn opens
only after that much *unbroken* speech — a cough does not open a turn.
`stop_secs` (default 0.8 s) means a turn closes at a sentence pause rather
than at every breath.

A speech probability below `confidence` **or** a volume below `min_volume`
counts as silence, so quiet noise never opens a turn.

### Backends

| Backend | Dependency | Notes |
|---|---|---|
| `energy` | none | Pure Python, adaptive noise floor. The default. |
| `webrtc` | `nitpick[webrtcvad]` | Google's model. Needs a C compiler to build. |
| `silero` | `nitpick[onnx]` | Neural. Best in noisy rooms. |

All three implement the same one-method interface:

```python
class VadBackend(Protocol):
    sample_rate: int
    def is_speech(self, window: bytes) -> float: ...
```

### The drain loop

A backend works on fixed windows (512 samples at 16 kHz). The audio arrives
in whatever size the transport produced — a browser sends 20 ms frames, a
Twilio serializer sends 960-sample frames.

The processor **drains every complete window in a chunk**, not just the
first:

```python
pending = chunk.data
while True:
    window = self._machine.next_window(pending)
    pending = b""
    if window is None:
        break
    ...
```

Calling `next_window` once per chunk leaves the remainder buffered, and the
VAD falls progressively further behind the live audio — permanently. This is
the single easiest bug to reintroduce here, and there is a test for it.

### Client and server coordination

The browser reports its own speaking edges, and the server has its own VAD.
Both feed a shared `SpeakingGate`:

```python
gate.claim_start()   # True for exactly one caller
gate.claim_stop()    # True for exactly one caller
```

Whichever side detects speech first wins; the other becomes a no-op. So
exactly one turn boundary is emitted per utterance, no matter how many
detectors are running.

A stop with no matching start still closes the turn — the gate only
suppresses a *second* stop for the same turn.

---

## Interruption (barge-in)

`src/processors/interruption.py`

When the user starts speaking while the bot is still talking:

```
VAD_USER_STARTED_SPEAKING
        │
        ▼
InterruptionHandler
   (bot is speaking? → yes)
        │
        ▼
broadcast INTERRUPTION  ─────▶ LLMService: abort the in-flight request
        │                   ─────▶ TTSService:  drop the buffer
        │                   ─────▶ RoomTransport.output: send `room_interrupt`
        │                                              │
        ▼                                              ▼
   the pipeline replays          the browser stops its audio sources
```

The `room_interrupt` message matters as much as the internal cancellation.
The agent's audio has *already* been sent to the browser by the time the
user interrupts — so without telling the browser to drop what it has
queued, the bot keeps talking over the user.

---

## Rooms and the event bus

`src/rooms/`

A **room** holds participants and multiplexes events between them.

```python
@dataclass
class Room:
    room_id: str
    max_participants: int
    created_at: float
    created_by: str | None
    _participants: dict[str, Participant]
```

Participants have a **role** (`local`, `remote`, `agent`) and a **state**
(`connecting`, `in_room`, `disconnected`). The agent is a participant like
any other — it joins a room and is addressed the same way.

`RoomManager` owns the state and the event bus:

```python
manager.add_event_listener(listener)   # listeners get a plain dict
manager.emit_participant_transcript(participant, transcript_text=..., ...)
```

Listeners receive a **plain dict**, not an event object, so a listener
written against one version keeps working when fields are added. Async
listeners are scheduled as tracked tasks — a fire-and-forget `create_task`
with no registry leaks tasks and swallows exceptions.

**Broadcast rules.** Five event types are fanned out to the room:

| Event | Broadcast to |
|---|---|
| `participant_joined` | everyone except the joiner |
| `participant_left` | everyone except the leaver |
| `participant_speaking_started` | everyone except the speaker |
| `participant_speaking_stopped` | everyone except the speaker |
| `participant_transcript` | everyone except the speaker |

`room_created` and `room_destroyed` exist in the event history but are never
sent over the wire.

**Participant audio is not broadcast.** It is delivered to in-process
listeners only — re-encoding every frame as base64 for every peer would be
wasteful. The agent listens in-process; the browser gets its audio over
WebRTC.

---

## Billing

`src/billing/`

Cost tracking must never touch audio latency, so recording is a single
non-blocking write:

```python
def record(self, event: BillingEvent) -> None:
    self._queue.put_nowait(event)   # unbounded — never blocks, never drops
```

A background task drains the queue and writes to storage (a JSON-lines file,
or memory for tests). The hot path costs one `put_nowait`.

The services report what they consume: the LLM service asks for
`stream_options.include_usage` and records token counts; the TTS service
records characters synthesised; the STT service records audio duration.

---

## Module map

| Module | Responsibility |
|---|---|
| `nitpick.frames` | `Frame`, `FrameKind`, `FrameProcessor`, the two lanes |
| `nitpick.pipeline` | `Pipeline`, `PipelineTask`, `PipelineParams` |
| `nitpick.vad` | The state machine, the backends, `VadProcessor` |
| `nitpick.rooms` | `Room`, `Participant`, `RoomManager`, `RoomControlModule` |
| `nitpick.signaling` | The FastAPI server and the WebSocket protocol |
| `nitpick.transport` | `RoomTransport`, `ChannelTransport`, `WebRtcTransport` |
| `nitpick.services.stt` | `SttProvider` (the protocol), `SttService` (the engine) |
| `nitpick.services.llm` | `LLMService` — streaming, tool calls, the tool loop |
| `nitpick.services.tts` | `TTSService` — sentence buffering, synthesis |
| `nitpick.processors` | The user, assistant and interruption processors |
| `nitpick.billing` | Usage and transcript events |
| `nitpick.sdk` | `RoomClient` — drive a room from your own code |
| `nitpick.agent` | The `Agent` facade — the public API |

---

## Extension points

### A new STT backend

Implement four methods. Everything else — turn detection, buffering, the
frame plumbing, the ordering guarantee — is handled by `SttService`:

```python
class MyProvider(SttProvider):
    name = "myprovider"
    sample_rate = 16_000

    async def start(self, on_event): ...
    async def push_audio(self, pcm: bytes): ...
    async def flush(self): ...
    async def stop(self): ...
```

`on_event` receives `SttEvent(kind="partial" | "final" | "error", text=...)`.
Report a `final` and the turn closes with the transcript attached.

### A new processor

Write a handler and link it in:

```python
class MyHandler:
    async def on_process_frame(self, processor, frame, direction):
        if frame.kind is FrameKind.LLM_TEXT:
            ...
        await processor.push_frame(frame, direction)

processor = FrameProcessor("my-processor", MyHandler())
```

### A new transport

Implement `input()` and `output()` returning two `FrameProcessor`s, and
push frames into the input. `RoomTransport`, `ChannelTransport` and
`WebRtcTransport` are all that shape.

### A new tool

```python
@tool("get_weather", "Look up the weather for a city.")
def get_weather(city: str) -> str:
    return f"It is 28°C in {city}."
```

The LLM service registers it, exposes its schema, and runs the
tool → result → re-invoke loop, bounded by `max_tool_rounds`.

---

## Design trade-offs

**Two queues per processor.** Costs a task and a queue per processor, buys
interrupts that are not stuck behind audio. Worth it.

**The transcript rides the stop frame.** Slightly awkward — a stop frame
carries text — but it makes a whole class of race condition impossible.

**Everything is in-process.** A voice agent is one asyncio event loop, so
components share memory freely. There is no IPC, no serialisation, no
schema negotiation. The cost is that the agent cannot be split across
processes without a redesign.

**The browser is a peer, not a client.** Browser-to-browser calls use
native WebRTC and never touch the server's media path. The server relays
SDP and ICE and nothing else.

**`energy` is the default VAD.** It has no dependencies and no model file,
so `pip install nitpick` always produces a working agent. `silero` is one
extra away for noisy environments.

---

## Known limits

* **Turn detection quality is backend-dependent.** The `energy` backend is
  fine in a quiet room; a noisy street wants `silero`.
* **The agent is not a native WebRTC peer by default.** `WebRtcTransport`
  exists (`nitpick[webrtc]`) but the default path relays PCM over the
  signaling socket.
* **No server-side resampling.** The STT provider's sample rate must match
  what the transport delivers.
* **Single-process.** One agent per process; there is no supervisor.
