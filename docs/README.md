# nitpick documentation

| Document | What it covers |
|---|---|
| [architecture.md](architecture.md) | How the framework works — frames, the two-lane processor, the pipeline, turn detection, interruption, rooms, billing, and the design trade-offs behind each |
| [sdk.md](sdk.md) | The client SDK and the signaling protocol — the room model, the event bus, every message type, every error code, and how to write your own client |

Start with the [README](../README.md) for installation and a quick start.

---

## Reading order

**If you are building an agent**, read `architecture.md` in order. The
sections build on each other: the frame taxonomy, then the two-lane
processor, then the pipeline, and then the turn-close guarantee — which is
the idea that makes the rest cohere.

**If you are integrating with nitpick** — driving a room from your own code
— read `sdk.md`. You only need the first two sections of `architecture.md`
(the mental model and the room model) for context.

**If you are extending nitpick**, `architecture.md` ends with the extension
points: adding an STT backend, a processor, a transport, or a tool.

---

## The three things worth knowing

If you read nothing else:

**1. System frames overtake data frames.**
Each processor keeps two lanes. Lifecycle frames are processed inline and
jump the data queue, so an in-flight LLM call can never delay a barge-in.

**2. The transcript rides inside the turn-close frame.**
Because of (1), a transcript and a "user stopped speaking" as two separate
frames can reorder — and the agent silently never answers. nitpick makes the
turn-close frame carry the transcript, so the ordering is structural.

**3. The agent is just a participant.**
A room holds participants; the agent joins one like any other. There is no
special case in the room layer, which is why the SDK can drive an agent the
same way it drives a browser.
