"""nitpick — a Python framework for building voice agents.

Quick start::

    from nitpick import Agent, tool

    @tool("get_time", "Get the current time.")
    def get_time() -> str:
        return "It is half past three."

    agent = Agent(name="assistant", prompt="You are a friendly voice assistant.")
    agent.run(room_id="demo", port=8000)

Then open http://localhost:8000 and talk to it.
"""
from __future__ import annotations

from .agent import Agent, AgentConfig, ToolDefinition, tool
from .billing import (
    BillingCollector,
    BillingEvent,
    LogBillingStorage,
    MemoryBillingStorage,
    SessionBilling,
    TranscriptRole,
)
from .context import LLMContext, Message, ToolSchema, shared_context
from .frames import (
    AudioChunk,
    Frame,
    FrameKind,
    FrameProcessor,
    Transcription,
    VadStarted,
    VadStopped,
)
from .pipeline import FinishReason, Pipeline, PipelineParams, PipelineTask
from .processors.assistant_aggregator import AssistantAggregator
from .processors.interruption import InterruptionHandler
from .processors.user_aggregator import UserAggregator
from .rooms import (
    Participant,
    ParticipantRole,
    Room,
    RoomControlModule,
    RoomEventType,
    RoomManager,
)
from .sdk import RoomClient, SDKError
from .services.llm import LLMConfig, LLMService, LLMTool
from .services.stt import (
    DeepgramSttProvider,
    OpenAISttProvider,
    SarvamSttProvider,
    SttEvent,
    SttProvider,
    SttService,
)
from .services.tts import TTSConfig, TTSService
from .transport import (
    ChannelTransport,
    RoomTransport,
    SpeakingGate,
    TransportParams,
)
from .vad import (
    EnergyVad,
    SileroVad,
    VadParams,
    VadProcessor,
    VadState,
    WebRtcVad,
)

__version__ = "0.2.0"

__all__ = [
    "Agent",
    "AgentConfig",
    "AssistantAggregator",
    "AudioChunk",
    "BillingCollector",
    "BillingEvent",
    "ChannelTransport",
    "DeepgramSttProvider",
    "EnergyVad",
    "FinishReason",
    "Frame",
    "FrameKind",
    "FrameProcessor",
    "InterruptionHandler",
    "LLMConfig",
    "LLMContext",
    "LLMService",
    "LLMTool",
    "LogBillingStorage",
    "MemoryBillingStorage",
    "Message",
    "OpenAISttProvider",
    "Participant",
    "ParticipantRole",
    "Pipeline",
    "PipelineParams",
    "PipelineTask",
    "Room",
    "RoomClient",
    "RoomControlModule",
    "RoomEventType",
    "RoomManager",
    "RoomTransport",
    "SDKError",
    "SarvamSttProvider",
    "SessionBilling",
    "SileroVad",
    "SpeakingGate",
    "SttEvent",
    "SttProvider",
    "SttService",
    "TTSConfig",
    "TTSService",
    "ToolDefinition",
    "ToolSchema",
    "TranscriptRole",
    "Transcription",
    "TransportParams",
    "UserAggregator",
    "VadParams",
    "VadProcessor",
    "VadState",
    "VadStarted",
    "VadStopped",
    "WebRtcVad",
    "__version__",
    "shared_context",
    "tool",
]
