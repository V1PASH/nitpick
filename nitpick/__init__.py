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
from .processors.user_aggregator import UserAggregator
from .rooms import (
    Participant,
    ParticipantRole,
    Room,
    RoomControlModule,
    RoomEventType,
    RoomManager,
)
from .services.llm import LLMConfig, LLMService, LLMTool
from .services.stt import (
    DeepgramSttProvider,
    OpenAISttProvider,
    SttEvent,
    SttProvider,
    SttService,
)
from .services.tts import TTSConfig, TTSService
from .transport import ChannelTransport, RoomTransport, TransportParams

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentConfig",
    "AssistantAggregator",
    "AudioChunk",
    "ChannelTransport",
    "DeepgramSttProvider",
    "FinishReason",
    "Frame",
    "FrameKind",
    "FrameProcessor",
    "LLMConfig",
    "LLMContext",
    "LLMService",
    "LLMTool",
    "Message",
    "OpenAISttProvider",
    "Participant",
    "ParticipantRole",
    "Pipeline",
    "PipelineParams",
    "PipelineTask",
    "Room",
    "RoomControlModule",
    "RoomEventType",
    "RoomManager",
    "RoomTransport",
    "SttEvent",
    "SttProvider",
    "SttService",
    "TTSConfig",
    "TTSService",
    "ToolDefinition",
    "ToolSchema",
    "Transcription",
    "TransportParams",
    "UserAggregator",
    "VadStarted",
    "VadStopped",
    "__version__",
    "shared_context",
    "tool",
]
