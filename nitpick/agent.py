"""The :class:`Agent` facade — the API a nitpick user actually touches.

The goal is that building a voice agent is one expression::

    from nitpick import Agent, tool

    agent = Agent(
        name="assistant",
        prompt="You are a helpful voice assistant. Keep replies short.",
        stt="openai",
        llm="openai:gpt-4o-mini",
        tts="openai:alloy",
        tools=[weather],
    )

    agent.run(room_id="demo", port=8000)

That serves the browser client, waits for someone to join the room, and runs
the voice loop end to end.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .context import LLMContext, ToolSchema
from .frames import FrameProcessor
from .pipeline import PipelineParams, PipelineTask
from .processors.assistant_aggregator import AssistantAggregator
from .processors.user_aggregator import UserAggregator
from .rooms import RoomControlModule, RoomManager
from .rooms.model import Participant
from .services.llm import LLMConfig, LLMService, LLMTool
from .services.stt import (
    DeepgramSttProvider,
    OpenAISttProvider,
    SttProvider,
    SttService,
)
from .services.tts import TTSConfig, TTSService
from .transport import RoomTransport, TransportParams

log = logging.getLogger(__name__)

ToolFunction = Callable[..., Awaitable[str] | str]


@dataclass
class ToolDefinition:
    """A callable exposed to the model."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolFunction

    def to_tool(self) -> LLMTool:
        schema = ToolSchema(
            name=self.name, description=self.description, parameters=self.parameters
        )
        return LLMTool(schema=schema, handler=self._invoke)

    async def _invoke(self, arguments: str) -> str:
        import json

        try:
            parsed = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            parsed = {}
        result = self.handler(**parsed) if isinstance(parsed, dict) else self.handler(parsed)
        if asyncio.iscoroutine(result):
            result = await result
        return str(result)


def tool(
    name: str,
    description: str,
    parameters: dict[str, Any] | None = None,
) -> Callable[[ToolFunction], ToolDefinition]:
    """Decorator that registers a function as a model-callable tool.

    Example::

        @tool("get_weather", "Look up the weather for a city.")
        def get_weather(city: str) -> str:
            return f"It is 28 degrees in {city}."
    """

    def decorator(func: ToolFunction) -> ToolDefinition:
        return ToolDefinition(
            name=name,
            description=description,
            parameters=parameters
            or {
                "type": "object",
                "properties": {
                    key: {"type": "string"} for key in func.__code__.co_varnames[
                        : func.__code__.co_argcount
                    ]
                },
                "required": list(func.__code__.co_varnames[: func.__code__.co_argcount]),
            },
            handler=func,
        )

    return decorator


@dataclass
class AgentConfig:
    """Everything needed to build an agent."""

    name: str = "Agent"
    prompt: str = "You are a helpful voice assistant. Keep replies to one or two sentences."
    stt: str = "openai"
    llm: str = "openai:gpt-4o-mini"
    tts: str = "openai:alloy"
    language: str | None = None
    input_sample_rate: int = 16_000
    output_sample_rate: int = 24_000


class Agent:
    """A voice agent you can run locally.

    Parameters
    ----------
    name:
        The agent's display name in the room.
    prompt:
        The system prompt.
    stt:
        ``"openai"``, ``"deepgram"``, or ``"sarvam"``.
    llm:
        ``"openai"`` or ``"openai:<model>"``.
    tts:
        ``"openai"`` or ``"openai:<voice>"``.
    tools:
        Tools the model may call.
    """

    def __init__(
        self,
        name: str = "Agent",
        *,
        prompt: str | None = None,
        stt: str = "openai",
        llm: str = "openai:gpt-4o-mini",
        tts: str = "openai:alloy",
        language: str | None = None,
        tools: list[ToolDefinition] | None = None,
        input_sample_rate: int = 16_000,
        output_sample_rate: int = 24_000,
        stt_provider: SttProvider | None = None,
        llm_service: LLMService | None = None,
        tts_service: TTSService | None = None,
    ) -> None:
        self.config = AgentConfig(
            name=name,
            prompt=prompt or AgentConfig.prompt,
            stt=stt,
            llm=llm,
            tts=tts,
            language=language,
            input_sample_rate=input_sample_rate,
            output_sample_rate=output_sample_rate,
        )
        self.tools = list(tools or [])
        self.context = LLMContext(self.config.prompt)
        self.manager = RoomManager()
        self.control = RoomControlModule(self.manager)
        # Service overrides — used by tests and by anyone who wants to plug
        # in a provider nitpick does not ship.
        self._stt_provider = stt_provider
        self._llm_service = llm_service
        self._tts_service = tts_service
        self._task: PipelineTask | None = None
        self._transport: RoomTransport | None = None
        self._participant: Participant | None = None
        self._server: Any = None
        self._server_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Service construction
    # ------------------------------------------------------------------

    def _build_stt(self) -> SttService:
        if self._stt_provider is not None:
            return SttService(self._stt_provider)

        spec = self.config.stt
        provider_name, _, options = spec.partition(":")
        options = dict(
            pair.split("=", 1) for pair in options.split(",") if "=" in pair
        )

        provider: SttProvider
        if provider_name == "deepgram":
            provider = DeepgramSttProvider(
                language=self.config.language,
                sample_rate=self.config.input_sample_rate,
                **options,
            )
        elif provider_name == "openai":
            provider = OpenAISttProvider(
                language=self.config.language,
                sample_rate=self.config.input_sample_rate,
                **options,
            )
        else:
            raise ValueError(
                f"Unknown STT provider {provider_name!r}. Use 'openai' or 'deepgram'."
            )
        return SttService(provider)

    def _build_llm(self) -> LLMService:
        if self._llm_service is not None:
            return self._llm_service

        spec = self.config.llm
        provider_name, _, model = spec.partition(":")
        if provider_name != "openai":
            raise ValueError(f"Unknown LLM provider {provider_name!r}. Use 'openai'.")
        config = LLMConfig(model=model or "gpt-4o-mini")
        service = LLMService(self.context, config)
        for definition in self.tools:
            service.add_tool(definition.to_tool())
        return service

    def _build_tts(self) -> TTSService:
        if self._tts_service is not None:
            return self._tts_service

        spec = self.config.tts
        provider_name, _, voice = spec.partition(":")
        if provider_name != "openai":
            raise ValueError(f"Unknown TTS provider {provider_name!r}. Use 'openai'.")
        config = TTSConfig(
            voice=voice or "alloy",
            sample_rate=self.config.output_sample_rate,
        )
        return TTSService(config)

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def build_pipeline(self, room_id: str) -> PipelineTask:
        """Assemble the pipeline for a room. Returns the task."""
        participant = self.manager.register_agent_participant(self.config.name)
        self._participant = participant
        self.control.join_participant(
            participant,
            room_name=room_id,
            max_participants=2,
            create_if_missing=True,
        )

        transport = RoomTransport(
            self.manager,
            participant,
            params=TransportParams(
                audio_in_sample_rate=self.config.input_sample_rate,
                audio_out_sample_rate=self.config.output_sample_rate,
            ),
        )
        self._transport = transport

        stt = self._build_stt()
        user_agg = UserAggregator(self.context)
        llm = self._build_llm()
        assistant_agg = AssistantAggregator(self.context)
        tts = self._build_tts()

        processors: list[FrameProcessor] = [
            transport.input(),
            stt,
            user_agg,
            llm,
            assistant_agg,
            tts,
            transport.output(),
        ]

        task = PipelineTask(
            processors,
            PipelineParams(allow_interruptions=True, metadata={"agent": self.config.name}),
        )
        self._task = task
        return task

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------

    async def serve(
        self,
        *,
        room_id: str = "demo",
        host: str = "0.0.0.0",
        port: int = 8000,
        serve: bool = True,
    ) -> None:
        """Run the agent. Blocks until interrupted."""
        from .signaling.server import create_app

        task = self.build_pipeline(room_id)
        task.start()

        app = create_app(self._server_with_manager())

        if not serve:
            await task.wait()
            return

        import uvicorn

        config = uvicorn.Config(app, host=host, port=port, log_level="info")
        self._server = uvicorn.Server(config)
        log.info(
            "nitpick agent %r listening on http://%s:%s (room: %s)",
            self.config.name,
            host,
            port,
            room_id,
        )
        try:
            await self._server.serve()
        finally:
            await task.cancel()
            with contextlib.suppress(Exception):
                await task.wait()

    def _server_with_manager(self) -> Any:
        from .signaling.server import SignalingServer

        server = SignalingServer(self.manager, control=self.control)
        return server

    def run(
        self,
        *,
        room_id: str = "demo",
        host: str = "0.0.0.0",
        port: int = 8000,
        serve: bool = True,
    ) -> None:
        """Blocking convenience wrapper around :meth:`serve`."""
        try:
            asyncio.run(self.serve(room_id=room_id, host=host, port=port, serve=serve))
        except KeyboardInterrupt:
            print("\nbye")
