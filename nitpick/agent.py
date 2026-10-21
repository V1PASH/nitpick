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
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .billing import SessionBilling
from .context import LLMContext, ToolSchema
from .frames import FrameProcessor
from .pipeline import PipelineParams, PipelineTask
from .processors.assistant_aggregator import AssistantAggregator
from .processors.interruption import InterruptionHandler
from .processors.user_aggregator import UserAggregator
from .rooms import RoomControlModule, RoomManager
from .rooms.model import Participant
from .services.llm import LLMConfig, LLMService, LLMTool
from .services.stt import (
    DeepgramSttProvider,
    OpenAISttProvider,
    SarvamSttProvider,
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
    #: The base URL of the LLM endpoint, for OpenAI-compatible servers
    #: (a self-hosted model, or a gateway).
    llm_base_url: str | None = None
    language: str | None = None
    input_sample_rate: int = 16_000
    output_sample_rate: int = 24_000
    #: Server-side VAD backend: ``"energy"`` (default, no dependencies),
    #: ``"webrtc"``, ``"silero"``, or ``None`` to rely on the browser's own
    #: speaking detection.
    vad: str | None = "energy"
    #: Whether the user can interrupt the bot mid-sentence.
    allow_interruptions: bool = True
    #: Whether to collect usage and transcript events for this session.
    billing: bool = False


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
        llm_base_url: str | None = None,
        language: str | None = None,
        tools: list[ToolDefinition] | None = None,
        input_sample_rate: int = 16_000,
        output_sample_rate: int = 24_000,
        vad: str | None = "energy",
        allow_interruptions: bool = True,
        billing: bool = False,
        session_id: str | None = None,
        stt_provider: SttProvider | None = None,
        llm_service: LLMService | None = None,
        tts_service: TTSService | None = None,
        vad_backend: object | None = None,
        manager: RoomManager | None = None,
    ) -> None:
        self.config = AgentConfig(
            name=name,
            prompt=prompt or AgentConfig.prompt,
            stt=stt,
            llm=llm,
            tts=tts,
            llm_base_url=llm_base_url,
            language=language,
            input_sample_rate=input_sample_rate,
            output_sample_rate=output_sample_rate,
            vad=vad,
            allow_interruptions=allow_interruptions,
            billing=billing,
        )
        self.tools = list(tools or [])
        self.context = LLMContext(self.config.prompt)
        # A shared manager lets one server host many agents, each in its
        # own room. When None the agent owns its manager and serves it.
        self.manager = manager if manager is not None else RoomManager()
        self.control = RoomControlModule(self.manager)
        self.session_id = session_id or f"s-{uuid.uuid4().hex[:12]}"

        # Service overrides — used by tests and by anyone who wants to plug
        # in a provider nitpick does not ship.
        self._stt_provider = stt_provider
        self._llm_service = llm_service
        self._tts_service = tts_service
        self._vad_backend = vad_backend

        # Billing.
        self.billing: SessionBilling | None = None
        if billing:
            self.billing = SessionBilling(self.session_id)
            self.billing.start()

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
        elif provider_name == "sarvam":
            provider = SarvamSttProvider(
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
                f"Unknown STT provider {provider_name!r}. "
                "Use 'openai', 'deepgram' or 'sarvam'."
            )
        return SttService(provider)

    def _build_vad(self, gate: object | None, on_speaking: Any = None) -> FrameProcessor | None:
        """Build the server-side VAD processor, if enabled.

        If the backend is not installed the agent still works — the browser
        reports its own speaking edges, which is enough for turn detection.
        Server-side VAD is an improvement, not a requirement.
        """
        if self._vad_backend is not None:
            backend = self._vad_backend
        elif self.config.vad == "energy":
            from .vad import EnergyVad

            backend = EnergyVad(self.config.input_sample_rate)
        elif self.config.vad == "webrtc":
            from .vad import WebRtcVad

            try:
                backend = WebRtcVad(self.config.input_sample_rate)
            except (RuntimeError, ValueError) as exc:
                log.warning(
                    "Server-side VAD unavailable (%s) — falling back to the "
                    "browser's speaking detection",
                    exc,
                )
                return None
        elif self.config.vad == "silero":
            from .vad import SileroVad

            try:
                backend = SileroVad(self.config.input_sample_rate)
            except (RuntimeError, ValueError, FileNotFoundError) as exc:
                log.warning(
                    "Server-side VAD unavailable (%s) — falling back to the "
                    "browser's speaking detection",
                    exc,
                )
                return None
        elif self.config.vad is None:
            return None
        else:
            raise ValueError(
                f"Unknown VAD backend {self.config.vad!r}. Use 'webrtc', 'silero' or None."
            )

        from .vad import VadProcessor

        return VadProcessor(backend, speaking_gate=gate, on_speaking=on_speaking)

    def _build_llm(self) -> LLMService:
        if self._llm_service is not None:
            return self._llm_service

        spec = self.config.llm
        provider_name, _, model = spec.partition(":")
        if provider_name != "openai":
            raise ValueError(f"Unknown LLM provider {provider_name!r}. Use 'openai'.")
        config = LLMConfig(model=model or "gpt-4o-mini")
        if self.config.llm_base_url:
            config.base_url = self.config.llm_base_url
        service = LLMService(self.context, config, billing=self.billing)
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
        return TTSService(config, billing=self.billing)

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
        vad = self._build_vad(transport.gate, transport.publish_user_speaking)

        processors: list[FrameProcessor] = [transport.input()]
        if vad is not None:
            processors.append(vad)
        processors.extend(
            [
                stt,
                user_agg,
                llm,
                assistant_agg,
                tts,
                # The interruption handler sits after the TTS so it sees both
                # the bot-speaking signals and the user's turn starts.
                InterruptionHandler(enabled=self.config.allow_interruptions),
                transport.output(),
            ]
        )

        task = PipelineTask(
            processors,
            PipelineParams(
                allow_interruptions=self.config.allow_interruptions,
                metadata={"agent": self.config.name, "session_id": self.session_id},
            ),
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

        server = SignalingServer(
            self.manager,
            control=self.control,
            audio_output_sample_rate=self.config.output_sample_rate,
        )
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
