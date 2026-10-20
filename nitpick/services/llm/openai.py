"""The LLM service.

Talks to any OpenAI-compatible chat-completions endpoint, streaming the
response so the first words can reach TTS while the model is still writing.

Tool calling
------------
When the model requests a tool, the service executes it and re-invokes the
model with the result. The loop is bounded by ``max_tool_rounds`` so a model
that keeps calling tools cannot spin forever.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from ...context import LLMContext, ToolSchema
from ...frames import (
    DOWNSTREAM,
    Frame,
    FrameHandler,
    FrameKind,
    FrameProcessor,
    frame_function_call_end,
    frame_function_call_in_progress,
    frame_function_call_result,
    frame_function_call_start,
    frame_llm_full_response_end,
    frame_llm_full_response_start,
    frame_llm_text,
)

log = logging.getLogger(__name__)

ToolHandler = Callable[[str, str], Awaitable[str] | str]


@dataclass
class LLMTool:
    """A tool the model may call."""

    schema: ToolSchema
    handler: ToolHandler

    @property
    def name(self) -> str:
        return self.schema.name


@dataclass
class LLMConfig:
    """Configuration for :class:`LLMService`."""

    api_key: str | None = None
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.7
    max_tokens: int | None = None
    max_tool_rounds: int = 5
    timeout: float = 60.0

    def resolved_key(self) -> str:
        key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        return key


class LLMService(FrameProcessor):
    """Runs the LLM turn inside a pipeline.

    The service is triggered by the user aggregator pushing a context frame
    downstream. It streams the response as ``LLM_TEXT`` frames and finishes
    with an ``LLM_FULL_RESPONSE_END`` frame.
    """

    def __init__(
        self,
        context: LLMContext,
        config: LLMConfig | None = None,
        *,
        name: str = "LLMService",
    ) -> None:
        super().__init__(name, _LLMHandler(self))
        self.context = context
        self.config = config or LLMConfig()
        self._tools: dict[str, LLMTool] = {}

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def add_tool(self, tool: LLMTool) -> None:
        self._tools[tool.name] = tool

    def add_function(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: ToolHandler,
    ) -> None:
        """Register a function the model may call."""
        schema = ToolSchema(name=name, description=description, parameters=parameters)
        self.add_tool(LLMTool(schema=schema, handler=handler))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    async def _run_turn(self) -> None:
        await self.push_frame(frame_llm_full_response_start(), DOWNSTREAM)

        for round_index in range(self.config.max_tool_rounds + 1):
            tool_calls = await self._stream_once()

            if not tool_calls:
                break

            if round_index >= self.config.max_tool_rounds:
                log.warning(
                    "LLM hit the tool-round limit (%s) — stopping", self.config.max_tool_rounds
                )
                break

            await self.push_frame(frame_function_call_start(), DOWNSTREAM)
            for call in tool_calls:
                await self._execute_tool(call)
            await self.push_frame(frame_function_call_end(), DOWNSTREAM)

        await self.push_frame(frame_llm_full_response_end(), DOWNSTREAM)

    async def _stream_once(self) -> list[dict[str, Any]]:
        """One model invocation. Returns any tool calls it requested."""
        messages = self.context.snapshot()
        tools = [tool.schema.to_dict() for tool in self._tools.values()]

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": True,
            "temperature": self.config.temperature,
        }
        if tools:
            payload["tools"] = tools
        if self.config.max_tokens:
            payload["max_tokens"] = self.config.max_tokens

        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.resolved_key()}",
            "Content-Type": "application/json",
        }

        tool_calls: list[dict[str, Any]] = []
        content_parts: list[str] = []

        import httpx

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if body == "[DONE]":
                        break
                    try:
                        chunk = json.loads(body)
                    except json.JSONDecodeError:
                        continue

                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta") or {}
                        text = delta.get("content")
                        if text:
                            content_parts.append(text)
                            await self.push_frame(frame_llm_text(text), DOWNSTREAM)
                        for call in delta.get("tool_calls") or []:
                            tool_calls.append(self._merge_tool_call(tool_calls, call))

        if content_parts and tool_calls:
            # A tool round needs the assistant's own message in the context so
            # the provider can continue the conversation. A plain-text round
            # is recorded by the assistant aggregator instead — two writers
            # for the same message would duplicate it.
            await self.context.add_assistant_message("".join(content_parts))

        return tool_calls

    @staticmethod
    def _merge_tool_call(
        accumulated: list[dict[str, Any]], delta: dict[str, Any]
    ) -> dict[str, Any]:
        """Accumulate a streamed tool call, which arrives in pieces."""
        index = delta.get("index", 0)
        for existing in accumulated:
            if existing.get("index") == index:
                function = existing.setdefault("function", {})
                if delta.get("id"):
                    existing["id"] = delta["id"]
                if delta.get("type"):
                    existing["type"] = delta["type"]
                if delta.get("function", {}).get("name"):
                    function["name"] = delta["function"]["name"]
                if delta.get("function", {}).get("arguments"):
                    function["arguments"] = (
                        function.get("arguments", "") + delta["function"]["arguments"]
                    )
                return existing

        function = delta.get("function") or {}
        entry = {
            "index": index,
            "id": delta.get("id", ""),
            "type": delta.get("type", "function"),
            "function": {
                "name": function.get("name", ""),
                "arguments": function.get("arguments", ""),
            },
        }
        accumulated.append(entry)
        return entry

    async def _execute_tool(self, call: dict[str, Any]) -> None:
        function = call.get("function") or {}
        name = function.get("name", "")
        arguments = function.get("arguments", "")
        call_id = call.get("id", "")

        await self.push_frame(
            frame_function_call_in_progress(
                _call_data(call_id, name, arguments)
            ),
            DOWNSTREAM,
        )

        tool = self._tools.get(name)
        if tool is None:
            result = f"Unknown tool: {name}"
        else:
            try:
                outcome = tool.handler(arguments)
                result = await outcome if asyncio.iscoroutine(outcome) else outcome
            except Exception as exc:  # noqa: BLE001 - a failing tool must not kill the turn
                log.exception("Tool %s failed", name)
                result = f"Tool error: {exc}"

        await self.context.add_tool_message(result, tool_call_id=call_id, name=name)
        await self.push_frame(
            frame_function_call_result(
                _result_data(call_id, name, result)
            ),
            DOWNSTREAM,
        )


def _call_data(call_id: str, name: str, arguments: str) -> Any:
    from ...frames import FunctionCall

    return FunctionCall(id=call_id, function_name=name, arguments=arguments)


def _result_data(call_id: str, name: str, result: str) -> Any:
    from ...frames import FunctionCallResult

    return FunctionCallResult(id=call_id, function_name=name, result=result)


class _LLMHandler:
    """Triggers the LLM when a context frame arrives."""

    def __init__(self, service: LLMService) -> None:
        self._service = service

    async def on_process_frame(
        self, processor: FrameProcessor, frame: Frame, direction: str
    ) -> None:
        service = self._service
        if frame.kind is FrameKind.LLM_CONTEXT:
            await service._run_turn()
            return
        await processor.push_frame(frame, direction)
