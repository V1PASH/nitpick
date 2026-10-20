"""Shared LLM conversation context.

The context is the single mutable piece of state shared between the
aggregators and the LLM service. It holds the message history, the tool
schemas, and the tool-choice policy.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass
class Message:
    """One entry in the conversation history."""

    role: str
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            payload["name"] = self.name
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            payload["tool_calls"] = self.tool_calls
        return payload


@dataclass
class ToolSchema:
    """A function the model is allowed to call."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
                or {"type": "object", "properties": {}},
            },
        }


class LLMContext:
    """Thread-safe conversation state shared across a pipeline.

    An ``asyncio.Lock`` guards mutations because the user aggregator, the
    assistant aggregator, and the LLM service all touch it concurrently.
    """

    def __init__(self, system_prompt: str | None = None) -> None:
        self._lock = asyncio.Lock()
        self._messages: list[Message] = []
        self._tools: list[ToolSchema] = []
        self._tool_choice: Any = None
        if system_prompt:
            self._messages.append(Message(role="system", content=system_prompt))

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    @property
    def messages(self) -> list[Message]:
        """A snapshot of the message history."""
        return list(self._messages)

    async def add_message(self, message: Message) -> None:
        async with self._lock:
            self._messages.append(message)

    async def add_user_message(self, content: str) -> None:
        await self.add_message(Message(role="user", content=content))

    async def add_assistant_message(self, content: str) -> None:
        await self.add_message(Message(role="assistant", content=content))

    async def add_tool_message(
        self, content: str, *, tool_call_id: str, name: str | None = None
    ) -> None:
        await self.add_message(
            Message(
                role="tool", content=content, tool_call_id=tool_call_id, name=name
            )
        )

    async def clear(self) -> None:
        async with self._lock:
            self._messages.clear()

    def snapshot(self) -> list[dict[str, Any]]:
        """The message history as plain dicts, ready for a provider payload."""
        return [message.to_dict() for message in self._messages]

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    async def add_tool(self, tool: ToolSchema) -> None:
        async with self._lock:
            self._tools.append(tool)

    async def remove_tool(self, name: str) -> None:
        async with self._lock:
            self._tools = [tool for tool in self._tools if tool.name != name]

    @property
    def tools(self) -> list[ToolSchema]:
        return list(self._tools)

    async def set_tool_choice(self, choice: Any) -> None:
        async with self._lock:
            self._tool_choice = choice

    @property
    def tool_choice(self) -> Any:
        return self._tool_choice

    # ------------------------------------------------------------------
    # Trimming
    # ------------------------------------------------------------------

    async def trim_to(self, max_messages: int, *, keep_system: bool = True) -> None:
        """Drop the oldest non-system messages until the history fits.

        A crude but effective guard against unbounded context growth in long
        conversations.
        """
        if max_messages <= 0:
            return
        async with self._lock:
            system = [m for m in self._messages if keep_system and m.role == "system"]
            rest = [m for m in self._messages if not (keep_system and m.role == "system")]
            if len(rest) + len(system) <= max_messages:
                return
            keep = max(0, max_messages - len(system))
            self._messages = system + rest[-keep:] if keep else system


def shared_context(system_prompt: str | None = None) -> LLMContext:
    """Create an :class:`LLMContext`."""
    return LLMContext(system_prompt)
