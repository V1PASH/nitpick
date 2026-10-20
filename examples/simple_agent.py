"""A minimal nitpick agent.

Run it::

    export OPENAI_API_KEY=sk-...
    python examples/simple_agent.py

Then open http://localhost:8000 and talk to it.
"""
from __future__ import annotations

import asyncio
import os

from nitpick import Agent, tool


@tool("get_time", "Get the current time in the user's timezone.")
def get_time() -> str:
    """Return the current time."""
    from datetime import datetime

    return datetime.now().strftime("%H:%M:%S")


@tool("get_weather", "Look up the current weather for a city.")
def get_weather(city: str) -> str:
    """Return a canned weather report."""
    return f"It is 28°C and partly cloudy in {city}."


async def main() -> None:
    agent = Agent(
        name="Nitpick",
        prompt=(
            "You are Nitpick, a friendly voice assistant. "
            "Keep every answer to one or two short sentences."
        ),
        stt="openai",
        llm="openai:gpt-4o-mini",
        tts="openai:alloy",
        tools=[get_time, get_weather],
    )

    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY is not set — the agent will fail on the first turn.")

    await agent.serve(room_id="demo", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nbye")
