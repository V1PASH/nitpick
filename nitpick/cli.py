"""The nitpick command-line interface.

::

    nitpick serve --room demo --port 8000     # run a demo agent
    nitpick version
"""
from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nitpick",
        description="Build and run voice agents with nitpick.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Run a demo voice agent.")
    serve.add_argument("--room", default="demo", help="Room id to join.")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--name", default="Assistant", help="Agent display name.")
    serve.add_argument(
        "--stt", default="openai", help="STT provider: openai | deepgram"
    )
    serve.add_argument("--llm", default="openai:gpt-4o-mini")
    serve.add_argument("--tts", default="openai:alloy")
    serve.add_argument(
        "--prompt",
        default="You are a helpful voice assistant. Keep answers to one or two sentences.",
    )

    version = subparsers.add_parser("version", help="Print the nitpick version.")
    version.set_defaults(handler=_version)

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    if args.command == "serve":
        return _serve(args)
    if args.command == "version":
        return _version(args)
    return 1


def _serve(args: argparse.Namespace) -> int:
    from .agent import Agent

    agent = Agent(
        name=args.name,
        prompt=args.prompt,
        stt=args.stt,
        llm=args.llm,
        tts=args.tts,
    )
    try:
        agent.run(room_id=args.room, host=args.host, port=args.port)
    except KeyboardInterrupt:
        return 0
    return 0


def _version(_args: argparse.Namespace) -> int:
    from . import __version__

    print(f"nitpick {__version__}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
