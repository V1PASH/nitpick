"""The documentation must match the code.

Documentation drifts. These tests parse the docs and check them against the
source, so a message type or error code cannot be added, renamed or removed
without the docs being updated too.
"""
from __future__ import annotations

import re
from pathlib import Path


from nitpick.rooms import RoomEventType

DOCS = Path(__file__).resolve().parent.parent / "docs"
SDK_DOC = DOCS / "sdk.md"

SOURCE = Path(__file__).resolve().parent.parent / "nitpick"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _server_source() -> str:
    return _read(SOURCE / "signaling" / "server.py")


def _documented_error_codes() -> set[str]:
    """The error codes in the sdk.md table."""
    text = _read(SDK_DOC)
    section = text.split("### Error codes", 1)
    assert len(section) == 2, "sdk.md must have an 'Error codes' section"

    codes: set[str] = set()
    for line in section[1].splitlines():
        match = re.match(r"^\|\s*`([a-z_]+)`\s*\|", line)
        if match:
            codes.add(match.group(1))
    return codes


def _implemented_error_codes() -> set[str]:
    """Every error code the server can actually emit.

    Most are raised as ``ProtocolError("code", ...)``, but the transport
    level checks (malformed JSON, a non-object payload) are emitted
    directly by the WebSocket endpoint.
    """
    source = _server_source()
    codes = set(re.findall(r'ProtocolError\(\s*"([a-z_]+)"', source))
    codes |= set(re.findall(r'"code":\s*"([a-z_]+)"', source))
    return codes


def _implemented_message_types() -> set[str]:
    """Every client→server message type the server handles."""
    source = _server_source()
    block = source.split("_HANDLERS = {", 1)[1].split("}", 1)[0]
    return set(re.findall(r'"([a-z_]+)":', block))


def _documented_client_types() -> set[str]:
    """The client→server types listed in the sdk.md headings.

    Only headings are used: the JSON examples in that section contain
    backticked *field* names, which are not message types.
    """
    text = _read(SDK_DOC)
    section = text.split("### Client → server", 1)[1].split("### Server → client", 1)[0]

    types: set[str] = set()
    for line in section.splitlines():
        heading = re.match(r"^####\s+(.*)$", line)
        if heading:
            types |= set(re.findall(r"`([a-z_]+)`", heading.group(1)))
    return types


def test_every_implemented_error_code_is_documented():
    implemented = _implemented_error_codes()
    documented = _documented_error_codes()

    missing = implemented - documented
    assert not missing, f"these error codes are not documented: {sorted(missing)}"


def test_every_documented_error_code_is_implemented():
    implemented = _implemented_error_codes()
    documented = _documented_error_codes()

    extra = documented - implemented
    assert not extra, f"these error codes are documented but never raised: {sorted(extra)}"


def test_every_implemented_message_type_is_documented():
    implemented = _implemented_message_types()
    documented = _documented_client_types()

    missing = implemented - documented
    assert not missing, f"these message types are not documented: {sorted(missing)}"


def test_every_documented_message_type_is_implemented():
    implemented = _implemented_message_types()
    documented = _documented_client_types()

    # The docs also mention server→client types in the same file; those are
    # emitted rather than handled, so only the client→server set is compared.
    extra = documented - implemented
    assert not extra, f"these message types are documented but not handled: {sorted(extra)}"


def test_every_room_event_type_is_documented():
    text = _read(SDK_DOC)
    section = text.split("### Event types", 1)[1].split("### Event shape", 1)[0]

    documented: set[str] = set()
    for line in section.splitlines():
        match = re.match(r"^\|\s*`([a-z_]+)`\s*\|", line)
        if match and match.group(1) != "event_type":
            documented.add(match.group(1))

    implemented = {event.value for event in RoomEventType}
    assert implemented - documented == set(), "some event types are not documented"
    assert documented - implemented == set(), "some event types are documented but do not exist"


def test_the_docs_exist():
    assert (DOCS / "README.md").exists()
    assert (DOCS / "architecture.md").exists()
    assert (DOCS / "sdk.md").exists()


def test_the_readme_links_the_documentation():
    readme = _read(SOURCE.parent / "README.md")
    for name in ("docs/architecture.md", "docs/sdk.md", "docs/README.md"):
        assert name in readme, f"the README must link {name}"


def test_the_documented_room_manager_api_exists():
    """The SDK doc names methods — they must exist."""
    from nitpick.rooms import RoomManager

    for method in (
        "add_event_listener",
        "remove_event_listener",
        "list_events",
        "stats",
        "emit_participant_transcript",
        "update_participant_speaking",
        "broadcast_message_to_room",
    ):
        assert hasattr(RoomManager, method), f"RoomManager.{method} is missing"


def test_the_documented_client_api_exists():
    """The SDK doc names methods — they must exist."""
    from nitpick.sdk import RoomClient

    for method in (
        "connect",
        "close",
        "join_room",
        "create_room",
        "leave_room",
        "publish_pcm",
        "publish_audio",
        "set_speaking",
        "send_transcript",
        "send_offer",
        "send_answer",
        "send_ice_candidate",
        "recv",
        "messages",
        "send",
    ):
        assert hasattr(RoomClient, method), f"RoomClient.{method} is missing"


def test_the_documented_server_app_endpoints_exist():
    """The SDK doc names HTTP endpoints — they must be routed."""
    from nitpick.signaling.server import create_app

    application = create_app()
    paths = {getattr(route, "path", None) for route in application.routes}
    assert "/ws" in paths
    assert "/health" in paths
    assert "/" in paths
