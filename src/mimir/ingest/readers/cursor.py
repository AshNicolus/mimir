"""Reads Cursor's chat/composer history into normalized turns.

Unlike Claude Code, Cursor doesn't keep one file per session: every chat
("composer") lives as rows in a SQLite key-value store, `cursorDiskKV` inside
`globalStorage/state.vscdb`. One `composerData:<id>` row holds a session's
metadata plus the ordered list of message ids (`fullConversationHeadersOnly`);
one `bubbleId:<id>:<bubbleId>` row per message holds its role (`type` 1 user,
2 assistant), text, and any tool call (`toolFormerData`).

Only some assistant bubbles carry a real timestamp (`timingInfo.clientRpcSendTime`);
most don't. A turn without one inherits the previous turn's timestamp plus a
one-second nudge, so ordering is preserved and time-gap segmentation still
works without inventing large fake gaps between untimed turns.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..models import ToolCall, Turn

PREVIEW_CHARS = 200
ROLE_BY_TYPE = {1: "user", 2: "assistant"}


def default_db_path() -> Path:
    home = Path.home()
    if sys.platform == "darwin":
        base = home / "Library" / "Application Support"
    elif sys.platform == "win32":
        base = home / "AppData" / "Roaming"
    else:
        base = home / ".config"
    return base / "Cursor" / "User" / "globalStorage" / "state.vscdb"


def preview_of(value: object) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = text.strip()
    return text[:PREVIEW_CHARS] + ("..." if len(text) > PREVIEW_CHARS else "")


def epoch_ms_to_datetime(value: object) -> datetime | None:
    if not isinstance(value, (int, float)):
        return None
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc)


def load_json(cursor: sqlite3.Cursor, key: str) -> dict | None:
    cursor.execute("SELECT value FROM cursorDiskKV WHERE key = ?", (key,))
    row = cursor.fetchone()
    if row is None or row[0] is None:
        return None
    try:
        return json.loads(row[0])
    except json.JSONDecodeError:
        return None


def bubble_to_turn(bubble: dict, fallback_timestamp: datetime) -> Turn | None:
    role = ROLE_BY_TYPE.get(bubble.get("type"))
    if role is None:
        return None

    text = bubble.get("text") or ""
    tool_calls: list[ToolCall] = []
    tool = bubble.get("toolFormerData")
    if isinstance(tool, dict) and tool.get("name"):
        preview_source = tool.get("rawArgs") or tool.get("result")
        tool_calls.append(ToolCall(name=tool["name"], preview=preview_of(preview_source)))
    if not text and not tool_calls:
        return None

    timing = bubble.get("timingInfo") or {}
    timestamp = epoch_ms_to_datetime(timing.get("clientRpcSendTime")) or fallback_timestamp
    return Turn(role=role, text=text, timestamp=timestamp, tool_calls=tool_calls)


def read_composer(cursor: sqlite3.Cursor, composer_id: str, created_at: datetime) -> list[Turn]:
    composer = load_json(cursor, f"composerData:{composer_id}")
    if composer is None:
        return []

    turns: list[Turn] = []
    clock = created_at
    for header in composer.get("fullConversationHeadersOnly", []):
        bubble_id = header.get("bubbleId")
        if not bubble_id:
            continue
        bubble = load_json(cursor, f"bubbleId:{composer_id}:{bubble_id}")
        if bubble is None:
            continue
        turn = bubble_to_turn(bubble, clock + timedelta(seconds=1))
        if turn is None:
            continue
        clock = turn.timestamp
        turns.append(turn)
    return turns


def read_all_sessions(db_path: Path | None = None) -> dict[str, list[Turn]]:
    """composer_id -> turns, for every composer found in the given (or
    default) Cursor state.vscdb. Returns {} if the file doesn't exist or
    can't be opened, e.g. Cursor is running and holding an exclusive lock.
    """
    path = db_path or default_db_path()
    if not path.exists():
        return {}

    try:
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        connection.execute("PRAGMA busy_timeout = 5000")
    except sqlite3.OperationalError:
        return {}

    try:
        cursor = connection.cursor()
        cursor.execute("SELECT key, value FROM cursorDiskKV WHERE key LIKE 'composerData:%'")
        rows = cursor.fetchall()

        sessions: dict[str, list[Turn]] = {}
        for key, value in rows:
            if value is None:
                continue
            try:
                composer = json.loads(value)
            except json.JSONDecodeError:
                continue
            composer_id = key.split(":", 1)[1]
            created_at = (
                epoch_ms_to_datetime(composer.get("createdAt")) or datetime.now(timezone.utc)
            )
            turns = read_composer(cursor, composer_id, created_at)
            if turns:
                sessions[composer_id] = turns
        return sessions
    except sqlite3.OperationalError:
        return {}
    finally:
        connection.close()
