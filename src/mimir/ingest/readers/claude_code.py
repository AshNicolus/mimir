"""Reads Claude Code's session JSONL files into normalized turns.

Each project gets a directory under ~/.claude/projects/<sanitized-path>/, and
each session is one <uuid>.jsonl file, appended to as the session runs. Entries
form a tree via parentUuid/isSidechain (sub-agent branches fork off); this
reader keeps only the main line, in file order, since a session is appended
chronologically and that already reconstructs the primary conversation.

Tool payloads are previewed, never captured in full: a Bash command's full
stdout or a Read's full file contents would otherwise dominate an episode's
size long before the compressor ever runs.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from ..models import Episode, ToolCall, Turn
from ..segment import segment

DEFAULT_ROOT = Path.home() / ".claude" / "projects"
PREVIEW_CHARS = 200


def find_sessions(root: Path = DEFAULT_ROOT) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.glob("*/*.jsonl"))


def read_session(path: Path) -> list[Turn]:
    turns: list[Turn] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            turn = parse_entry(entry)
            if turn is not None:
                turns.append(turn)
    return turns


def parse_entry(entry: dict) -> Turn | None:
    if entry.get("isSidechain"):
        return None  # a sub-agent branch, not the main conversation
    message = entry.get("message")
    if not isinstance(message, dict):
        return None
    role = message.get("role")
    if role not in ("user", "assistant"):
        return None
    timestamp = parse_timestamp(entry.get("timestamp"))
    if timestamp is None:
        return None

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    content = message.get("content")
    if isinstance(content, str):
        text_parts.append(content)
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(block.get("text", ""))
            elif block_type == "tool_use":
                name = block.get("name", "unknown")
                tool_calls.append(ToolCall(name=name, preview=preview_of(block.get("input"))))
            elif block_type == "tool_result":
                preview = preview_of(block.get("content"))
                tool_calls.append(ToolCall(name="tool_result", preview=preview))

    text = "\n".join(p for p in text_parts if p)
    if not text and not tool_calls:
        return None
    return Turn(role=role, text=text, timestamp=timestamp, tool_calls=tool_calls)


def preview_of(value: object) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = text.strip()
    return text[:PREVIEW_CHARS] + ("..." if len(text) > PREVIEW_CHARS else "")


def parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def read_all_sessions(root: Path = DEFAULT_ROOT) -> dict[str, list[Turn]]:
    """session_id -> turns, for every session file found under root."""
    sessions = {}
    for path in find_sessions(root):
        turns = read_session(path)
        if turns:
            sessions[path.stem] = turns
    return sessions


def read_all_episodes(root: Path = DEFAULT_ROOT) -> list[Episode]:
    """Every session found under root, already split into episodes."""
    episodes = []
    for session_id, turns in read_all_sessions(root).items():
        episodes.extend(segment("claude-code", session_id, turns))
    return episodes
