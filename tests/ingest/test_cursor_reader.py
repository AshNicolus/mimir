"""The Cursor reader: real schema shape (composerData/bubbleId rows in a
SQLite cursorDiskKV table), built fresh per test, never against a real
Cursor install.
"""

import json
import sqlite3

import pytest

from mimir.ingest.readers.cursor import (
    bubble_to_turn,
    default_db_path,
    epoch_ms_to_datetime,
    read_all_sessions,
)

CREATED_AT = 1748100000000  # 2025-05-24T14:40:00Z, arbitrary real-looking epoch ms


def make_db(tmp_path, composers):
    """composers: {composer_id: [(bubble_id, bubble_dict), ...]}"""
    path = tmp_path / "state.vscdb"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE cursorDiskKV (key TEXT PRIMARY KEY, value BLOB)")
    for composer_id, bubbles in composers.items():
        headers = [
            {"bubbleId": bubble_id, "type": bubble["type"] if bubble else 1}
            for bubble_id, bubble in bubbles
        ]
        composer_data = {
            "composerId": composer_id,
            "createdAt": CREATED_AT,
            "fullConversationHeadersOnly": headers,
        }
        connection.execute(
            "INSERT INTO cursorDiskKV VALUES (?, ?)",
            (f"composerData:{composer_id}", json.dumps(composer_data)),
        )
        for bubble_id, bubble in bubbles:
            if bubble is None:
                continue  # simulate a tombstoned/missing row: header exists, bubble doesn't
            connection.execute(
                "INSERT INTO cursorDiskKV VALUES (?, ?)",
                (f"bubbleId:{composer_id}:{bubble_id}", json.dumps(bubble)),
            )
    connection.commit()
    connection.close()
    return path


def test_reads_composer_messages_in_order(tmp_path):
    db = make_db(
        tmp_path,
        {
            "c1": [
                ("b1", {"type": 1, "text": "add a login page"}),
                ("b2", {"type": 2, "text": "I'll add it now."}),
            ]
        },
    )
    sessions = read_all_sessions(db)
    assert list(sessions) == ["c1"]
    assert [t.role for t in sessions["c1"]] == ["user", "assistant"]
    assert [t.text for t in sessions["c1"]] == ["add a login page", "I'll add it now."]


def test_tool_call_is_captured_as_a_preview():
    bubble = {
        "type": 2,
        "text": "",
        "toolFormerData": {"name": "read_file", "rawArgs": "x" * 300, "result": "..."},
    }
    turn = bubble_to_turn(bubble, fallback_timestamp=epoch_ms_to_datetime(CREATED_AT))
    assert turn.tool_calls[0].name == "read_file"
    assert len(turn.tool_calls[0].preview) <= 203  # 200 chars + "..."


def test_unknown_bubble_type_is_dropped():
    bubble = {"type": 99, "text": "some future message kind"}
    assert bubble_to_turn(bubble, fallback_timestamp=epoch_ms_to_datetime(CREATED_AT)) is None


def test_blank_bubble_with_no_tool_call_is_dropped():
    bubble = {"type": 1, "text": ""}
    assert bubble_to_turn(bubble, fallback_timestamp=epoch_ms_to_datetime(CREATED_AT)) is None


def test_missing_bubble_row_is_skipped_not_fatal(tmp_path):
    db = make_db(
        tmp_path,
        {
            "c1": [
                ("b1", {"type": 1, "text": "first"}),
                ("b2", None),  # header references a bubble that was never written
                ("b3", {"type": 2, "text": "third"}),
            ]
        },
    )
    sessions = read_all_sessions(db)
    assert [t.text for t in sessions["c1"]] == ["first", "third"]


def test_turns_without_timing_info_get_strictly_increasing_timestamps(tmp_path):
    db = make_db(
        tmp_path,
        {
            "c1": [
                ("b1", {"type": 1, "text": "first"}),
                ("b2", {"type": 2, "text": "second"}),
                ("b3", {"type": 1, "text": "third"}),
            ]
        },
    )
    turns = read_all_sessions(db)["c1"]
    timestamps = [t.timestamp for t in turns]
    assert timestamps == sorted(timestamps)
    assert len(set(timestamps)) == len(timestamps)


def test_real_timing_info_is_used_when_present():
    real_ms = CREATED_AT + 60_000
    bubble = {"type": 2, "text": "reply", "timingInfo": {"clientRpcSendTime": real_ms}}
    turn = bubble_to_turn(bubble, fallback_timestamp=epoch_ms_to_datetime(CREATED_AT))
    assert turn.timestamp == epoch_ms_to_datetime(real_ms)


def test_multiple_composers_become_separate_sessions(tmp_path):
    db = make_db(
        tmp_path,
        {
            "c1": [("b1", {"type": 1, "text": "task one"})],
            "c2": [("b1", {"type": 1, "text": "task two"})],
        },
    )
    sessions = read_all_sessions(db)
    assert set(sessions) == {"c1", "c2"}


def test_missing_db_file_returns_empty(tmp_path):
    assert read_all_sessions(tmp_path / "does-not-exist.vscdb") == {}


def test_composer_with_no_surviving_turns_is_omitted(tmp_path):
    db = make_db(tmp_path, {"c1": [("b1", None)]})
    assert read_all_sessions(db) == {}


@pytest.mark.parametrize(
    "platform, expected_parts",
    [
        ("darwin", ("Library", "Application Support", "Cursor")),
        ("win32", ("AppData", "Roaming", "Cursor")),
        ("linux", (".config", "Cursor")),
    ],
)
def test_default_db_path_is_platform_specific(monkeypatch, platform, expected_parts):
    monkeypatch.setattr("mimir.ingest.readers.cursor.sys.platform", platform)
    path = default_db_path()
    assert path.name == "state.vscdb"
    for part in expected_parts:
        assert part in path.parts
