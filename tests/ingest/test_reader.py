"""The Claude Code reader: real schema shape, in a fixture, not personal data."""

from pathlib import Path

from mimir.ingest.readers.claude_code import parse_entry, parse_timestamp, read_session

FIXTURE = Path(__file__).parent / "fixtures" / "sample_session.jsonl"


def test_reads_the_main_line_only():
    turns = read_session(FIXTURE)
    assert [t.role for t in turns] == ["user", "assistant", "user", "user"]


def test_sidechain_is_dropped():
    turns = read_session(FIXTURE)
    texts = " ".join(t.text for t in turns)
    assert "sub-agent branch" not in texts


def test_malformed_line_is_skipped_not_fatal():
    turns = read_session(FIXTURE)
    assert len(turns) == 4  # the invalid-JSON line contributed nothing, and nothing raised


def test_tool_use_is_captured_as_a_preview_not_inlined():
    turns = read_session(FIXTURE)
    assistant_turn = turns[1]
    assert assistant_turn.tool_calls[0].name == "Edit"
    assert "app.py" in assistant_turn.tool_calls[0].preview


def test_tool_result_is_captured_on_the_next_turn():
    turns = read_session(FIXTURE)
    result_turn = turns[2]
    assert result_turn.tool_calls[0].name == "tool_result"
    assert "edited successfully" in result_turn.tool_calls[0].preview


def test_queue_operation_lines_are_ignored():
    turns = read_session(FIXTURE)
    assert all(t.text != "" or t.tool_calls for t in turns)


def test_parse_entry_rejects_non_user_assistant_roles():
    assert parse_entry({"message": {"role": "system", "content": "x"}}) is None


def test_parse_timestamp_handles_zulu_suffix():
    ts = parse_timestamp("2026-01-01T10:00:00.000Z")
    assert ts is not None and ts.year == 2026


def test_parse_timestamp_rejects_garbage():
    assert parse_timestamp("not a date") is None
    assert parse_timestamp(None) is None
