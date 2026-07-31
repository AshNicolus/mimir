"""The MCP server: tool bodies, then the registration layer.

The MimirTools tests need no MCP SDK, which is the point of keeping the logic
separate from the protocol wiring.
"""

import asyncio

import pytest

from mimir import Mimir
from mimir.mcp_server import (
    DEFAULT_DB_PATH,
    MimirTools,
    build_server,
    clamp,
    default_db_path,
    experience_to_dict,
    parse_outcome,
)


@pytest.fixture
def tools(memory):
    return MimirTools(memory)


def test_record_and_recall_round_trip(tools):
    stored = tools.record_experience("fix login latency", "add a redis cache", "success")
    assert stored["outcome"] == "success"
    assert stored["score"] == 1.0

    found = tools.recall_experiences("login latency")
    assert found["count"] == 1
    assert found["experiences"][0]["id"] == stored["id"]


def test_record_failure_keeps_the_reason(tools):
    stored = tools.record_failure("throttle clients", "fixed window limiter", "missed WebSockets")
    assert stored["outcome"] == "failure"
    assert stored["context"]["failure_reason"] == "missed WebSockets"


def test_recall_can_filter_to_failures(tools):
    tools.record_experience("api latency", "add a cache", "success")
    tools.record_failure("api latency", "raise thread count", "thrashing")

    failures = tools.recall_experiences("api latency", outcome="failure")
    assert [e["action"] for e in failures["experiences"]] == ["raise thread count"]


def test_recommend_reports_the_track_record(tools):
    for _ in range(5):
        tools.record_experience("api latency", "add a cache", "success")

    rec = tools.recommend_action("api latency")
    assert rec["found"] is True
    assert rec["recommended_action"] == "add a cache"
    assert rec["success_count"] == 5
    assert 0.0 < rec["confidence"] <= 1.0
    assert rec["supporting_ids"]


def test_recommend_explains_itself_when_memory_is_empty(tools):
    rec = tools.recommend_action("something never seen")
    assert rec["found"] is False
    assert "record" in rec["detail"]


def test_recommend_can_explore(tools):
    for _ in range(5):
        tools.record_experience("api latency", "add a cache", "success")
    assert tools.recommend_action("api latency", explore=True)["found"] is True


def test_recent_and_stats(tools):
    tools.record_experience("first task", "first action")
    tools.record_experience("second task", "second action")

    recent = tools.recent_experiences()
    assert recent["count"] == 2
    assert recent["experiences"][0]["task"] == "second task"  # newest first
    assert tools.memory_stats()["total_experiences"] == 2


def test_oversized_requests_are_clamped(tools):
    for i in range(12):
        tools.record_experience(f"task {i} about latency", "an action")

    assert tools.recall_experiences("latency", k=10_000)["count"] <= 50
    assert tools.recent_experiences(n=10_000)["count"] <= 100


def test_bad_outcome_names_the_valid_options(tools):
    with pytest.raises(ValueError, match="success, failure, partial"):
        tools.record_experience("a task", "an action", outcome="worked-ish")


def test_parse_outcome_accepts_valid_values():
    assert parse_outcome("partial").value == "partial"


def test_clamp_bounds_values():
    assert clamp(0, 1, 50) == 1
    assert clamp(99, 1, 50) == 50
    assert clamp(7, 1, 50) == 7


def test_serialized_experience_omits_the_embedding(memory):
    exp = memory.record("a task", "an action")
    payload = experience_to_dict(exp)
    assert "embedding" not in payload  # clients can't use hundreds of floats
    assert payload["superseded"] is False


def test_default_db_path_prefers_the_environment(monkeypatch):
    monkeypatch.setenv("MIMIR_DB_PATH", "/tmp/shared.db")
    assert default_db_path() == "/tmp/shared.db"

    monkeypatch.delenv("MIMIR_DB_PATH")
    assert default_db_path() == str(DEFAULT_DB_PATH)


def test_shared_store_is_visible_to_a_second_instance(tmp_path):
    # Two Mimir instances on one file, as two MCP client processes would be.
    db = str(tmp_path / "shared.db")
    first, second = Mimir(db), Mimir(db)
    try:
        MimirTools(first).record_experience("shared task", "shared action", "success")
        found = MimirTools(second).recall_experiences("shared task")
        assert found["experiences"][0]["action"] == "shared action"
    finally:
        first.close()
        second.close()


EXPECTED_TOOLS = {
    "record_experience",
    "record_failure",
    "recall_experiences",
    "recommend_action",
    "recent_experiences",
    "memory_stats",
}


@pytest.fixture
def server(tools):
    pytest.importorskip("mcp", reason="the mcp extra is not installed")
    return build_server(tools)


def listed(server):
    # The MCP API is async; drive it directly rather than add a pytest plugin.
    return asyncio.run(server.list_tools())


def test_server_registers_every_tool(server):
    assert {tool.name for tool in listed(server)} == EXPECTED_TOOLS


def test_every_tool_is_described_for_the_model(server):
    for tool in listed(server):
        assert tool.description, f"{tool.name} has no description"


def test_read_and_write_tools_are_annotated(server):
    modes = {t.name: t.annotations.read_only_hint for t in listed(server)}
    assert modes["recall_experiences"] is True
    assert modes["recommend_action"] is True
    assert modes["memory_stats"] is True
    assert modes["record_experience"] is False
    assert modes["record_failure"] is False


def test_calling_a_tool_through_the_server_writes_to_the_store(server, tools):
    asyncio.run(
        server.call_tool(
            "record_experience",
            {"task": "fix the deploy", "action": "roll back first", "outcome": "success"},
        )
    )
    assert tools.memory_stats()["total_experiences"] == 1
