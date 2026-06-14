"""v1 behaviour tests for Mimir. All run against an in-memory SQLite store."""

import pytest

from mimir import Mimir, Outcome


@pytest.fixture
def memory():
    m = Mimir(db_path=":memory:")
    yield m
    m.close()


def test_record_and_count(memory):
    memory.record("Fix login latency", "Added Redis cache", outcome="success", score=0.9)
    assert memory.count() == 1


def test_record_defaults_score_from_outcome(memory):
    exp = memory.record("task", "action", outcome="failure")
    assert exp.outcome is Outcome.FAILURE
    assert exp.score == 0.0


def test_recall_finds_relevant_experience(memory):
    memory.record("Fix authentication timeout", "Implemented Redis caching", score=0.95)
    memory.record("Render a chart", "Used matplotlib", score=0.8)

    results = memory.recall("authentication latency", k=5)

    assert results, "expected at least one recalled experience"
    assert "Redis" in results[0].action


def test_recall_returns_nothing_for_unrelated_query(memory):
    memory.record("Fix authentication timeout", "Implemented Redis caching")
    assert memory.recall("how to bake sourdough bread") == []


def test_record_failure_is_queryable_separately(memory):
    memory.record_failure(
        "Throttle abusive clients",
        "Fixed-window rate limiter",
        reason="WebSocket traffic wasn't handled",
    )
    failures = memory.recall("rate limiter", outcome="failure")
    assert len(failures) == 1
    assert failures[0].context["failure_reason"].startswith("WebSocket")


def test_recall_filter_by_context(memory):
    memory.record("speed up api", "add cache", context={"service": "auth"})
    memory.record("speed up api", "add index", context={"service": "billing"})

    results = memory.recall("speed up api", context={"service": "auth"})
    assert len(results) == 1
    assert results[0].action == "add cache"


def test_recommend_prefers_more_proven_action(memory):
    # "Redis caching" succeeds many times; "rewrite in rust" succeeds once.
    for _ in range(9):
        memory.record("auth is slow", "Redis caching", outcome="success")
    memory.record("auth is slow", "Redis caching", outcome="failure")
    memory.record("auth is slow", "rewrite in rust", outcome="success")

    rec = memory.recommend("authentication is slow")

    assert rec is not None
    assert rec.recommended_action == "Redis caching"
    assert rec.success_count == 9
    assert rec.failure_count == 1
    assert 0.0 < rec.confidence <= 1.0


def test_recommend_returns_none_without_data(memory):
    assert memory.recommend("anything at all") is None


def test_recommendation_str_is_readable(memory):
    memory.record("auth slow", "Redis caching", outcome="success")
    rec = memory.recommend("auth slow")
    text = str(rec)
    assert "Redis caching" in text
    assert "confidence" in text
