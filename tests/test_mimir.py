"""v1 behaviour tests for Mimir. All run against an in-memory SQLite store."""

import threading

import pytest
from pydantic import ValidationError

from mimir import Experience, Mimir, Outcome


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


def test_search_with_zero_limit_returns_empty(memory):
    memory.record("a task", "an action")
    assert memory._storage.search("task", k=0) == []


def test_recall_ignores_common_stopwords(memory):
    memory.record("improve database latency", "add an index")
    memory.record("the weather is nice today", "go outside")
    results = memory.recall("what is the latency")
    assert len(results) == 1
    assert results[0].action == "add an index"


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


def test_recommend_counts_full_population_not_a_sample(memory):
    # Regression: counts used to cap at k=20; must reflect the real total.
    for _ in range(30):
        memory.record("auth is slow", "Redis caching", outcome="success")

    rec = memory.recommend("auth is slow")

    assert rec is not None
    assert rec.success_count == 30
    assert rec.based_on == 30


def test_recommend_returns_none_without_data(memory):
    assert memory.recommend("anything at all") is None


def test_recommend_ignores_actions_that_only_failed(memory):
    memory.record_failure("deploy fails", "force push", reason="broke prod")
    assert memory.recommend("deploy fails") is None


def test_recommend_skips_failed_action_for_a_proven_one(memory):
    memory.record_failure("deploy fails", "force push")
    memory.record("deploy fails", "run migrations first", outcome="success")
    rec = memory.recommend("deploy fails")
    assert rec is not None
    assert rec.recommended_action == "run migrations first"


def test_recommendation_str_is_readable(memory):
    memory.record("auth slow", "Redis caching", outcome="success")
    rec = memory.recommend("auth slow")
    text = str(rec)
    assert "Redis caching" in text
    assert "confidence" in text


# -- robustness ------------------------------------------------------------


def test_blank_task_or_action_is_rejected(memory):
    with pytest.raises(ValidationError):
        memory.record("   ", "did something")
    with pytest.raises(ValidationError):
        memory.record("a real task", "")


def test_task_and_action_are_trimmed(memory):
    exp = memory.record("  fix login  ", "  add cache  ")
    assert exp.task == "fix login"
    assert exp.action == "add cache"


def test_invalid_score_is_rejected(memory):
    with pytest.raises(ValidationError):
        memory.record("task", "action", score=1.5)


def test_non_json_context_raises_clear_error(memory):
    with pytest.raises(ValueError, match="JSON-serializable"):
        memory.record("task", "action", context={"obj": object()})


def test_get_and_delete(memory):
    exp = memory.record("fix login", "add cache")
    assert memory.get(exp.id).action == "add cache"
    assert memory.delete(exp.id) is True
    assert memory.get(exp.id) is None
    assert memory.delete(exp.id) is False  # already gone


def test_recent_returns_newest_first(memory):
    memory.record("first task", "action one")
    memory.record("second task", "action two")
    memory.record("third task", "action three")
    recent = memory.recent(2)
    assert len(recent) == 2
    assert recent[0].task == "third task"


def test_rerecording_same_id_does_not_duplicate_in_search(memory):
    # Re-saving an edited experience under the same id must not leave a stale
    # FTS row (the bug the FTS dedup fix addresses).
    exp = Experience(task="fix login latency", action="first attempt")
    memory._write(exp)
    exp.action = "second attempt"
    memory._write(exp)

    results = memory.recall("login latency", k=10)
    assert len(results) == 1
    assert results[0].action == "second attempt"
    assert memory.count() == 1


def test_persists_across_reopen(tmp_path):
    db = str(tmp_path / "mimir.db")
    m1 = Mimir(db_path=db)
    m1.record("fix auth latency", "Redis cache", outcome="success", score=0.9)
    m1.close()

    m2 = Mimir(db_path=db)
    try:
        assert m2.count() == 1
        results = m2.recall("auth latency")
        assert results and results[0].action == "Redis cache"
    finally:
        m2.close()


def test_creates_parent_directory(tmp_path):
    db = str(tmp_path / "nested" / "dir" / "mimir.db")
    m = Mimir(db_path=db)
    try:
        m.record("task", "action")
        assert m.count() == 1
    finally:
        m.close()


def test_concurrent_writes_are_safe(memory):
    def worker(n):
        for i in range(20):
            memory.record(f"task {n}-{i}", f"action {n}-{i}")

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert memory.count() == 100


def test_context_manager_closes(tmp_path):
    db = str(tmp_path / "ctx.db")
    with Mimir(db_path=db) as m:
        m.record("task", "action")
        assert m.count() == 1
