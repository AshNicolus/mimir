"""Storage behaviour: validation, persistence, migration, indexes, concurrency."""

import threading

import pytest
from pydantic import ValidationError

from mimir import Experience, Mimir


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
    memory.write(exp)
    exp.action = "second attempt"
    memory.write(exp)

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


def test_recommend_works_on_database_without_action_norm(tmp_path):
    # A database written before the action_norm column existed must be migrated
    # and backfilled on open so recommend() still groups correctly.
    import sqlite3

    db = str(tmp_path / "legacy.db")
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE experiences (id TEXT PRIMARY KEY, task TEXT NOT NULL, "
        "action TEXT NOT NULL, outcome TEXT NOT NULL, score REAL NOT NULL, "
        "context TEXT NOT NULL, embedding TEXT, created_at TEXT NOT NULL, superseded_by TEXT)"
    )
    con.execute("CREATE VIRTUAL TABLE experiences_fts USING fts5(id UNINDEXED, task, action)")
    for i in range(5):
        con.execute(
            "INSERT INTO experiences VALUES (?,?,?,?,?,?,?,?,?)",
            (
                str(i),
                "auth is slow",
                "Redis caching",
                "success",
                1.0,
                "{}",
                None,
                "2026-01-01",
                None,
            ),
        )
        con.execute(
            "INSERT INTO experiences_fts VALUES (?,?,?)", (str(i), "auth is slow", "Redis caching")
        )
    con.commit()
    con.close()

    m = Mimir(db_path=db)
    try:
        rec = m.recommend("auth is slow")
        assert rec is not None
        assert rec.recommended_action == "Redis caching"
        assert rec.success_count == 5
        assert rec.based_on == 5
    finally:
        m.close()


def index_names(memory):
    rows = memory._storage._conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'index'"
    ).fetchall()
    return {row["name"] for row in rows}


def test_outcome_index_is_not_created(memory):
    # The outcome index is never used for reads (FTS recall joins by primary key
    # and filters outcome as a residual), so it must not exist.
    assert "idx_experiences_outcome" not in index_names(memory)


def test_outcome_index_is_dropped_on_reopen(tmp_path):
    # A database created before this change should shed the stale index too.
    db = str(tmp_path / "mimir.db")
    seed = Mimir(db_path=db)
    seed._storage._conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_experiences_outcome ON experiences(outcome)"
    )
    seed._storage._conn.commit()
    seed.close()

    reopened = Mimir(db_path=db)
    try:
        assert "idx_experiences_outcome" not in index_names(reopened)
    finally:
        reopened.close()


def test_context_manager_closes(tmp_path):
    db = str(tmp_path / "ctx.db")
    with Mimir(db_path=db) as m:
        m.record("task", "action")
        assert m.count() == 1
