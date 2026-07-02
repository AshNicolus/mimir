"""Storage behaviour: validation, persistence, migration, indexes, concurrency."""

import threading
import warnings

import pytest
from pydantic import ValidationError

from mimir import Experience, Mimir, OutcomeScoreWarning


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


def test_failure_with_high_score_warns(memory):
    with pytest.warns(OutcomeScoreWarning, match="failure"):
        exp = memory.record("deploy", "force push", outcome="failure", score=1.0)
    assert exp.score == 1.0  # still recorded, not clamped


def test_success_with_low_score_warns(memory):
    with pytest.warns(OutcomeScoreWarning, match="success"):
        memory.record("fix login", "add cache", outcome="success", score=0.1)


def test_consistent_outcome_and_score_do_not_warn(memory):
    with warnings.catch_warnings():
        warnings.simplefilter("error", OutcomeScoreWarning)
        memory.record("fix login", "add cache", outcome="success", score=0.9)
        memory.record_failure("deploy", "force push")  # defaults score to 0.0
        memory.record("tune gc", "raise heap", outcome="partial", score=1.0)  # partial never warns


def test_contradiction_can_be_escalated_to_error(memory):
    with warnings.catch_warnings():
        warnings.simplefilter("error", OutcomeScoreWarning)
        with pytest.raises(OutcomeScoreWarning):
            memory.record("deploy", "force push", outcome="failure", score=1.0)


def test_recall_of_contradictory_record_does_not_rewarn(memory):
    with pytest.warns(OutcomeScoreWarning):
        memory.record("deploy fails", "force push", outcome="failure", score=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", OutcomeScoreWarning)
        results = memory.recall("deploy", k=5)
        assert results and results[0].score == 1.0


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


def test_supersede_marks_old_and_is_still_retrievable_by_id(memory):
    old = memory.record("fix login", "add a write cache")
    new = memory.record("fix login", "add a read cache")
    assert memory.supersede(old.id, new.id) is True

    fetched = memory.get(old.id)
    assert fetched is not None
    assert fetched.superseded_by == new.id


def test_supersede_unknown_id_returns_false(memory):
    assert memory.supersede("does-not-exist", "also-missing") is False


def test_record_with_supersedes_links_in_one_call(memory):
    old = memory.record("fix login", "add a write cache")
    new = memory.record("fix login", "add a read cache", supersedes=old.id)

    assert memory.get(old.id).superseded_by == new.id
    assert [e.id for e in memory.recall("login", k=5)] == [new.id]


def test_created_at_is_normalized_to_utc():
    from datetime import datetime, timedelta, timezone

    ist = timezone(timedelta(hours=5, minutes=30))
    aware = Experience(task="t", action="a", created_at=datetime(2026, 1, 1, 12, 0, tzinfo=ist))
    assert aware.created_at == datetime(2026, 1, 1, 6, 30, tzinfo=timezone.utc)

    naive = Experience(task="t", action="a", created_at=datetime(2026, 1, 1, 12, 0))
    assert naive.created_at.tzinfo == timezone.utc  # naive read as already-UTC


def test_recent_orders_by_utc_instant_across_timezones(memory):
    # 11:00 +05:30 reads later than 09:00 UTC as a string but is earlier in real time.
    from datetime import datetime, timedelta, timezone

    ist = timezone(timedelta(hours=5, minutes=30))
    memory.write(Experience(task="early", action="a",
                            created_at=datetime(2026, 1, 1, 11, 0, tzinfo=ist)))  # 05:30 UTC
    memory.write(Experience(task="late", action="b",
                            created_at=datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)))

    assert [e.task for e in memory.recent(2)] == ["late", "early"]


def test_write_does_not_mutate_callers_experience():
    from mimir.embeddings import Embedder

    class TinyEmbedder(Embedder):
        def embed(self, text):
            return [1.0, 0.0]

    m = Mimir(":memory:", embedder=TinyEmbedder())
    try:
        exp = Experience(task="fix login", action="add cache")
        stored = m.write(exp)
        assert exp.embedding is None  # caller's object left untouched
        assert stored.embedding == [1.0, 0.0]  # returned copy carries the embedding
        assert stored.id == exp.id
    finally:
        m.close()


def test_rerecording_same_id_does_not_duplicate_in_search(memory):
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


def run_threads(target, n):
    threads = [threading.Thread(target=target, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def test_concurrent_writes_on_file_db_are_safe(tmp_path):
    m = Mimir(db_path=str(tmp_path / "m.db"))
    try:
        run_threads(lambda n: [m.record(f"task {n}-{i}", f"action {n}-{i}") for i in range(20)], 5)
        assert m.count() == 100
    finally:
        m.close()


def test_concurrent_reads_on_file_db_are_consistent(tmp_path):
    m = Mimir(db_path=str(tmp_path / "m.db"))
    for i in range(200):
        m.record(f"fix latency in service {i}", "add cache", outcome="success")
    errors = []

    def reader(_):
        try:
            for _ in range(100):
                assert m.recall("latency service", k=5)
                assert m.count() == 200
        except Exception as exc:  # surface any thread failure to the main thread
            errors.append(repr(exc))

    try:
        run_threads(reader, 6)
        assert errors == []
    finally:
        m.close()


def test_file_db_opens_a_connection_per_reader_thread(tmp_path):
    m = Mimir(db_path=str(tmp_path / "m.db"))
    m.record("task", "action")
    storage = m.storage
    assert not storage.shared

    run_threads(lambda _: m.recall("task"), 4)
    assert len(storage.connections) >= 5  # one writer plus a reader per thread

    m.close()
    assert storage.connections == []


def test_memory_db_stays_single_connection(memory):
    # In-memory databases can't share data across connections.
    assert memory.storage.shared
    run_threads(lambda _: memory.recall("anything"), 4)
    assert len(memory.storage.connections) == 1


def test_recommend_works_on_database_without_action_norm(tmp_path):
    # A pre-action_norm database must be migrated and backfilled on open.
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
    rows = memory.storage.conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'index'"
    ).fetchall()
    return {row["name"] for row in rows}


def test_outcome_index_is_not_created(memory):
    assert "idx_experiences_outcome" not in index_names(memory)


def user_version(memory):
    return memory.storage.conn.execute("PRAGMA user_version").fetchone()[0]


def test_fresh_db_is_stamped_at_latest_version(memory):
    from mimir.storage.migrations import SCHEMA_VERSION

    assert user_version(memory) == SCHEMA_VERSION


def test_outcome_index_dropped_on_upgrade(tmp_path):
    # A database one version behind sheds the stale index when reopened.
    from mimir.storage.migrations import SCHEMA_VERSION

    db = str(tmp_path / "mimir.db")
    seed = Mimir(db_path=db)
    conn = seed.storage.conn
    conn.execute("CREATE INDEX IF NOT EXISTS idx_experiences_outcome ON experiences(outcome)")
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION - 1}")
    conn.commit()
    seed.close()

    reopened = Mimir(db_path=db)
    try:
        assert "idx_experiences_outcome" not in index_names(reopened)
        assert user_version(reopened) == SCHEMA_VERSION
    finally:
        reopened.close()


def test_legacy_db_at_version_zero_fully_upgrades(tmp_path):
    import sqlite3

    from mimir.storage.migrations import SCHEMA_VERSION

    db = str(tmp_path / "legacy.db")
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE experiences (id TEXT PRIMARY KEY, task TEXT NOT NULL, "
        "action TEXT NOT NULL, outcome TEXT NOT NULL, score REAL NOT NULL, "
        "context TEXT NOT NULL, embedding TEXT, created_at TEXT NOT NULL, superseded_by TEXT)"
    )
    con.execute("CREATE INDEX idx_experiences_outcome ON experiences(outcome)")
    con.execute(
        "INSERT INTO experiences VALUES (?,?,?,?,?,?,?,?,?)",
        ("1", "auth is slow", "Redis caching", "success", 1.0, "{}", None, "2026-01-01", None),
    )
    con.commit()
    con.close()

    m = Mimir(db_path=db)
    try:
        assert user_version(m) == SCHEMA_VERSION
        columns = {r["name"] for r in m.storage.conn.execute("PRAGMA table_info(experiences)")}
        assert "action_norm" in columns
        assert "idx_experiences_outcome" not in index_names(m)
        row = m.storage.conn.execute(
            "SELECT action_norm FROM experiences WHERE id = '1'"
        ).fetchone()
        assert row["action_norm"] == "redis caching"
    finally:
        m.close()


def test_current_db_is_not_remigrated(tmp_path):
    from mimir.storage.migrations import SCHEMA_VERSION

    db = str(tmp_path / "mimir.db")
    Mimir(db_path=db).close()
    reopened = Mimir(db_path=db)
    try:
        assert user_version(reopened) == SCHEMA_VERSION
    finally:
        reopened.close()


def test_context_manager_closes(tmp_path):
    db = str(tmp_path / "ctx.db")
    with Mimir(db_path=db) as m:
        m.record("task", "action")
        assert m.count() == 1
