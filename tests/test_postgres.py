"""Behaviour only the Postgres backend has: shared concurrent writers, schemas,
DSN redaction. The shared behaviour runs through the whole suite instead, when
MIMIR_TEST_POSTGRES_DSN is set (see conftest.py)."""

import os
import threading
import uuid

import pytest

from mimir import Mimir
from mimir.storage import postgres
from mimir.storage.postgres import PostgresStorage, redact

DSN = os.environ.get("MIMIR_TEST_POSTGRES_DSN")
needs_pg = pytest.mark.skipif(not DSN, reason="set MIMIR_TEST_POSTGRES_DSN to run")


@pytest.fixture
def schema():
    name = f"mt_{uuid.uuid4().hex[:16]}"
    yield name
    import psycopg

    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute(f'DROP SCHEMA IF EXISTS "{name}" CASCADE')


def store(schema):
    return Mimir(storage=PostgresStorage(DSN, schema=schema))


def test_redact_masks_passwords_in_both_dsn_forms():
    assert redact("postgresql://u:s3cret@h:5432/db") == "postgresql://u:***@h:5432/db"
    assert redact("host=h password=s3cret dbname=x") == "host=h password=*** dbname=x"
    assert redact("postgresql://h/db?password=s3cret&sslmode=require") == (
        "postgresql://h/db?password=***&sslmode=require"
    )
    assert redact("postgresql://u@h/db") == "postgresql://u@h/db"


def test_missing_driver_names_the_extra(monkeypatch):
    monkeypatch.setattr(postgres, "psycopg", None)
    with pytest.raises(ImportError, match=r"mimir-learn\[postgres\]"):
        PostgresStorage("postgresql://h/db")


def test_schema_must_be_a_plain_identifier():
    if postgres.psycopg is None:
        pytest.skip("psycopg not installed")
    with pytest.raises(ValueError, match="identifier"):
        PostgresStorage("postgresql://h/db", schema='x"; DROP TABLE experiences; --')
    with pytest.raises(ValueError, match="pg_"):
        PostgresStorage("postgresql://h/db", schema="pg_catalog")


@needs_pg
def test_postgres_url_routes_to_postgres_and_hides_the_password():
    m = Mimir(DSN)
    try:
        assert isinstance(m.storage, PostgresStorage)
        assert m.storage.db_path == redact(DSN)  # what memory_stats shows MCP clients
    finally:
        m.close()


@needs_pg
def test_dotted_words_are_searchable_by_their_parts(schema):
    # Postgres's own parser keeps "sqlite.py" as one word; Mimir's tokenizer splits it.
    m = store(schema)
    try:
        m.record("speed up sqlite.py on db.example.com", "add an index")
        assert [e.action for e in m.recall("sqlite")] == ["add an index"]
        assert [e.action for e in m.recall("example")] == ["add an index"]
    finally:
        m.close()


@needs_pg
def test_schemas_are_separate_stores(schema):
    a = store(schema)
    b = Mimir(storage=PostgresStorage(DSN, schema=schema + "_b"))
    try:
        a.record("deploy is flaky", "pin the base image")
        assert a.count() == 1 and b.count() == 0
    finally:
        a.close()
        b.close()
        import psycopg

        with psycopg.connect(DSN, autocommit=True) as conn:
            conn.execute(f'DROP SCHEMA IF EXISTS "{schema}_b" CASCADE')


@needs_pg
def test_stores_opening_at_once_migrate_once(schema):
    errors, stores = [], []

    def open_store():
        try:
            stores.append(PostgresStorage(DSN, schema=schema))
        except Exception as exc:  # noqa: BLE001 - surfaced by the assert below
            errors.append(exc)

    threads = [threading.Thread(target=open_store) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    with stores[0].reading() as conn:
        versions = conn.execute("SELECT version FROM mimir_schema").fetchall()
    assert [r["version"] for r in versions] == [len(postgres.MIGRATIONS)]
    for s in stores:
        s.close()


@needs_pg
def test_concurrent_writers_from_separate_connections(schema):
    # The point of this backend: several agents writing to one store at once,
    # including racing to re-record the same experience.
    from mimir import Experience

    writers = [store(schema) for _ in range(4)]
    shared = Experience(id="shared", task="shared task", action="shared action")
    errors = []

    def write(m, n):
        try:
            for i in range(25):
                m.record(f"task {n}-{i}", "an action", outcome="success")
                m.write(shared)
        except Exception as exc:  # noqa: BLE001 - surfaced by the assert below
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(m, n)) for n, m in enumerate(writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    try:
        assert errors == []
        assert writers[0].count() == 101  # 100 distinct rows + one shared row
        rec = writers[0].recommend("task an action")
        assert rec.recommended_action == "an action" and rec.success_count == 100
    finally:
        for m in writers:
            m.close()


@needs_pg
def test_rerecord_replaces_the_whole_row(schema):
    m = store(schema)
    try:
        first = m.record("deploy is flaky", "retry", context={"env": "prod"})
        m.record("login is slow", "add a cache")
        first.action, first.context = "pin the base image", {}
        m.write(first)
        assert m.count() == 2
        assert m.get(first.id).action == "pin the base image"
        assert m.get(first.id).context == {}
        assert [e.action for e in m.recall("deploy flaky")] == ["pin the base image"]
    finally:
        m.close()


# The pgvector index. These need a server with the extension (CI's
# pgvector/pgvector image); without it the store must fall back to the scan,
# which the whole suite covers when MIMIR_TEST_PGVECTOR is unset.
needs_vec = pytest.mark.skipif(
    not (DSN and os.environ.get("MIMIR_TEST_PGVECTOR")),
    reason="set MIMIR_TEST_POSTGRES_DSN and MIMIR_TEST_PGVECTOR=1 to run",
)


def embedded(task, vec, **fields):
    from mimir import Experience

    return Experience(task=task, action=f"act on {task}", embedding=vec, **fields)


def points(n, dim, seed):
    import random

    rng = random.Random(seed)
    return [[rng.uniform(-1, 1) for _ in range(dim)] for _ in range(n)]


@needs_vec
def test_index_is_built_on_first_embedding_and_used(schema):
    m = store(schema)
    try:
        assert m.storage.vec_schema is not None
        m.write(embedded("first", [1.0, 0.0, 0.0]))
        assert m.storage.vec_dim == 3
        with m.storage.reading() as conn, conn.transaction():
            conn.execute("SET LOCAL enable_seqscan = off")
            plan = conn.execute(
                "EXPLAIN SELECT id FROM experiences e WHERE cardinality(e.embedding) = 3 "
                f"ORDER BY (e.embedding::{m.storage.vec_schema}.vector(3)) "
                f"OPERATOR({m.storage.vec_schema}.<=>) '[1,0,0]'::{m.storage.vec_schema}.vector(3) "
                "LIMIT 5"
            ).fetchall()
        assert postgres.VEC_INDEX in " ".join(r["QUERY PLAN"] for r in plan)
    finally:
        m.close()


@needs_vec
def test_ann_returns_the_same_neighbours_as_the_exact_scan(schema):
    m = store(schema)
    try:
        for i, vec in enumerate(points(300, 8, seed=1)):
            m.write(embedded(f"task {i}", vec))
        assert m.storage.vec_dim == 8
        for query in points(10, 8, seed=2):
            ann = m.storage.vector_search_ann(query, 5, None, None, False)
            scan = m.storage.vector_search_scan(query, 5, None, None, False)
            assert ann is not None
            # Ids, not scores: the index compares in float4, the scan in float8.
            assert [e.id for e, _ in ann] == [e.id for e, _ in scan]
    finally:
        m.close()


@needs_vec
def test_ann_respects_filters_and_never_returns_short(schema):
    m = store(schema)
    try:
        for i, vec in enumerate(points(60, 4, seed=3)):
            outcome, score = ("failure", 0.0) if i % 3 == 0 else ("success", 1.0)
            m.write(
                embedded(
                    f"task {i}", vec, outcome=outcome, score=score, context={"team": f"t{i % 2}"}
                )
            )
        old = m.write(embedded("stale", [1.0, 1.0, 1.0, 1.0]))
        m.supersede(old.id, m.write(embedded("fresh", [1.0, 1.0, 1.0, 0.9])).id)

        q = [1.0, 1.0, 1.0, 1.0]
        failures = m.storage.vector_search(q, k=5, outcome="failure")
        assert len(failures) == 5 and all(e.outcome.value == "failure" for e, _ in failures)
        team = m.storage.vector_search(q, k=5, context={"team": "t1"})
        assert len(team) == 5 and all(e.context["team"] == "t1" for e, _ in team)
        assert old.id not in [e.id for e, _ in m.storage.vector_search(q, k=5)]
        # No row matches, so HNSW comes back short and the exact scan answers.
        assert [e.task for e, _ in m.storage.vector_search(q, k=5, context={"team": "none"})] == []
    finally:
        m.close()


@needs_vec
def test_rows_written_before_the_index_are_covered(schema):
    # Stores written by 0.2.0, or before pgvector was installed, already hold
    # embeddings; reopening builds the index over them with no backfill.
    m = store(schema)
    vecs = points(20, 6, seed=4)
    for i, vec in enumerate(vecs):
        m.write(embedded(f"task {i}", vec))
    with m.storage.reading() as conn:
        conn.execute(f'DROP INDEX "{schema}".{postgres.VEC_INDEX}')
    m.close()

    reopened = store(schema)
    try:
        assert reopened.storage.vec_dim == 6
        ann = reopened.storage.vector_search_ann(vecs[0], 1, None, None, False)
        assert ann is not None and ann[0][0].task == "task 0"
    finally:
        reopened.close()


@needs_vec
def test_other_dimensions_warn_and_fall_back_to_the_scan(schema):
    from mimir import VectorIndexWarning

    m = store(schema)
    try:
        m.write(embedded("three", [1.0, 0.0, 0.0]))
        with pytest.warns(VectorIndexWarning, match="not indexed"):
            m.write(embedded("four", [0.0, 1.0, 0.0, 0.0]))
        assert [e.task for e, _ in m.storage.vector_search([0.0, 1.0, 0.0, 0.0], k=1)] == ["four"]
    finally:
        m.close()


@needs_vec
def test_large_existing_stores_get_the_sql_instead_of_a_blocking_build(schema, monkeypatch):
    from mimir import VectorIndexWarning

    m = store(schema)
    for i, vec in enumerate(points(3, 4, seed=5)):
        m.write(embedded(f"task {i}", vec))
    with m.storage.reading() as conn:
        conn.execute(f'DROP INDEX "{schema}".{postgres.VEC_INDEX}')
    m.close()

    monkeypatch.setattr(postgres, "VEC_INLINE_BUILD_MAX", 2)
    with pytest.warns(VectorIndexWarning, match="CREATE INDEX CONCURRENTLY"):
        reopened = store(schema)
    try:
        assert reopened.storage.vec_dim is None and reopened.storage.vec_skipped
        assert len(reopened.storage.vector_search([1.0, 0.0, 0.0, 0.0], k=2)) == 2  # scan works
    finally:
        reopened.close()
