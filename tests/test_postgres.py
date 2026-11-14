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
