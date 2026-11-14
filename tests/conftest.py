import os
import uuid

import pytest

import mimir.core
from mimir import Mimir

# Set to a Postgres DSN to run the whole suite against PostgresStorage: every
# in-memory store becomes a fresh schema in that database, dropped afterwards.
PG_DSN = os.environ.get("MIMIR_TEST_POSTGRES_DSN")


@pytest.fixture(autouse=True)
def backend(request, monkeypatch):
    if not PG_DSN:
        yield
        return
    if request.node.get_closest_marker("sqlite_only"):
        pytest.skip("exercises SQLite internals")

    import psycopg

    from mimir.storage import PostgresStorage

    sqlite_storage = mimir.core.SQLiteStorage
    created = []

    def storage_for(db_path=":memory:", *, clusterer=None):
        if db_path not in (":memory:", ""):
            return sqlite_storage(db_path, clusterer=clusterer)  # file stores stay SQLite
        store = PostgresStorage(PG_DSN, schema=f"t_{uuid.uuid4().hex[:16]}", clusterer=clusterer)
        created.append(store)
        return store

    monkeypatch.setattr(mimir.core, "SQLiteStorage", storage_for)
    yield
    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        for store in created:
            store.close()
            conn.execute(f'DROP SCHEMA IF EXISTS "{store.schema}" CASCADE')


@pytest.fixture
def memory():
    m = Mimir(db_path=":memory:")
    yield m
    m.close()
