"""SQLite storage backend — the v1 default.

No external service required: the store is a single file (or in-memory). Keyword
recall uses SQLite's FTS5 when available, and falls back to a portable
Python token-overlap scorer when it isn't, so this works everywhere CPython does.

The backend is internally thread-safe: every connection access is guarded by a
lock, so reads and writes from multiple threads are serialized safely.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from datetime import datetime

from ..models import Experience, Outcome
from .base import Storage

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class SQLiteStorage(Storage):
    def __init__(self, db_path: str = ":memory:") -> None:
        if db_path not in (":memory:", "") and not db_path.startswith("file:"):
            parent = os.path.dirname(os.path.abspath(db_path))
            os.makedirs(parent, exist_ok=True)
        # check_same_thread=False + an explicit lock lets the store be shared
        # across threads safely.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._fts = self._init_schema()

    # -- schema -------------------------------------------------------------

    def _init_schema(self) -> bool:
        with self._lock:
            # WAL improves read/write concurrency and durability; NORMAL sync is
            # the standard safe-and-fast pairing with WAL.
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiences (
                    id            TEXT PRIMARY KEY,
                    task          TEXT NOT NULL,
                    action        TEXT NOT NULL,
                    outcome       TEXT NOT NULL,
                    score         REAL NOT NULL,
                    context       TEXT NOT NULL,
                    embedding     TEXT,
                    created_at    TEXT NOT NULL,
                    superseded_by TEXT
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_experiences_outcome ON experiences(outcome)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_experiences_created ON experiences(created_at)"
            )
            fts_ok = True
            try:
                self._conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS experiences_fts "
                    "USING fts5(id UNINDEXED, task, action)"
                )
            except sqlite3.OperationalError:
                fts_ok = False  # FTS5 not compiled in — use the Python fallback
            self._conn.commit()
            return fts_ok

    # -- writes -------------------------------------------------------------

    def add(self, exp: Experience) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO experiences "
                "(id, task, action, outcome, score, context, embedding, created_at, "
                "superseded_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    exp.id,
                    exp.task,
                    exp.action,
                    exp.outcome.value,
                    exp.score,
                    json.dumps(exp.context),
                    json.dumps(exp.embedding) if exp.embedding is not None else None,
                    exp.created_at.isoformat(),
                    exp.superseded_by,
                ),
            )
            if self._fts:
                # Keep FTS in sync on re-record: clear any stale row for this id
                # first, otherwise INSERT OR REPLACE above would leave a duplicate
                # FTS entry and skew search results.
                self._conn.execute("DELETE FROM experiences_fts WHERE id = ?", (exp.id,))
                self._conn.execute(
                    "INSERT INTO experiences_fts (id, task, action) VALUES (?, ?, ?)",
                    (exp.id, exp.task, exp.action),
                )
            self._conn.commit()

    def delete(self, experience_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM experiences WHERE id = ?", (experience_id,)
            )
            if self._fts:
                self._conn.execute(
                    "DELETE FROM experiences_fts WHERE id = ?", (experience_id,)
                )
            self._conn.commit()
            return cur.rowcount > 0

    # -- reads --------------------------------------------------------------

    def get(self, experience_id: str) -> Experience | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM experiences WHERE id = ?", (experience_id,)
            ).fetchone()
        return self._row_to_experience(row) if row else None

    def recent(self, n: int = 10) -> list[Experience]:
        with self._lock:
            # Tie-break on rowid (monotonic insertion order) because created_at
            # resolution can collide for records written in the same instant.
            rows = self._conn.execute(
                "SELECT * FROM experiences ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (n,),
            ).fetchall()
        return [self._row_to_experience(r) for r in rows]

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]

    def search(
        self,
        query: str,
        k: int | None = 5,
        outcome: str | None = None,
        context: dict | None = None,
    ) -> list[tuple[Experience, float]]:
        scored = self._fts_search(query) if self._fts else self._fallback_search(query)
        results: list[tuple[Experience, float]] = []
        for exp, score in scored:
            if outcome is not None and exp.outcome.value != outcome:
                continue
            if context and not _context_matches(exp.context, context):
                continue
            results.append((exp, score))
            if k is not None and len(results) >= k:
                break
        return results

    def _fts_search(self, query: str) -> list[tuple[Experience, float]]:
        terms = _tokens(query)
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in terms)
        with self._lock:
            rows = self._conn.execute(
                "SELECT e.*, bm25(experiences_fts) AS rank "
                "FROM experiences_fts "
                "JOIN experiences e ON e.id = experiences_fts.id "
                "WHERE experiences_fts MATCH ? "
                "ORDER BY rank",  # bm25: lower is better
                (match,),
            ).fetchall()
        out = []
        for row in rows:
            # bm25 is an unbounded negative-ish score; squash to (0, 1] for a stable API
            rank = row["rank"]
            score = 1.0 / (1.0 + max(rank, 0.0))
            out.append((self._row_to_experience(row), score))
        return out

    def _fallback_search(self, query: str) -> list[tuple[Experience, float]]:
        q = set(_tokens(query))
        if not q:
            return []
        with self._lock:
            rows = self._conn.execute("SELECT * FROM experiences").fetchall()
        scored = []
        for row in rows:
            exp = self._row_to_experience(row)
            doc = set(_tokens(exp.text()))
            if not doc:
                continue
            overlap = len(q & doc)
            if overlap == 0:
                continue
            score = overlap / len(q)  # fraction of query terms matched
            scored.append((exp, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored

    # -- helpers ------------------------------------------------------------

    def _row_to_experience(self, row: sqlite3.Row) -> Experience:
        return Experience(
            id=row["id"],
            task=row["task"],
            action=row["action"],
            outcome=Outcome(row["outcome"]),
            score=row["score"],
            context=json.loads(row["context"]),
            embedding=json.loads(row["embedding"]) if row["embedding"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
            superseded_by=row["superseded_by"],
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _context_matches(stored: dict, wanted: dict) -> bool:
    return all(stored.get(key) == value for key, value in wanted.items())
