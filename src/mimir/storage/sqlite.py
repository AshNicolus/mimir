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
from .base import ActionStat, Storage

_TOKEN = re.compile(r"[a-z0-9]+")
_SIMPLE_KEY = re.compile(r"[A-Za-z0-9_]+")

# Common function words that add retrieval noise without signal.
_STOPWORDS = frozenset(
    "a an the is are was were be been to of in on for and or it this that with "
    "as at by from how what i my we".split()
)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def query_terms(query: str) -> list[str]:
    terms = tokenize(query)
    meaningful = [t for t in terms if t not in _STOPWORDS]
    return meaningful or terms  # fall back if the query is all stopwords


def normalize_action(action: str) -> str:
    """Collapse case and whitespace so phrasings of the same action group together."""
    return " ".join(action.lower().split())


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
                    action_norm   TEXT,
                    outcome       TEXT NOT NULL,
                    score         REAL NOT NULL,
                    context       TEXT NOT NULL,
                    embedding     TEXT,
                    created_at    TEXT NOT NULL,
                    superseded_by TEXT
                )
                """
            )
            # Backfill action_norm for databases created before this column.
            columns = {r["name"] for r in self._conn.execute("PRAGMA table_info(experiences)")}
            if "action_norm" not in columns:
                self._conn.execute("ALTER TABLE experiences ADD COLUMN action_norm TEXT")
                stale = self._conn.execute("SELECT id, action FROM experiences").fetchall()
                self._conn.executemany(
                    "UPDATE experiences SET action_norm = ? WHERE id = ?",
                    [(normalize_action(r["action"]), r["id"]) for r in stale],
                )
            # The outcome index is unused: FTS recall joins by primary key and
            # filters outcome as a residual. Drop it, including on older dbs.
            self._conn.execute("DROP INDEX IF EXISTS idx_experiences_outcome")
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

    def add(self, exp: Experience) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO experiences "
                "(id, task, action, action_norm, outcome, score, context, embedding, "
                "created_at, superseded_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    exp.id,
                    exp.task,
                    exp.action,
                    normalize_action(exp.action),
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

    def aggregate_actions(self, query: str) -> list[ActionStat]:
        terms = query_terms(query)
        if not terms:
            return []

        if self._fts:
            # Group and count in SQL: one row per action, not one per row.
            match = " OR ".join(f'"{t}"' for t in terms)
            with self._lock:
                rows = self._conn.execute(
                    "SELECT MIN(e.action) AS action, "
                    "SUM(e.outcome = 'success') AS success, "
                    "SUM(e.outcome = 'failure') AS failure, "
                    "SUM(e.outcome = 'partial') AS partial, "
                    "COUNT(*) AS total, "
                    "group_concat(e.id) AS ids "
                    "FROM experiences_fts JOIN experiences e ON e.id = experiences_fts.id "
                    "WHERE experiences_fts MATCH ? "
                    "GROUP BY e.action_norm",
                    (match,),
                ).fetchall()
            return [
                ActionStat(
                    action=row["action"],
                    success=row["success"],
                    failure=row["failure"],
                    partial=row["partial"],
                    total=row["total"],
                    supporting_ids=row["ids"].split(","),
                )
                for row in rows
            ]

        # No FTS5: group in Python over the lightweight columns.
        wanted = set(terms)
        with self._lock:
            rows = self._conn.execute("SELECT id, task, action, outcome FROM experiences").fetchall()
        groups: dict[str, dict] = {}
        for row in rows:
            if not wanted & set(tokenize(f"{row['task']}\n{row['action']}")):
                continue
            group = groups.setdefault(
                normalize_action(row["action"]),
                {"action": row["action"], "success": 0, "failure": 0, "partial": 0, "ids": []},
            )
            group[row["outcome"]] += 1
            group["ids"].append(row["id"])
            group["action"] = min(group["action"], row["action"])  # stable representative
        return [
            ActionStat(
                action=g["action"],
                success=g["success"],
                failure=g["failure"],
                partial=g["partial"],
                total=g["success"] + g["failure"] + g["partial"],
                supporting_ids=g["ids"],
            )
            for g in groups.values()
        ]

    def search(
        self,
        query: str,
        k: int | None = 5,
        outcome: str | None = None,
        context: dict | None = None,
    ) -> list[tuple[Experience, float]]:
        # Push the outcome and any scalar context filters into SQL so we don't
        # hydrate the whole store on every recall. Only apply LIMIT when every
        # filter was pushed down — otherwise a leftover Python-side context
        # check could drop matches that the limit already cut off.
        filters = context_sql_filters(context)
        fully_pushed = not context or len(filters) == len(context)
        limit = k if k is not None and fully_pushed else None

        if self._fts:
            scored = self._fts_search(query, outcome, filters, limit)
        else:
            scored = self._fallback_search(query, outcome, filters)

        # Exact check for context values SQL can't compare (nested, missing).
        if context:
            scored = [(e, s) for e, s in scored if context_matches(e.context, context)]
        return scored[:k] if k is not None else scored

    def _fts_search(
        self,
        query: str,
        outcome: str | None,
        filters: list[tuple[str, object]],
        limit: int | None,
    ) -> list[tuple[Experience, float]]:
        terms = query_terms(query)
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in terms)

        where = ["experiences_fts MATCH ?"]
        params: list[object] = [match]
        if outcome is not None:
            where.append("e.outcome = ?")
            params.append(outcome)
        for path, value in filters:
            where.append("json_extract(e.context, ?) = ?")
            params += [path, value]

        sql = (
            "SELECT e.*, bm25(experiences_fts) AS rank "
            "FROM experiences_fts JOIN experiences e ON e.id = experiences_fts.id "
            f"WHERE {' AND '.join(where)} "
            "ORDER BY rank"  # bm25: lower is better
        )
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [(self._row_to_experience(r), bm25_to_score(r["rank"])) for r in rows]

    def _fallback_search(
        self,
        query: str,
        outcome: str | None,
        filters: list[tuple[str, object]],
    ) -> list[tuple[Experience, float]]:
        terms = set(query_terms(query))
        if not terms:
            return []

        where = []
        params: list[object] = []
        if outcome is not None:
            where.append("outcome = ?")
            params.append(outcome)
        for path, value in filters:
            where.append("json_extract(context, ?) = ?")
            params += [path, value]
        sql = "SELECT * FROM experiences"
        if where:
            sql += " WHERE " + " AND ".join(where)

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        scored = []
        for row in rows:
            exp = self._row_to_experience(row)
            overlap = len(terms & set(tokenize(exp.text())))
            if overlap:
                scored.append((exp, overlap / len(terms)))  # fraction of terms matched
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored

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


def context_matches(stored: dict, wanted: dict) -> bool:
    return all(stored.get(key) == value for key, value in wanted.items())


def context_sql_filters(context: dict | None) -> list[tuple[str, object]]:
    """Context entries we can safely push into SQL as json_extract equality.

    Only plain keys and scalar values qualify; nested values, None, and keys
    with awkward characters are left for context_matches to check in Python.
    Returns (json_path, value) pairs.
    """
    if not context:
        return []
    filters = []
    for key, value in context.items():
        if isinstance(value, (str, int, float)) and _SIMPLE_KEY.fullmatch(key):
            filters.append((f"$.{key}", value))
    return filters


def bm25_to_score(rank: float) -> float:
    """Squash an unbounded, negative-ish bm25 rank into a stable (0, 1] score."""
    return 1.0 / (1.0 + max(rank, 0.0))
