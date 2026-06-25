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

from ..clustering import ActionClusterer, Cluster, ExactClusterer, normalize_action
from ..embeddings import cosine_similarity
from ..models import Experience, Outcome
from .base import ActionStat, Storage

TOKEN = re.compile(r"[a-z0-9]+")
SIMPLE_KEY = re.compile(r"[A-Za-z0-9_]+")

# Common function words that add retrieval noise without signal.
STOPWORDS = frozenset(
    "a an the is are was were be been to of in on for and or it this that with "
    "as at by from how what i my we".split()
)


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def query_terms(query: str) -> list[str]:
    terms = tokenize(query)
    meaningful = [t for t in terms if t not in STOPWORDS]
    return meaningful or terms  # fall back if the query is all stopwords


class SQLiteStorage(Storage):
    def __init__(
        self, db_path: str = ":memory:", *, clusterer: ActionClusterer | None = None
    ) -> None:
        if db_path not in (":memory:", "") and not db_path.startswith("file:"):
            parent = os.path.dirname(os.path.abspath(db_path))
            os.makedirs(parent, exist_ok=True)
        self._clusterer = clusterer or ExactClusterer()
        # check_same_thread=False + an explicit lock lets the store be shared
        # across threads safely.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._fts = self.init_schema()

    def init_schema(self) -> bool:
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

    def known_clusters(self) -> list[Cluster]:
        # Representative phrasing per cluster, for a clusterer to merge against.
        rows = self._conn.execute(
            "SELECT action_norm AS key, MIN(action) AS action FROM experiences GROUP BY action_norm"
        ).fetchall()
        return [Cluster(key=r["key"], action=r["action"]) for r in rows]

    def add(self, exp: Experience) -> None:
        with self._lock:
            action_norm = self._clusterer.key(exp.action, self.known_clusters)
            self._conn.execute(
                "INSERT OR REPLACE INTO experiences "
                "(id, task, action, action_norm, outcome, score, context, embedding, "
                "created_at, superseded_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    exp.id,
                    exp.task,
                    exp.action,
                    action_norm,
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
            cur = self._conn.execute("DELETE FROM experiences WHERE id = ?", (experience_id,))
            if self._fts:
                self._conn.execute("DELETE FROM experiences_fts WHERE id = ?", (experience_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def get(self, experience_id: str) -> Experience | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM experiences WHERE id = ?", (experience_id,)
            ).fetchone()
        return self.row_to_experience(row) if row else None

    def recent(self, n: int = 10) -> list[Experience]:
        with self._lock:
            # Tie-break on rowid (monotonic insertion order) because created_at
            # resolution can collide for records written in the same instant.
            rows = self._conn.execute(
                "SELECT * FROM experiences ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (n,),
            ).fetchall()
        return [self.row_to_experience(r) for r in rows]

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]

    def aggregate_actions(self, query: str) -> list[ActionStat]:
        terms = query_terms(query)
        if not terms:
            return []

        if self._fts:
            # Group and count in SQL: one row per action, not one per row. The
            # inner query scores each match with bm25 so the outer query can sum
            # relevance per action alongside the raw counts.
            match = " OR ".join(f'"{t}"' for t in terms)
            with self._lock:
                rows = self._conn.execute(
                    "SELECT key, MIN(action) AS action, "
                    "SUM(outcome = 'success') AS success, "
                    "SUM(outcome = 'failure') AS failure, "
                    "SUM(outcome = 'partial') AS partial, "
                    "COUNT(*) AS total, "
                    "SUM(weight * (outcome = 'success')) AS weighted_success, "
                    "SUM(weight * (outcome = 'partial')) AS weighted_partial, "
                    "SUM(weight) AS weighted_total "
                    "FROM ("
                    # bm25 is negative and more relevant the lower it goes, so
                    # s = -bm25 is the relevance magnitude. s/(1+s) squashes it
                    # into [0, 1) monotonically, staying sensitive at the strong
                    # end where the simpler 1/(1+rank) clamp would flatten to 1.
                    "  SELECT key, action, outcome, s / (1.0 + s) AS weight FROM ("
                    "    SELECT e.action_norm AS key, e.action AS action, e.outcome AS outcome, "
                    "    max(-bm25(experiences_fts), 0.0) AS s "
                    "    FROM experiences_fts JOIN experiences e ON e.id = experiences_fts.id "
                    "    WHERE experiences_fts MATCH ? "
                    # LIMIT keeps SQLite from flattening this into the outer
                    # aggregate, where bm25() is not allowed to run.
                    "    LIMIT -1"
                    "  )"
                    ") "
                    "GROUP BY key",
                    (match,),
                ).fetchall()
            return [
                ActionStat(
                    action=row["action"],
                    key=row["key"],
                    success=row["success"],
                    failure=row["failure"],
                    partial=row["partial"],
                    total=row["total"],
                    weighted_success=row["weighted_success"],
                    weighted_partial=row["weighted_partial"],
                    weighted_total=row["weighted_total"],
                )
                for row in rows
            ]

        # No FTS5: group in Python over the lightweight columns. Relevance is
        # the fraction of query terms a row matches, same as fallback search.
        wanted = set(terms)
        with self._lock:
            rows = self._conn.execute(
                "SELECT task, action, action_norm, outcome FROM experiences"
            ).fetchall()
        groups: dict[str, dict] = {}
        for row in rows:
            matched = wanted & set(tokenize(f"{row['task']}\n{row['action']}"))
            if not matched:
                continue
            weight = len(matched) / len(wanted)
            key = row["action_norm"]  # the stored cluster key
            group = groups.setdefault(
                key,
                {
                    "action": row["action"],
                    "success": 0,
                    "failure": 0,
                    "partial": 0,
                    "weighted_success": 0.0,
                    "weighted_partial": 0.0,
                    "weighted_total": 0.0,
                },
            )
            group[row["outcome"]] += 1
            group["weighted_total"] += weight
            if row["outcome"] in ("success", "partial"):
                group[f"weighted_{row['outcome']}"] += weight
            group["action"] = min(group["action"], row["action"])  # stable representative
        return [
            ActionStat(
                action=g["action"],
                key=key,
                success=g["success"],
                failure=g["failure"],
                partial=g["partial"],
                total=g["success"] + g["failure"] + g["partial"],
                weighted_success=g["weighted_success"],
                weighted_partial=g["weighted_partial"],
                weighted_total=g["weighted_total"],
            )
            for key, g in groups.items()
        ]

    def supporting_ids(self, query: str, action_key: str, limit: int = 100) -> list[str]:
        terms = query_terms(query)
        if not terms:
            return []
        if self._fts:
            match = " OR ".join(f'"{t}"' for t in terms)
            with self._lock:
                rows = self._conn.execute(
                    "SELECT e.id FROM experiences_fts JOIN experiences e "
                    "ON e.id = experiences_fts.id "
                    "WHERE experiences_fts MATCH ? AND e.action_norm = ? LIMIT ?",
                    (match, action_key, limit),
                ).fetchall()
            return [row["id"] for row in rows]

        wanted = set(terms)
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, task, action, action_norm FROM experiences WHERE action_norm = ?",
                (action_key,),
            ).fetchall()
        ids = []
        for row in rows:
            if wanted & set(tokenize(f"{row['task']}\n{row['action']}")):
                ids.append(row["id"])
                if len(ids) >= limit:
                    break
        return ids

    def vector_search(
        self,
        embedding: list[float],
        k: int | None = 5,
        outcome: str | None = None,
        context: dict | None = None,
    ) -> list[tuple[Experience, float]]:
        if not embedding:
            return []
        # Scan embedded rows and rank by cosine in Python. This is O(N) and the
        # dependency-free fallback; a vector-index backend overrides the method.
        filters = context_sql_filters(context)
        where = ["embedding IS NOT NULL"]
        params: list[object] = []
        if outcome is not None:
            where.append("outcome = ?")
            params.append(outcome)
        for path, value in filters:
            where.append("json_extract(context, ?) = ?")
            params += [path, value]
        sql = "SELECT * FROM experiences WHERE " + " AND ".join(where)

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        scored = []
        for row in rows:
            exp = self.row_to_experience(row)
            if context and not context_matches(exp.context, context):
                continue
            sim = cosine_similarity(embedding, exp.embedding) if exp.embedding else 0.0
            if sim > 0:
                scored.append((exp, sim))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k] if k is not None else scored

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
            scored = self.fts_search(query, outcome, filters, limit)
        else:
            scored = self.fallback_search(query, outcome, filters)

        # Exact check for context values SQL can't compare (nested, missing).
        if context:
            scored = [(e, s) for e, s in scored if context_matches(e.context, context)]
        return scored[:k] if k is not None else scored

    def fts_search(
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
        return [(self.row_to_experience(r), bm25_to_score(r["rank"])) for r in rows]

    def fallback_search(
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
            exp = self.row_to_experience(row)
            overlap = len(terms & set(tokenize(exp.text())))
            if overlap:
                scored.append((exp, overlap / len(terms)))  # fraction of terms matched
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored

    def row_to_experience(self, row: sqlite3.Row) -> Experience:
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
        if isinstance(value, (str, int, float)) and SIMPLE_KEY.fullmatch(key):
            filters.append((f"$.{key}", value))
    return filters


def bm25_to_score(rank: float) -> float:
    """Squash an unbounded, negative-ish bm25 rank into a stable (0, 1] score."""
    return 1.0 / (1.0 + max(rank, 0.0))
