"""SQLite storage backend: a single file or in-memory database, no services.

Keyword recall uses FTS5 when the build has it and a Python token-overlap
scorer when it doesn't. Vector recall uses the sqlite-vec extension when
installed and a Python cosine scan otherwise.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime

try:
    import sqlite_vec
except ImportError:
    sqlite_vec = None

from ..clustering import ActionClusterer, Cluster, ExactClusterer
from ..embeddings import cosine_similarity
from ..models import Experience, Outcome
from .base import ActionStat, Storage
from .migrations import run_migrations
from .query import bm25_to_score, context_matches, context_sql_filters, query_terms, tokenize

VEC_DIM_PATTERN = re.compile(r"float\[(\d+)\]")
KNN_OVERFETCH = 8  # KNN can't filter inline, so fetch extra and filter after


class SQLiteStorage(Storage):
    def __init__(
        self, db_path: str = ":memory:", *, clusterer: ActionClusterer | None = None
    ) -> None:
        if db_path not in (":memory:", "") and not db_path.startswith("file:"):
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.db_path = db_path
        # In-memory databases can't share data across connections, so they run
        # single-connection. File databases get WAL: one connection per reader
        # thread, writes serialized on the primary connection.
        self.shared = db_path in (":memory:", "")
        self.clusterer = clusterer or ExactClusterer()
        self.shared_lock = threading.RLock()
        self.write_lock = threading.RLock()
        self.registry_lock = threading.Lock()
        self.connections: list[sqlite3.Connection] = []
        self.readers = threading.local()
        self.vec_dim: int | None = None
        self.conn = self.open_connection(load_extension=False)
        self.vec_enabled = self.load_vec(self.conn)
        self.fts_enabled = self.init_schema()

    def open_connection(self, *, load_extension: bool) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.db_path, check_same_thread=False, uri=self.db_path.startswith("file:")
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        if load_extension and self.vec_enabled:
            self.load_vec(conn)
        with self.registry_lock:
            self.connections.append(conn)
        return conn

    def reader(self) -> sqlite3.Connection:
        conn = getattr(self.readers, "conn", None)
        if conn is None:
            conn = self.open_connection(load_extension=True)
            self.readers.conn = conn
        return conn

    @contextmanager
    def reading(self):
        if self.shared:
            with self.shared_lock:
                yield self.conn
        else:
            yield self.reader()

    @contextmanager
    def writing(self):
        lock = self.shared_lock if self.shared else self.write_lock
        with lock:
            try:
                yield self.conn
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def close(self) -> None:
        with self.registry_lock:
            for conn in self.connections:
                conn.close()
            self.connections.clear()

    def init_schema(self) -> bool:
        # Runs at construction, before other threads can touch the store.
        run_migrations(self.conn)
        fts_ok = self.init_fts()
        if self.vec_enabled:
            self.vec_dim = self.detect_vec_dim()
            if self.vec_dim is None:
                self.backfill_vec()
        self.conn.commit()
        return fts_ok

    def init_fts(self) -> bool:
        try:
            self.conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS experiences_fts "
                "USING fts5(id UNINDEXED, task, action)"
            )
            return True
        except sqlite3.OperationalError:
            return False

    def load_vec(self, conn: sqlite3.Connection) -> bool:
        if sqlite_vec is None:
            return False
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            return True
        except (AttributeError, sqlite3.OperationalError):
            return False

    def detect_vec_dim(self) -> int | None:
        row = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'vec_experiences'"
        ).fetchone()
        if not row:
            return None
        match = VEC_DIM_PATTERN.search(row["sql"])
        return int(match.group(1)) if match else None

    def ensure_vec_table(self, dim: int) -> None:
        # Created lazily: the dimension isn't known until the first embedding.
        if self.vec_dim is not None:
            return
        self.conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS vec_experiences "
            f"USING vec0(experience_id TEXT PRIMARY KEY, embedding float[{dim}] "
            "distance_metric=cosine)"
        )
        self.vec_dim = dim

    def backfill_vec(self) -> None:
        # Rows embedded before the extension was installed exist only as JSON.
        rows = self.conn.execute(
            "SELECT id, embedding FROM experiences WHERE embedding IS NOT NULL"
        ).fetchall()
        for row in rows:
            vec = json.loads(row["embedding"])
            if not vec:
                continue
            self.ensure_vec_table(len(vec))
            if len(vec) != self.vec_dim:
                continue
            self.conn.execute(
                "INSERT INTO vec_experiences(experience_id, embedding) VALUES (?, ?)",
                (row["id"], sqlite_vec.serialize_float32([float(x) for x in vec])),
            )

    def add(self, exp: Experience) -> None:
        with self.writing() as conn:
            action_norm = self.clusterer.key(exp.action, self.known_clusters)
            conn.execute(
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
            if self.fts_enabled:
                # Delete first: a re-record under the same id would otherwise
                # leave a duplicate FTS row and skew results.
                conn.execute("DELETE FROM experiences_fts WHERE id = ?", (exp.id,))
                conn.execute(
                    "INSERT INTO experiences_fts (id, task, action) VALUES (?, ?, ?)",
                    (exp.id, exp.task, exp.action),
                )
            if self.vec_enabled and exp.embedding is not None:
                self.ensure_vec_table(len(exp.embedding))
                if len(exp.embedding) == self.vec_dim:
                    conn.execute("DELETE FROM vec_experiences WHERE experience_id = ?", (exp.id,))
                    conn.execute(
                        "INSERT INTO vec_experiences(experience_id, embedding) VALUES (?, ?)",
                        (exp.id, sqlite_vec.serialize_float32([float(x) for x in exp.embedding])),
                    )

    def delete(self, experience_id: str) -> bool:
        with self.writing() as conn:
            cur = conn.execute("DELETE FROM experiences WHERE id = ?", (experience_id,))
            if self.fts_enabled:
                conn.execute("DELETE FROM experiences_fts WHERE id = ?", (experience_id,))
            if self.vec_enabled and self.vec_dim is not None:
                conn.execute("DELETE FROM vec_experiences WHERE experience_id = ?", (experience_id,))
            return cur.rowcount > 0

    def set_superseded_by(self, experience_id: str, superseded_by: str | None) -> bool:
        with self.writing() as conn:
            cur = conn.execute(
                "UPDATE experiences SET superseded_by = ? WHERE id = ?",
                (superseded_by, experience_id),
            )
            return cur.rowcount > 0

    def get(self, experience_id: str) -> Experience | None:
        with self.reading() as conn:
            row = conn.execute(
                "SELECT * FROM experiences WHERE id = ?", (experience_id,)
            ).fetchone()
        return self.row_to_experience(row) if row else None

    def recent(self, n: int = 10) -> list[Experience]:
        # created_at is UTC ISO (enforced on the model), so string order is time
        # order; rowid breaks ties between same-instant writes.
        with self.reading() as conn:
            rows = conn.execute(
                "SELECT * FROM experiences ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (n,),
            ).fetchall()
        return [self.row_to_experience(r) for r in rows]

    def count(self) -> int:
        with self.reading() as conn:
            return conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]

    def known_clusters(self) -> list[Cluster]:
        rows = self.conn.execute(
            "SELECT action_norm AS key, MIN(action) AS action FROM experiences GROUP BY action_norm"
        ).fetchall()
        return [Cluster(key=r["key"], action=r["action"]) for r in rows]

    def search(
        self,
        query: str,
        k: int | None = 5,
        outcome: str | None = None,
        context: dict | None = None,
        include_superseded: bool = False,
    ) -> list[tuple[Experience, float]]:
        # Only apply LIMIT in SQL when every filter was pushed down, otherwise a
        # Python-side context check could drop rows the limit already cut off.
        filters = context_sql_filters(context)
        fully_pushed = not context or len(filters) == len(context)
        limit = k if k is not None and fully_pushed else None

        if self.fts_enabled:
            scored = self.fts_search(query, outcome, filters, limit, include_superseded)
        else:
            scored = self.fallback_search(query, outcome, filters, include_superseded)

        if context:
            scored = [(e, s) for e, s in scored if context_matches(e.context, context)]
        return scored[:k] if k is not None else scored

    def fts_search(
        self,
        query: str,
        outcome: str | None,
        filters: list[tuple[str, object]],
        limit: int | None,
        include_superseded: bool = False,
    ) -> list[tuple[Experience, float]]:
        terms = query_terms(query)
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in terms)

        where = ["experiences_fts MATCH ?"]
        params: list[object] = [match]
        if not include_superseded:
            where.append("e.superseded_by IS NULL")
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
            "ORDER BY rank"
        )
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        with self.reading() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [(self.row_to_experience(r), bm25_to_score(r["rank"])) for r in rows]

    def fallback_search(
        self,
        query: str,
        outcome: str | None,
        filters: list[tuple[str, object]],
        include_superseded: bool = False,
    ) -> list[tuple[Experience, float]]:
        terms = set(query_terms(query))
        if not terms:
            return []

        where = []
        params: list[object] = []
        if not include_superseded:
            where.append("superseded_by IS NULL")
        if outcome is not None:
            where.append("outcome = ?")
            params.append(outcome)
        for path, value in filters:
            where.append("json_extract(context, ?) = ?")
            params += [path, value]
        sql = "SELECT * FROM experiences"
        if where:
            sql += " WHERE " + " AND ".join(where)

        with self.reading() as conn:
            rows = conn.execute(sql, params).fetchall()
        scored = []
        for row in rows:
            exp = self.row_to_experience(row)
            overlap = len(terms & set(tokenize(exp.text())))
            if overlap:
                scored.append((exp, overlap / len(terms)))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored

    def vector_search(
        self,
        embedding: list[float],
        k: int | None = 5,
        outcome: str | None = None,
        context: dict | None = None,
        include_superseded: bool = False,
    ) -> list[tuple[Experience, float]]:
        if not embedding:
            return []
        if self.vec_enabled and self.vec_dim == len(embedding):
            return self.vector_search_ann(embedding, k, outcome, context, include_superseded)
        return self.vector_search_scan(embedding, k, outcome, context, include_superseded)

    def vector_search_ann(
        self,
        embedding: list[float],
        k: int | None,
        outcome: str | None,
        context: dict | None,
        include_superseded: bool,
    ) -> list[tuple[Experience, float]]:
        filters = context_sql_filters(context)
        has_filter = outcome is not None or bool(context)
        query = sqlite_vec.serialize_float32([float(x) for x in embedding])
        with self.reading() as conn:
            total = conn.execute("SELECT COUNT(*) FROM vec_experiences").fetchone()[0]
            if total == 0:
                return []
            if k is None:
                pool = total
            elif has_filter:
                pool = min(total, max(k, k * KNN_OVERFETCH))
            else:
                pool = min(total, k)
            if pool < 1:
                return []
            neighbours = conn.execute(
                "SELECT experience_id, distance FROM vec_experiences "
                "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (query, pool),
            ).fetchall()
        distances = {row["experience_id"]: row["distance"] for row in neighbours}
        if not distances:
            return []

        ids = list(distances)
        where = [f"id IN ({','.join('?' * len(ids))})"]
        params: list[object] = list(ids)
        if not include_superseded:
            where.append("superseded_by IS NULL")
        if outcome is not None:
            where.append("outcome = ?")
            params.append(outcome)
        for path, value in filters:
            where.append("json_extract(context, ?) = ?")
            params += [path, value]
        sql = "SELECT * FROM experiences WHERE " + " AND ".join(where)
        with self.reading() as conn:
            rows = conn.execute(sql, params).fetchall()

        scored = []
        for row in rows:
            exp = self.row_to_experience(row)
            if context and not context_matches(exp.context, context):
                continue
            sim = 1.0 - distances[exp.id]
            if sim > 0:
                scored.append((exp, sim))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k] if k is not None else scored

    def vector_search_scan(
        self,
        embedding: list[float],
        k: int | None,
        outcome: str | None,
        context: dict | None,
        include_superseded: bool,
    ) -> list[tuple[Experience, float]]:
        filters = context_sql_filters(context)
        where = ["embedding IS NOT NULL"]
        params: list[object] = []
        if not include_superseded:
            where.append("superseded_by IS NULL")
        if outcome is not None:
            where.append("outcome = ?")
            params.append(outcome)
        for path, value in filters:
            where.append("json_extract(context, ?) = ?")
            params += [path, value]
        sql = "SELECT * FROM experiences WHERE " + " AND ".join(where)

        with self.reading() as conn:
            rows = conn.execute(sql, params).fetchall()
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

    def aggregate_actions(self, query: str, include_superseded: bool = False) -> list[ActionStat]:
        terms = query_terms(query)
        if not terms:
            return []
        live = "" if include_superseded else " AND e.superseded_by IS NULL"

        if self.fts_enabled:
            match = " OR ".join(f'"{t}"' for t in terms)
            with self.reading() as conn:
                rows = conn.execute(
                    "SELECT key, MIN(action) AS action, "
                    "SUM(outcome = 'success') AS success, "
                    "SUM(outcome = 'failure') AS failure, "
                    "SUM(outcome = 'partial') AS partial, "
                    "COUNT(*) AS total, "
                    "SUM(weight * (outcome = 'success')) AS weighted_success, "
                    "SUM(weight * (outcome = 'partial')) AS weighted_partial, "
                    "SUM(weight) AS weighted_total "
                    "FROM ("
                    # s = -bm25 is the relevance magnitude; s/(1+s) squashes it into [0, 1).
                    "  SELECT key, action, outcome, s / (1.0 + s) AS weight FROM ("
                    "    SELECT e.action_norm AS key, e.action AS action, e.outcome AS outcome, "
                    "    max(-bm25(experiences_fts), 0.0) AS s "
                    "    FROM experiences_fts JOIN experiences e ON e.id = experiences_fts.id "
                    f"    WHERE experiences_fts MATCH ?{live} "
                    # LIMIT stops SQLite flattening the subquery into the outer
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

        # No FTS5: group in Python, weighting by the fraction of terms matched.
        wanted = set(terms)
        sql = "SELECT task, action, action_norm, outcome FROM experiences"
        if not include_superseded:
            sql += " WHERE superseded_by IS NULL"
        with self.reading() as conn:
            rows = conn.execute(sql).fetchall()
        groups: dict[str, dict] = {}
        for row in rows:
            matched = wanted & set(tokenize(f"{row['task']}\n{row['action']}"))
            if not matched:
                continue
            weight = len(matched) / len(wanted)
            group = groups.setdefault(
                row["action_norm"],
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
            group["action"] = min(group["action"], row["action"])
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

    def supporting_ids(
        self, query: str, action_key: str, limit: int = 100, include_superseded: bool = False
    ) -> list[str]:
        terms = query_terms(query)
        if not terms:
            return []
        live = "" if include_superseded else " AND e.superseded_by IS NULL"
        if self.fts_enabled:
            match = " OR ".join(f'"{t}"' for t in terms)
            with self.reading() as conn:
                rows = conn.execute(
                    "SELECT e.id FROM experiences_fts JOIN experiences e "
                    "ON e.id = experiences_fts.id "
                    f"WHERE experiences_fts MATCH ? AND e.action_norm = ?{live} LIMIT ?",
                    (match, action_key, limit),
                ).fetchall()
            return [row["id"] for row in rows]

        wanted = set(terms)
        sql = "SELECT id, task, action, action_norm FROM experiences WHERE action_norm = ?"
        if not include_superseded:
            sql += " AND superseded_by IS NULL"
        with self.reading() as conn:
            rows = conn.execute(sql, (action_key,)).fetchall()
        ids = []
        for row in rows:
            if wanted & set(tokenize(f"{row['task']}\n{row['action']}")):
                ids.append(row["id"])
                if len(ids) >= limit:
                    break
        return ids

    def row_to_experience(self, row: sqlite3.Row) -> Experience:
        # Rows were validated on write; skip re-validation so stored data never
        # re-triggers warnings on read.
        return Experience.model_construct(
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
