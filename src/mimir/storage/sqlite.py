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
from contextlib import contextmanager
from datetime import datetime

try:
    import sqlite_vec
except ImportError:  # optional: the ANN index is the [vector] extra
    sqlite_vec = None

from ..clustering import ActionClusterer, Cluster, ExactClusterer, normalize_action
from ..embeddings import cosine_similarity
from ..models import Experience, Outcome
from .base import ActionStat, Storage

TOKEN = re.compile(r"[a-z0-9]+")
SIMPLE_KEY = re.compile(r"[A-Za-z0-9_]+")
VEC_DIM = re.compile(r"float\[(\d+)\]")
# Neighbours to over-fetch before filtering, since KNN can't filter inline.
VEC_FILTER_OVERFETCH = 8

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


# Forward-only schema migrations, applied in order. Each must be idempotent:
# released databases predate user_version and all report 0, so every step has
# to be safe to re-run on a database that already has the modern shape.
def migrate_base_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
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
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_experiences_created ON experiences(created_at)"
    )


def migrate_action_norm(conn: sqlite3.Connection) -> None:
    # Add and backfill action_norm for databases created before the column.
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(experiences)")}
    if "action_norm" in columns:
        return
    conn.execute("ALTER TABLE experiences ADD COLUMN action_norm TEXT")
    stale = conn.execute("SELECT id, action FROM experiences").fetchall()
    conn.executemany(
        "UPDATE experiences SET action_norm = ? WHERE id = ?",
        [(normalize_action(r["action"]), r["id"]) for r in stale],
    )


def migrate_drop_outcome_index(conn: sqlite3.Connection) -> None:
    # Unused: FTS recall joins by primary key and filters outcome as a residual.
    conn.execute("DROP INDEX IF EXISTS idx_experiences_outcome")


MIGRATIONS = [
    migrate_base_schema,
    migrate_action_norm,
    migrate_drop_outcome_index,
]
SCHEMA_VERSION = len(MIGRATIONS)


class SQLiteStorage(Storage):
    def __init__(
        self, db_path: str = ":memory:", *, clusterer: ActionClusterer | None = None
    ) -> None:
        if db_path not in (":memory:", "") and not db_path.startswith("file:"):
            parent = os.path.dirname(os.path.abspath(db_path))
            os.makedirs(parent, exist_ok=True)
        self._db_path = db_path
        # In-memory databases can't share data across connections and don't get
        # WAL, so they stay single-connection. File databases use WAL: readers
        # get their own connection and run concurrently, writers are serialized.
        self._shared = db_path in (":memory:", "")
        self._clusterer = clusterer or ExactClusterer()
        self._shared_lock = threading.RLock()  # guards the single shared conn
        self._write_lock = threading.RLock()  # serializes writers (file mode)
        self._conn_lock = threading.Lock()  # guards the connection registry
        self._connections: list[sqlite3.Connection] = []
        self._readers = threading.local()
        self._vec_dim: int | None = None  # set once the ANN table exists
        # The primary connection runs migrations and serves as the writer.
        self._conn = self.open_connection(load_extension=False)
        self._vec = self.load_vec(self._conn)
        self._fts = self.init_schema()

    def open_connection(self, *, load_extension: bool) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self._db_path, check_same_thread=False, uri=self._db_path.startswith("file:")
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")  # wait out a busy writer, don't fail
        if load_extension and self._vec:
            self.load_vec(conn)
        with self._conn_lock:
            self._connections.append(conn)
        return conn

    def reader(self) -> sqlite3.Connection:
        """The calling thread's read connection, opened on first use. File mode
        only; shared mode reads go through the single connection."""
        conn = getattr(self._readers, "conn", None)
        if conn is None:
            conn = self.open_connection(load_extension=True)
            self._readers.conn = conn
        return conn

    @contextmanager
    def reading(self):
        # File mode: lock-free, each thread on its own connection (WAL readers
        # don't block each other). Shared mode: serialize on the one connection.
        if self._shared:
            with self._shared_lock:
                yield self._conn
        else:
            yield self.reader()

    @contextmanager
    def writing(self):
        # One writer at a time, committing on success and rolling back on error.
        lock = self._shared_lock if self._shared else self._write_lock
        with lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def init_schema(self) -> bool:
        # Runs once at construction on the primary connection, before any other
        # thread can touch the store, so it needs no locking.
        self.run_migrations()
        fts_ok = self.init_fts()
        if self._vec:
            # Reuse a prior index, else backfill one from existing rows.
            self._vec_dim = self.detect_vec_dim()
            if self._vec_dim is None:
                self.backfill_vec()
        self._conn.commit()
        return fts_ok

    def run_migrations(self) -> None:
        """Apply pending migrations, stamping user_version after each so an
        interrupted upgrade resumes where it left off."""
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        for target, migrate in enumerate(MIGRATIONS, start=1):
            if version >= target:
                continue
            migrate(self._conn)
            self._conn.execute(f"PRAGMA user_version = {target}")
            self._conn.commit()

    def init_fts(self) -> bool:
        # FTS lives outside the version gate: whether it exists depends on the
        # build (FTS5 compiled in), not the schema version.
        try:
            self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS experiences_fts "
                "USING fts5(id UNINDEXED, task, action)"
            )
            return True
        except sqlite3.OperationalError:
            return False

    def load_vec(self, conn: sqlite3.Connection) -> bool:
        """Load sqlite-vec into ``conn`` if installed and loadable; never raise.
        The ANN index is a pure optimization, so failure keeps the cosine fallback."""
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
        """Recover the indexed dimension from an ANN table left by an earlier run."""
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'vec_experiences'"
        ).fetchone()
        if not row:
            return None
        match = VEC_DIM.search(row["sql"])
        return int(match.group(1)) if match else None

    def ensure_vec_table(self, dim: int) -> None:
        # Created lazily: the dimension isn't known until the first embedding.
        if self._vec_dim is not None:
            return
        self._conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS vec_experiences "
            f"USING vec0(experience_id TEXT PRIMARY KEY, embedding float[{dim}] "
            "distance_metric=cosine)"
        )
        self._vec_dim = dim

    def backfill_vec(self) -> None:
        # Build the index from rows embedded before the extension was available.
        rows = self._conn.execute(
            "SELECT id, embedding FROM experiences WHERE embedding IS NOT NULL"
        ).fetchall()
        for row in rows:
            vec = json.loads(row["embedding"])
            if not vec:
                continue
            self.ensure_vec_table(len(vec))
            if len(vec) != self._vec_dim:
                continue  # mixed dimensions: keep the first, skip the rest
            self._conn.execute(
                "INSERT INTO vec_experiences(experience_id, embedding) VALUES (?, ?)",
                (row["id"], sqlite_vec.serialize_float32([float(x) for x in vec])),
            )

    def known_clusters(self) -> list[Cluster]:
        # Representative phrasing per cluster, for a clusterer to merge against.
        rows = self._conn.execute(
            "SELECT action_norm AS key, MIN(action) AS action FROM experiences GROUP BY action_norm"
        ).fetchall()
        return [Cluster(key=r["key"], action=r["action"]) for r in rows]

    def add(self, exp: Experience) -> None:
        with self.writing() as conn:
            action_norm = self._clusterer.key(exp.action, self.known_clusters)
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
            if self._fts:
                # Keep FTS in sync on re-record: clear any stale row for this id
                # first, otherwise INSERT OR REPLACE above would leave a duplicate
                # FTS entry and skew search results.
                conn.execute("DELETE FROM experiences_fts WHERE id = ?", (exp.id,))
                conn.execute(
                    "INSERT INTO experiences_fts (id, task, action) VALUES (?, ?, ?)",
                    (exp.id, exp.task, exp.action),
                )
            if self._vec and exp.embedding is not None:
                self.ensure_vec_table(len(exp.embedding))
                if len(exp.embedding) == self._vec_dim:
                    # Clear any prior row so a re-record replaces, not duplicates.
                    conn.execute("DELETE FROM vec_experiences WHERE experience_id = ?", (exp.id,))
                    conn.execute(
                        "INSERT INTO vec_experiences(experience_id, embedding) VALUES (?, ?)",
                        (exp.id, sqlite_vec.serialize_float32([float(x) for x in exp.embedding])),
                    )

    def delete(self, experience_id: str) -> bool:
        with self.writing() as conn:
            cur = conn.execute("DELETE FROM experiences WHERE id = ?", (experience_id,))
            if self._fts:
                conn.execute("DELETE FROM experiences_fts WHERE id = ?", (experience_id,))
            if self._vec and self._vec_dim is not None:
                conn.execute("DELETE FROM vec_experiences WHERE experience_id = ?", (experience_id,))
            return cur.rowcount > 0

    def get(self, experience_id: str) -> Experience | None:
        with self.reading() as conn:
            row = conn.execute(
                "SELECT * FROM experiences WHERE id = ?", (experience_id,)
            ).fetchone()
        return self.row_to_experience(row) if row else None

    def recent(self, n: int = 10) -> list[Experience]:
        with self.reading() as conn:
            # Tie-break on rowid (monotonic insertion order) because created_at
            # resolution can collide for records written in the same instant.
            rows = conn.execute(
                "SELECT * FROM experiences ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (n,),
            ).fetchall()
        return [self.row_to_experience(r) for r in rows]

    def count(self) -> int:
        with self.reading() as conn:
            return conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]

    def aggregate_actions(self, query: str) -> list[ActionStat]:
        terms = query_terms(query)
        if not terms:
            return []

        if self._fts:
            # Group and count in SQL: one row per action, not one per row. The
            # inner query scores each match with bm25 so the outer query can sum
            # relevance per action alongside the raw counts.
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
        with self.reading() as conn:
            rows = conn.execute(
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
            with self.reading() as conn:
                rows = conn.execute(
                    "SELECT e.id FROM experiences_fts JOIN experiences e "
                    "ON e.id = experiences_fts.id "
                    "WHERE experiences_fts MATCH ? AND e.action_norm = ? LIMIT ?",
                    (match, action_key, limit),
                ).fetchall()
            return [row["id"] for row in rows]

        wanted = set(terms)
        with self.reading() as conn:
            rows = conn.execute(
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
        if self._vec and self._vec_dim == len(embedding):
            return self.vector_search_ann(embedding, k, outcome, context)
        return self.vector_search_scan(embedding, k, outcome, context)

    def vector_search_ann(
        self,
        embedding: list[float],
        k: int | None,
        outcome: str | None,
        context: dict | None,
    ) -> list[tuple[Experience, float]]:
        # KNN can't filter inside the query, so over-fetch and filter after.
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
                pool = min(total, max(k, k * VEC_FILTER_OVERFETCH))
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
            sim = 1.0 - distances[exp.id]  # cosine distance back to similarity
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
    ) -> list[tuple[Experience, float]]:
        # Dependency-free fallback: O(N) cosine in Python over embedded rows.
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

        with self.reading() as conn:
            rows = conn.execute(sql, params).fetchall()
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

        with self.reading() as conn:
            rows = conn.execute(sql, params).fetchall()
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
        # Close every connection handed out, readers in other threads included.
        with self._conn_lock:
            for conn in self._connections:
                conn.close()
            self._connections.clear()


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
