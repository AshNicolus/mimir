"""Postgres backend, for many agents writing to one shared store at once.

SQLite serializes writers, which is fine on one machine. Postgres takes
concurrent writes from many processes and machines, so a team or a fleet of
agents can share one memory. Install it with ``pip install "mimir-learn[postgres]"``
and pass a ``postgresql://`` URL anywhere a database path is accepted.

Ranking matches the SQLite backend: keyword relevance is BM25 with the same
parameters and IDF floor as SQLite FTS5, computed in SQL over a ``tsvector``
built from Mimir's own tokenizer. Postgres's parser would keep ``core.py`` or
``db.example.com`` as single words, so a query for ``core`` would miss them.
"""

from __future__ import annotations

import re
import threading
from contextlib import contextmanager
from datetime import timezone
from urllib.parse import urlsplit, urlunsplit

from ..clustering import ActionClusterer, Cluster, ExactClusterer
from ..embeddings import cosine_similarity
from ..models import Experience, Outcome
from ..reflect import Reflection
from .base import ActionStat, Storage
from .query import context_matches, context_sql_filters, query_terms, tokenize

try:
    import psycopg
    from psycopg import sql
    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb
except ImportError:  # optional: the postgres extra
    psycopg = None

SCHEMA_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,62}")
K1, B = 1.2, 0.75  # SQLite FTS5's bm25() defaults

# BM25 per matching row, as SQLite FTS5 computes it: IDF floored at 1e-6, and
# corpus statistics (row count, average length, document frequency) taken over
# every row, superseded ones included, exactly as the FTS5 index holds them.
SCORED_CTE = f"""
WITH terms AS (SELECT DISTINCT unnest(%(terms)s::text[]) AS term),
stats AS (
    SELECT count(*)::float8 AS n, coalesce(avg(doc_len), 0)::float8 AS avgdl FROM experiences
),
idf AS (
    SELECT t.term, greatest(ln((st.n - d.df + 0.5) / (d.df + 0.5)), 1e-6) AS w
    FROM terms t CROSS JOIN stats st
    CROSS JOIN LATERAL (
        SELECT count(*)::float8 AS df FROM experiences x
        WHERE x.tsv @@ to_tsquery('simple', t.term)
    ) d
),
scored AS (
    SELECT e.id, (
        SELECT coalesce(sum(
            i.w * tf.c * {K1 + 1}
            / (tf.c + {K1} * ({1 - B} + {B} * e.doc_len / nullif(st.avgdl, 0)))
        ), 0)
        FROM idf i
        CROSS JOIN LATERAL (
            SELECT cardinality(u.positions)::float8 AS c
            FROM unnest(e.tsv) u WHERE u.lexeme = i.term
        ) tf
    ) AS s
    FROM experiences e CROSS JOIN stats st
    WHERE e.tsv @@ to_tsquery('simple', %(match)s){{filters}}
)
"""


def create_schema_v1(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS experiences (
            id            TEXT PRIMARY KEY,
            seq           BIGSERIAL NOT NULL,
            task          TEXT NOT NULL,
            action        TEXT NOT NULL,
            action_norm   TEXT NOT NULL,
            outcome       TEXT NOT NULL,
            score         DOUBLE PRECISION NOT NULL,
            context       JSONB NOT NULL,
            embedding     DOUBLE PRECISION[],
            created_at    TIMESTAMPTZ NOT NULL,
            superseded_by TEXT,
            tsv           TSVECTOR NOT NULL,
            doc_len       INTEGER NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_experiences_tsv ON experiences USING GIN (tsv)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_experiences_created ON experiences(created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_experiences_action ON experiences(action_norm)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS reflections (
            id                         TEXT PRIMARY KEY,
            seq                        BIGSERIAL NOT NULL,
            summary                    TEXT NOT NULL,
            pattern                    TEXT NOT NULL,
            supporting_experience_ids  JSONB NOT NULL,
            created_at                 TIMESTAMPTZ NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_reflections_created ON reflections(created_at)")


MIGRATIONS = [create_schema_v1]


def redact(dsn: str) -> str:
    """The DSN with any password masked: it is shown to MCP clients and in logs."""
    parts = urlsplit(dsn)
    if parts.scheme in ("postgresql", "postgres") and parts.password:
        netloc = parts.netloc.replace(f":{parts.password}@", ":***@", 1)
        dsn = urlunsplit(parts._replace(netloc=netloc))
    return re.sub(r"(password\s*=\s*)('[^']*'|[^\s&]+)", r"\1***", dsn)


def is_postgres_url(db_path: str) -> bool:
    return db_path.startswith(("postgresql://", "postgres://"))


class PostgresStorage(Storage):
    def __init__(
        self,
        dsn: str,
        *,
        schema: str = "public",
        clusterer: ActionClusterer | None = None,
    ) -> None:
        if psycopg is None:
            raise ImportError('PostgresStorage needs psycopg: pip install "mimir-learn[postgres]"')
        if not SCHEMA_NAME.fullmatch(schema) or schema.lower().startswith("pg_"):
            raise ValueError(
                f"schema must be a plain SQL identifier not starting with pg_, got {schema!r}"
            )
        self.dsn = dsn
        self.db_path = redact(dsn)  # what memory_stats and errors may show
        self.schema = schema
        self.clusterer = clusterer or ExactClusterer()
        self.registry_lock = threading.Lock()
        self.cluster_lock = threading.RLock()
        self.connections: list = []
        self.local = threading.local()
        # shortcut: the cluster cache is per process, so a cluster another
        # process creates is unseen here until a delete() resets the cache. Only
        # matters for EmbeddingClusterer; move clusters into a table if it bites.
        self.clusters: dict[str, str] | None = None
        self.migrate()

    def connect(self):
        conn = psycopg.connect(self.dsn, autocommit=True, row_factory=dict_row)
        conn.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(self.schema)))
        with self.registry_lock:
            self.connections.append(conn)
        return conn

    def migrate(self) -> None:
        # Many agents may start at once; the advisory lock lets one migrate
        # while the rest wait, instead of racing on CREATE TABLE.
        conn = psycopg.connect(self.dsn, autocommit=True, row_factory=dict_row)
        try:
            with conn.transaction():
                conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtext(%s))", (f"mimir.schema.{self.schema}",)
                )
                schema = sql.Identifier(self.schema)
                conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(schema))
                conn.execute(sql.SQL("SET LOCAL search_path TO {}").format(schema))
                conn.execute("CREATE TABLE IF NOT EXISTS mimir_schema (version INTEGER NOT NULL)")
                row = conn.execute("SELECT max(version) AS v FROM mimir_schema").fetchone()
                version = row["v"] or 0
                for target, step in enumerate(MIGRATIONS, start=1):
                    if version < target:
                        step(conn)
                        conn.execute("INSERT INTO mimir_schema (version) VALUES (%s)", (target,))
        finally:
            conn.close()

    @contextmanager
    def reading(self):
        conn = getattr(self.local, "conn", None)
        if conn is None or conn.closed:
            conn = self.local.conn = self.connect()
        yield conn

    @contextmanager
    def writing(self):
        with self.reading() as conn, conn.transaction():
            yield conn

    def close(self) -> None:
        with self.registry_lock:
            for conn in self.connections:
                conn.close()
            self.connections.clear()

    def add(self, exp: Experience) -> None:
        with self.cluster_lock:
            action_norm = self.clusterer.key(exp.action, self.known_clusters)
            tokens = tokenize(exp.text())
            with self.writing() as conn:
                self.update_clusters(conn, exp, action_norm)
                conn.execute(
                    "INSERT INTO experiences (id, task, action, action_norm, outcome, score, "
                    "context, embedding, created_at, superseded_by, tsv, doc_len) "
                    "VALUES (%(id)s, %(task)s, %(action)s, %(norm)s, %(outcome)s, %(score)s, "
                    "%(context)s, %(embedding)s, %(created)s, %(superseded)s, "
                    "to_tsvector('simple', %(doc)s), %(doc_len)s) "
                    # Replace the whole row, as SQLite's INSERT OR REPLACE does,
                    # and move it to the end of recent()'s order.
                    "ON CONFLICT (id) DO UPDATE SET task = EXCLUDED.task, "
                    "action = EXCLUDED.action, action_norm = EXCLUDED.action_norm, "
                    "outcome = EXCLUDED.outcome, score = EXCLUDED.score, "
                    "context = EXCLUDED.context, embedding = EXCLUDED.embedding, "
                    "created_at = EXCLUDED.created_at, superseded_by = EXCLUDED.superseded_by, "
                    "tsv = EXCLUDED.tsv, doc_len = EXCLUDED.doc_len, seq = DEFAULT",
                    {
                        "id": exp.id,
                        "task": exp.task,
                        "action": exp.action,
                        "norm": action_norm,
                        "outcome": exp.outcome.value,
                        "score": exp.score,
                        "context": Jsonb(exp.context),
                        "embedding": exp.embedding,
                        "created": exp.created_at,
                        "superseded": exp.superseded_by,
                        "doc": " ".join(tokens),
                        "doc_len": len(tokens),
                    },
                )

    def delete(self, experience_id: str) -> bool:
        with self.cluster_lock, self.writing() as conn:
            self.clusters = None  # may have removed a cluster's last member
            cur = conn.execute("DELETE FROM experiences WHERE id = %s", (experience_id,))
            return cur.rowcount > 0

    def set_superseded_by(self, experience_id: str, superseded_by: str | None) -> bool:
        with self.writing() as conn:
            cur = conn.execute(
                "UPDATE experiences SET superseded_by = %s WHERE id = %s",
                (superseded_by, experience_id),
            )
            return cur.rowcount > 0

    def get(self, experience_id: str) -> Experience | None:
        with self.reading() as conn:
            row = conn.execute(
                "SELECT * FROM experiences WHERE id = %s", (experience_id,)
            ).fetchone()
        return self.row_to_experience(row) if row else None

    def recent(self, n: int = 10) -> list[Experience]:
        with self.reading() as conn:
            rows = conn.execute(
                "SELECT * FROM experiences ORDER BY created_at DESC, seq DESC LIMIT %s", (n,)
            ).fetchall()
        return [self.row_to_experience(r) for r in rows]

    def count(self) -> int:
        with self.reading() as conn:
            return conn.execute("SELECT count(*) AS n FROM experiences").fetchone()["n"]

    def add_reflection(self, reflection: Reflection) -> None:
        with self.writing() as conn:
            conn.execute(
                "INSERT INTO reflections (id, summary, pattern, supporting_experience_ids, "
                "created_at) VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET summary = EXCLUDED.summary, "
                "pattern = EXCLUDED.pattern, "
                "supporting_experience_ids = EXCLUDED.supporting_experience_ids, "
                "created_at = EXCLUDED.created_at, seq = DEFAULT",
                (
                    reflection.id,
                    reflection.summary,
                    reflection.pattern,
                    Jsonb(reflection.supporting_experience_ids),
                    reflection.created_at,
                ),
            )

    def get_reflection(self, reflection_id: str) -> Reflection | None:
        with self.reading() as conn:
            row = conn.execute(
                "SELECT * FROM reflections WHERE id = %s", (reflection_id,)
            ).fetchone()
        return self.row_to_reflection(row) if row else None

    def recent_reflections(self, n: int = 10) -> list[Reflection]:
        with self.reading() as conn:
            rows = conn.execute(
                "SELECT * FROM reflections ORDER BY created_at DESC, seq DESC LIMIT %s", (n,)
            ).fetchall()
        return [self.row_to_reflection(r) for r in rows]

    def known_clusters(self) -> list[Cluster]:
        with self.cluster_lock:
            if self.clusters is None:
                with self.reading() as conn:
                    rows = conn.execute(
                        'SELECT action_norm AS key, MIN(action COLLATE "C") AS action '
                        "FROM experiences GROUP BY action_norm"
                    ).fetchall()
                self.clusters = {r["key"]: r["action"] for r in rows}
            return [Cluster(key=k, action=a) for k, a in self.clusters.items()]

    def update_clusters(self, conn, exp: Experience, action_norm: str) -> None:
        prior = conn.execute(
            "SELECT action_norm FROM experiences WHERE id = %s", (exp.id,)
        ).fetchone()
        if prior is not None and prior["action_norm"] != action_norm:
            self.clusters = None  # a re-record moved clusters; rebuild lazily
        elif self.clusters is not None:
            current = self.clusters.get(action_norm)
            self.clusters[action_norm] = exp.action if current is None else min(current, exp.action)

    def filter_sql(
        self, outcome: str | None, context: dict | None, include_superseded: bool
    ) -> tuple[str, dict]:
        """WHERE fragments (each starting with AND) over alias ``e``, plus params."""
        parts, params = [], {}
        if not include_superseded:
            parts.append("e.superseded_by IS NULL")
        if outcome is not None:
            parts.append("e.outcome = %(outcome)s")
            params["outcome"] = outcome
        for i, (path, value) in enumerate(context_sql_filters(context)):
            parts.append(f"e.context -> %(ck{i})s::text = %(cv{i})s")
            params[f"ck{i}"] = path.removeprefix("$.")
            params[f"cv{i}"] = Jsonb(value)
        return "".join(f" AND {p}" for p in parts), params

    def search(
        self,
        query: str,
        k: int | None = 5,
        outcome: str | None = None,
        context: dict | None = None,
        include_superseded: bool = False,
    ) -> list[tuple[Experience, float]]:
        terms = query_terms(query)
        if not terms:
            return []
        # LIMIT in SQL only when every context filter was pushed down, or a
        # Python-side check could drop rows the limit already kept out.
        fully_pushed = not context or len(context_sql_filters(context)) == len(context)
        filters, params = self.filter_sql(outcome, context, include_superseded)
        params.update(terms=terms, match=" | ".join(terms))
        query_sql = (
            SCORED_CTE.replace("{filters}", filters)
            + "SELECT e.*, sc.s FROM scored sc JOIN experiences e ON e.id = sc.id "
            "ORDER BY sc.s DESC, e.seq"
        )
        if k is not None and fully_pushed:
            query_sql += " LIMIT %(limit)s"
            params["limit"] = k
        with self.reading() as conn:
            rows = conn.execute(query_sql, params).fetchall()
        scored = [(self.row_to_experience(r), r["s"] / (1.0 + r["s"])) for r in rows]
        if context:
            scored = [(e, s) for e, s in scored if context_matches(e.context, context)]
        return scored[:k] if k is not None else scored

    def vector_search(
        self,
        embedding: list[float],
        k: int | None = 5,
        outcome: str | None = None,
        context: dict | None = None,
        include_superseded: bool = False,
    ) -> list[tuple[Experience, float]]:
        # shortcut: a full scan with cosine in Python, like SQLite without
        # sqlite-vec. Fine for thousands of rows; add a pgvector index for more.
        if not embedding:
            return []
        filters, params = self.filter_sql(outcome, context, include_superseded)
        with self.reading() as conn:
            rows = conn.execute(
                f"SELECT * FROM experiences e WHERE e.embedding IS NOT NULL{filters}", params
            ).fetchall()
        scored = []
        for row in rows:
            exp = self.row_to_experience(row)
            if context and not context_matches(exp.context, context):
                continue
            sim = cosine_similarity(embedding, exp.embedding)
            if sim > 0:
                scored.append((exp, sim))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k] if k is not None else scored

    def aggregate_actions(
        self, query: str, include_superseded: bool = False, half_life_days: float | None = None
    ) -> list[ActionStat]:
        terms = query_terms(query)
        if not terms:
            return []
        filters, params = self.filter_sql(None, None, include_superseded)
        params.update(terms=terms, match=" | ".join(terms))
        weight = "sc.s / (1.0 + sc.s)"
        if half_life_days and half_life_days > 0:
            weight += (
                " * power(2.0, -greatest(extract(epoch FROM now() - e.created_at), 0)"
                " / 86400.0 / %(half_life)s)"
            )
            params["half_life"] = half_life_days
        with self.reading() as conn:
            rows = conn.execute(
                SCORED_CTE.replace("{filters}", filters)
                + 'SELECT e.action_norm AS key, MIN(e.action COLLATE "C") AS action, '
                "count(*) FILTER (WHERE e.outcome = 'success') AS success, "
                "count(*) FILTER (WHERE e.outcome = 'failure') AS failure, "
                "count(*) FILTER (WHERE e.outcome = 'partial') AS partial, "
                f"count(*) AS total, sum({weight}) AS weighted_total "
                "FROM scored sc JOIN experiences e ON e.id = sc.id GROUP BY e.action_norm",
                params,
            ).fetchall()
        return [
            ActionStat(
                action=r["action"],
                key=r["key"],
                success=r["success"],
                failure=r["failure"],
                partial=r["partial"],
                total=r["total"],
                weighted_total=float(r["weighted_total"]),
            )
            for r in rows
        ]

    def supporting_ids(
        self, query: str, action_key: str, limit: int = 100, include_superseded: bool = False
    ) -> list[str]:
        terms = query_terms(query)
        if not terms:
            return []
        filters, params = self.filter_sql(None, None, include_superseded)
        params.update(match=" | ".join(terms), key=action_key, limit=limit)
        with self.reading() as conn:
            rows = conn.execute(
                "SELECT e.id FROM experiences e "
                "WHERE e.tsv @@ to_tsquery('simple', %(match)s) AND e.action_norm = %(key)s"
                f"{filters} ORDER BY e.seq LIMIT %(limit)s",
                params,
            ).fetchall()
        return [r["id"] for r in rows]

    def row_to_experience(self, row: dict) -> Experience:
        # Rows were validated on write; skip re-validation on read.
        return Experience.model_construct(
            id=row["id"],
            task=row["task"],
            action=row["action"],
            outcome=Outcome(row["outcome"]),
            score=row["score"],
            context=row["context"],
            embedding=list(row["embedding"]) if row["embedding"] is not None else None,
            created_at=row["created_at"].astimezone(timezone.utc),
            superseded_by=row["superseded_by"],
        )

    def row_to_reflection(self, row: dict) -> Reflection:
        return Reflection.model_construct(
            id=row["id"],
            summary=row["summary"],
            pattern=row["pattern"],
            supporting_experience_ids=row["supporting_experience_ids"],
            created_at=row["created_at"].astimezone(timezone.utc),
        )
