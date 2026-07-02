"""Forward-only schema migrations, tracked with PRAGMA user_version.

Every step must be idempotent: databases written before versioning existed
report version 0 regardless of their actual shape.
"""

from __future__ import annotations

import sqlite3

from ..clustering import normalize_action


def create_base_schema(conn: sqlite3.Connection) -> None:
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
    conn.execute("CREATE INDEX IF NOT EXISTS idx_experiences_created ON experiences(created_at)")


def add_action_norm(conn: sqlite3.Connection) -> None:
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(experiences)")}
    if "action_norm" in columns:
        return
    conn.execute("ALTER TABLE experiences ADD COLUMN action_norm TEXT")
    rows = conn.execute("SELECT id, action FROM experiences").fetchall()
    conn.executemany(
        "UPDATE experiences SET action_norm = ? WHERE id = ?",
        [(normalize_action(r["action"]), r["id"]) for r in rows],
    )


def drop_outcome_index(conn: sqlite3.Connection) -> None:
    conn.execute("DROP INDEX IF EXISTS idx_experiences_outcome")


MIGRATIONS = [create_base_schema, add_action_norm, drop_outcome_index]
SCHEMA_VERSION = len(MIGRATIONS)


def run_migrations(conn: sqlite3.Connection) -> None:
    # Stamp user_version after each step so an interrupted upgrade resumes.
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    for target, migrate in enumerate(MIGRATIONS, start=1):
        if version >= target:
            continue
        migrate(conn)
        conn.execute(f"PRAGMA user_version = {target}")
        conn.commit()
