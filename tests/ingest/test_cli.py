"""CLI wiring. Every test points --db at a tmp path; never the real default,
so a test run can never touch the developer's actual Mimir store.
"""

import argparse
from datetime import datetime, timedelta, timezone

from mimir import Mimir, mcp_server
from mimir.ingest import cli
from mimir.ingest.models import Turn

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def fake_sessions():
    return {
        "session-1": [
            Turn(role="user", text="add a login page", timestamp=BASE),
            Turn(role="assistant", text="added it", timestamp=BASE + timedelta(minutes=1)),
        ]
    }


def run_args(db, **overrides):
    defaults = dict(db=db, source="claude-code", model=None, limit=None, dry_run=False)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_build_extractor_is_none_without_a_model():
    assert cli.build_extractor(None) is None


def test_build_extractor_wraps_ollama_when_named():
    extractor = cli.build_extractor("qwen2.5:1.5b")
    assert extractor is not None and extractor.model == "qwen2.5:1.5b"


def test_default_db_path_honors_the_env_var(monkeypatch):
    monkeypatch.setenv("MIMIR_DB_PATH", "/tmp/somewhere/custom.db")
    assert cli.default_db_path() == "/tmp/somewhere/custom.db"


def test_default_db_path_falls_back_when_unset(monkeypatch):
    monkeypatch.delenv("MIMIR_DB_PATH", raising=False)
    # Same store the MCP server uses, so ingested sessions show up there.
    assert cli.default_db_path() == str(mcp_server.DEFAULT_DB_PATH)


def test_run_stores_episodes_with_the_heuristic_bucketer(tmp_path, monkeypatch):
    monkeypatch.setitem(cli.READERS, "claude-code", fake_sessions)
    db = str(tmp_path / "test.db")
    cli.run(run_args(db))
    memory = Mimir(db)
    try:
        assert memory.count() == 1
    finally:
        memory.close()


def test_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(cli.READERS, "claude-code", fake_sessions)
    db = str(tmp_path / "test.db")
    cli.run(run_args(db, dry_run=True))
    printed = capsys.readouterr().out
    assert not (tmp_path / "test.db").exists()
    assert "login" in printed


def test_report_prints_the_built_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(cli.READERS, "claude-code", fake_sessions)
    db = str(tmp_path / "test.db")
    cli.run(run_args(db))
    cli.report(argparse.Namespace(db=db))
    out = capsys.readouterr().out
    assert "experiences" in out


def test_main_parses_report_subcommand(tmp_path, monkeypatch, capsys):
    memory = Mimir(str(tmp_path / "test.db"))
    memory.record("a task", "an action", outcome="success")
    memory.close()
    monkeypatch.setattr("sys.argv", ["mimir-ingest", "--db", str(tmp_path / "test.db"), "report"])
    cli.main()
    assert "experiences" in capsys.readouterr().out
