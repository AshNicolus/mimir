"""build_report(): real counts over stored, bucketed episodes."""

from datetime import datetime, timezone

from mimir import Outcome
from mimir.ingest.models import Bucketed, Episode, Turn
from mimir.ingest.report import build_report
from mimir.ingest.store import store_episode

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def seed(memory, task, action, category, outcome, source="claude-code"):
    turns = [Turn(role="user", text=task, timestamp=NOW)]
    ep = Episode(source=source, session_id=task, turns=turns)
    bucketed = Bucketed(category=category, task=task, action=action, outcome=outcome)
    store_episode(memory, ep, bucketed, extractor_name="fixed")


def test_empty_store_says_so(memory):
    assert build_report(memory) == "No experiences stored yet."


def test_report_groups_by_category(memory):
    seed(memory, "fix the login bug", "patched auth", "fix", Outcome.SUCCESS)
    seed(memory, "add export button", "added button", "feature", Outcome.SUCCESS)
    report = build_report(memory)
    assert "[fix]" in report
    assert "[feature]" in report


def test_report_counts_outcomes_correctly(memory):
    seed(memory, "fix a", "action a", "fix", Outcome.SUCCESS)
    seed(memory, "fix b", "action b", "fix", Outcome.FAILURE)
    report = build_report(memory)
    fix_line = next(line for line in report.splitlines() if line.startswith("[fix]"))
    assert "1 success" in fix_line and "1 failure" in fix_line


def test_report_surfaces_the_most_common_action(memory):
    seed(memory, "fix a", "add a redis cache", "fix", Outcome.SUCCESS)
    seed(memory, "fix b", "add a redis cache", "fix", Outcome.SUCCESS)
    seed(memory, "fix c", "something else", "fix", Outcome.SUCCESS)
    report = build_report(memory)
    assert "'add a redis cache' (2x)" in report


def test_uncategorized_experiences_get_their_own_bucket(memory):
    memory.record("a manually recorded task", "an action", outcome="success")
    report = build_report(memory)
    assert "[uncategorized]" in report
