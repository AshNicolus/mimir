"""store_episode(): wiring a bucketed episode into Mimir via record_conversation."""

from datetime import datetime, timezone

from mimir import Outcome
from mimir.ingest.models import Bucketed, Episode, Turn
from mimir.ingest.store import episode_to_messages, store_episode

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def episode(*, task_text: str = "add a health check endpoint") -> Episode:
    return Episode(
        source="claude-code",
        session_id="s1",
        turns=[Turn(role="user", text=task_text, timestamp=NOW)],
    )


def test_store_episode_lands_in_the_store(memory):
    bucketed = Bucketed(category="feature", task="add health check", action="added /health route")
    exp = store_episode(memory, episode(), bucketed, extractor_name="fixed")
    assert exp is not None
    assert exp.task == "add health check"
    assert exp.action == "added /health route"
    assert memory.count() == 1


def test_category_source_and_session_land_in_context(memory):
    bucketed = Bucketed(category="feature", task="t", action="a")
    exp = store_episode(memory, episode(), bucketed, extractor_name="fixed")
    assert exp.context["category"] == "feature"
    assert exp.context["agent"] == "claude-code"
    assert exp.context["session_id"] == "s1"


def test_missing_outcome_is_inferred_not_left_blank(memory):
    bucketed = Bucketed(category="fix", task="fix the bug", action="patched it")
    exp = store_episode(memory, episode(), bucketed, extractor_name="fixed")
    assert exp.outcome in (Outcome.SUCCESS, Outcome.FAILURE, Outcome.PARTIAL)


def test_explicit_outcome_is_preserved(memory):
    bucketed = Bucketed(category="fix", task="t", action="a", outcome=Outcome.SUCCESS)
    exp = store_episode(memory, episode(), bucketed, extractor_name="fixed")
    assert exp.outcome is Outcome.SUCCESS


def test_rerunning_on_the_same_episode_replaces_not_duplicates(memory):
    bucketed = Bucketed(category="fix", task="t", action="first pass", outcome=Outcome.PARTIAL)
    first = store_episode(memory, episode(), bucketed, extractor_name="fixed")
    updated = Bucketed(category="fix", task="t", action="second pass", outcome=Outcome.SUCCESS)
    second = store_episode(memory, episode(), updated, extractor_name="fixed")
    assert first.id == second.id
    assert memory.count() == 1
    assert memory.get(first.id).action == "second pass"


def test_episode_to_messages_preserves_role_and_text():
    ep = episode(task_text="hello")
    messages = episode_to_messages(ep)
    assert messages == [{"role": "user", "content": "hello"}]
