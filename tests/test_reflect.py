"""reflect(): synthesizing a pattern across a set of experiences."""

from mimir import CallableReflector
from mimir.reflect import ReflectionDraft


def counting_reflector(calls):
    def fn(experiences):
        calls.append(experiences)
        actions = sorted({e.action for e in experiences})
        return ReflectionDraft(
            summary=f"{len(experiences)} experiences", pattern=f"actions: {', '.join(actions)}"
        )

    return CallableReflector(fn)


def test_reflect_hands_recalled_experiences_to_the_reflector(memory):
    for _ in range(3):
        memory.record("fix login latency", "add a redis cache", outcome="success")
    memory.record("fix login latency", "raise thread count", outcome="failure")

    calls = []
    reflection = memory.reflect("login latency", reflector=counting_reflector(calls))

    assert reflection is not None
    assert len(calls) == 1
    assert len(calls[0]) == 4
    assert reflection.summary == "4 experiences"
    assert "add a redis cache" in reflection.pattern


def test_reflect_returns_none_without_any_matching_experience(memory):
    calls = []
    reflection = memory.reflect("nothing recorded about this", reflector=counting_reflector(calls))
    assert reflection is None
    assert calls == []  # never even called the reflector


def test_reflector_can_abstain(memory):
    memory.record("fix login latency", "add a redis cache", outcome="success")
    reflection = memory.reflect("login latency", reflector=CallableReflector(lambda exps: None))
    assert reflection is None


def test_supporting_ids_match_what_was_recalled(memory):
    a = memory.record("fix login latency", "add a redis cache", outcome="success")
    b = memory.record("fix login latency", "add a redis cache", outcome="success")

    calls = []
    reflection = memory.reflect("login latency", reflector=counting_reflector(calls))
    assert set(reflection.supporting_experience_ids) == {a.id, b.id}


def test_reflecting_the_same_set_again_replaces_not_duplicates(memory):
    memory.record("fix login latency", "add a redis cache", outcome="success")

    first = memory.reflect("login latency", reflector=counting_reflector([]))
    second = memory.reflect("login latency", reflector=counting_reflector([]))

    assert first.id == second.id
    assert len(memory.recent_reflections(10)) == 1


def test_a_new_supporting_experience_produces_a_different_reflection(memory):
    memory.record("fix login latency", "add a redis cache", outcome="success")
    first = memory.reflect("login latency", reflector=counting_reflector([]))

    memory.record("fix login latency", "add a redis cache", outcome="success")
    second = memory.reflect("login latency", reflector=counting_reflector([]))

    assert first.id != second.id
    assert len(memory.recent_reflections(10)) == 2


def test_get_reflection_fetches_by_id(memory):
    memory.record("fix login latency", "add a redis cache", outcome="success")
    reflection = memory.reflect("login latency", reflector=counting_reflector([]))
    fetched = memory.get_reflection(reflection.id)
    assert fetched is not None
    assert fetched.summary == reflection.summary


def test_get_reflection_returns_none_for_unknown_id(memory):
    assert memory.get_reflection("does-not-exist") is None


def test_recent_reflections_orders_newest_first(memory):
    memory.record("fix login latency", "add a redis cache", outcome="success")
    memory.record("checkout is slow", "add caching", outcome="success")

    memory.reflect("login latency", reflector=counting_reflector([]))
    second = memory.reflect("checkout is slow", reflector=counting_reflector([]))

    assert memory.recent_reflections(1)[0].id == second.id


def test_include_superseded_is_passed_through_to_recall(memory):
    old = memory.record("fix login latency", "old approach", outcome="failure")
    memory.record("fix login latency", "new approach", supersedes=old.id)

    calls = []
    memory.reflect("login latency", reflector=counting_reflector(calls))
    assert len(calls[0]) == 1  # superseded row excluded by default

    calls_with_superseded = []
    reflector = counting_reflector(calls_with_superseded)
    memory.reflect("login latency", reflector=reflector, include_superseded=True)
    assert len(calls_with_superseded[0]) == 2


def test_blank_summary_or_pattern_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        ReflectionDraft(summary="   ", pattern="something")
    with pytest.raises(ValueError):
        ReflectionDraft(summary="something", pattern="")


def test_fresh_db_has_the_reflections_table(memory):
    tables = {r["name"] for r in memory.storage.conn.execute("SELECT name FROM sqlite_master")}
    assert "reflections" in tables
