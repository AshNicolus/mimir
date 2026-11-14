"""heuristic_bucket() and infer_outcome(): the always-succeeds fallback."""

from datetime import datetime, timedelta, timezone

from mimir import Outcome
from mimir.ingest.fallback import heuristic_bucket, infer_outcome
from mimir.ingest.models import Episode, ToolCall, Turn

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def episode_of(*texts: str, tool_names: tuple[str, ...] = ()) -> Episode:
    turns = [
        Turn(role="user", text=t, timestamp=BASE + timedelta(minutes=i))
        for i, t in enumerate(texts)
    ]
    if tool_names:
        turns.append(
            Turn(
                role="assistant",
                text="",
                timestamp=BASE + timedelta(minutes=len(texts)),
                tool_calls=[ToolCall(name=n, preview="") for n in tool_names],
            )
        )
    return Episode(source="claude-code", session_id="s1", turns=turns)


def test_fix_keywords_bucket_as_fix():
    assert heuristic_bucket(episode_of("there is a bug in the login flow")).category == "fix"


def test_feature_keywords_bucket_as_feature():
    assert heuristic_bucket(episode_of("add a new export button")).category == "feature"


def test_refactor_keywords_bucket_as_refactor():
    assert heuristic_bucket(episode_of("refactor the auth module")).category == "refactor"


def test_no_signal_falls_back_to_other():
    assert heuristic_bucket(episode_of("what time is it")).category == "other"


def test_task_is_the_first_user_message():
    bucketed = heuristic_bucket(episode_of("fix the flaky test", "some other later text"))
    assert bucketed.task == "fix the flaky test"


def turn(role: str, text: str, minute: int, tools: tuple[str, ...] = ()) -> Turn:
    calls = [ToolCall(name=n, preview="") for n in tools]
    return Turn(role=role, text=text, timestamp=BASE + timedelta(minutes=minute), tool_calls=calls)


def test_action_is_what_the_assistant_said_it_did():
    ep = Episode("claude-code", "s1", [
        turn("user", "fix the login bug", 0),
        turn("assistant", "Fixed the token\n  expiry check.", 1, tools=("Edit",)),
    ])
    assert heuristic_bucket(ep).action == "Fixed the token expiry check."


def test_action_skips_the_pleasantry_after_the_users_reaction():
    ep = Episode("claude-code", "s1", [
        turn("user", "fix the login bug", 0),
        turn("assistant", "Fixed the token expiry check.", 1),
        turn("user", "thanks, works now", 2),
        turn("assistant", "You're welcome!", 3),
    ])
    assert heuristic_bucket(ep).action == "Fixed the token expiry check."


def test_action_without_assistant_text_is_scoped_to_the_task():
    with_tools = heuristic_bucket(episode_of("fix it", tool_names=("Edit", "Bash")))
    assert "Bash, Edit" in with_tools.action and "fix it" in with_tools.action
    talk = heuristic_bucket(episode_of("just talking"))
    assert "discussion only" in talk.action and "just talking" in talk.action


def test_heuristic_rows_do_not_pool_into_one_recommendation():
    # Regression: a constant action like "used tools: Bash" collected every
    # episode's wins and won recommend() for any matching task.
    from mimir import Mimir

    memory = Mimir(":memory:")
    for i, work in enumerate(["Patched the login retry loop", "Raised the login pool size"]):
        for n in range(3):
            ep = Episode("claude-code", f"s{i}{n}", [
                turn("user", f"login is slow, case {n}", 0),
                turn("assistant", work, 1, tools=("Bash",)),
                turn("user", "thanks, works now", 2),
                turn("assistant", "Glad it helped!", 3),
            ])
            b = heuristic_bucket(ep)
            memory.record(b.task, b.action, outcome=b.outcome)
    stats = memory.storage.aggregate_actions("login slow")
    assert sorted(s.action for s in stats) == [
        "Patched the login retry loop", "Raised the login pool size",
    ]


def test_heuristic_bucket_always_sets_an_outcome():
    assert heuristic_bucket(episode_of("anything")).outcome is not None


def test_infer_outcome_reads_positive_reaction():
    assert infer_outcome(episode_of("fix the bug", "thanks, works now")) is Outcome.SUCCESS


def test_infer_outcome_reads_negative_reaction():
    assert infer_outcome(episode_of("fix the bug", "still broken")) is Outcome.FAILURE


def test_infer_outcome_defaults_to_partial_when_ambiguous():
    assert infer_outcome(episode_of("fix the bug", "ok let's move on")) is Outcome.PARTIAL
