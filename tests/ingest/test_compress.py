"""compress(): dedupe and budget-aware truncation."""

from datetime import datetime, timedelta, timezone

from mimir.ingest.compress import compress, dedupe, estimate_tokens
from mimir.ingest.models import Episode, Turn

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def turn(i: int, text: str, role: str = "user") -> Turn:
    return Turn(role=role, text=text, timestamp=BASE + timedelta(minutes=i))


def test_dedupe_drops_empty_turns():
    turns = [turn(0, "hello"), Turn(role="user", text="", timestamp=BASE)]
    assert len(dedupe(turns)) == 1


def test_dedupe_drops_back_to_back_repeats():
    turns = [turn(0, "same"), turn(1, "same")]
    assert len(dedupe(turns)) == 1


def test_dedupe_keeps_repeats_from_different_roles():
    turns = [turn(0, "same", role="user"), turn(1, "same", role="assistant")]
    assert len(dedupe(turns)) == 2


def test_small_episode_is_untouched():
    turns = [turn(0, "a short ask")]
    episode = Episode(source="claude-code", session_id="s1", turns=turns)
    compact = compress(episode, budget_tokens=2000)
    assert len(compact.turns) == 1


def oversized_turns() -> list[Turn]:
    # Distinct text per turn: identical text would collapse under dedupe
    # before the head/tail retention path is ever exercised.
    return [turn(i, f"turn {i} " + "x" * 4000) for i in range(10)]


def test_oversized_episode_keeps_head_and_tail():
    turns = oversized_turns()
    episode = Episode(source="claude-code", session_id="s1", turns=turns)
    compact = compress(episode, budget_tokens=100)
    assert compact.turns[0].text == turns[0].text
    assert compact.turns[-1].text == turns[-1].text
    assert compact.turns[0].timestamp == turns[0].timestamp


def test_oversized_episode_notes_what_was_omitted():
    turns = oversized_turns()
    episode = Episode(source="claude-code", session_id="s1", turns=turns)
    compact = compress(episode, budget_tokens=100)
    notices = [t for t in compact.turns if t.role == "system"]
    assert notices and "omitted" in notices[0].text


def test_compression_shrinks_an_oversized_episode():
    # Head+tail are always kept even if they alone exceed the budget, so this
    # checks real shrinkage, not that the result fits inside the budget.
    turns = oversized_turns()
    episode = Episode(source="claude-code", session_id="s1", turns=turns)
    compact = compress(episode, budget_tokens=100)
    assert estimate_tokens(compact.turns) < estimate_tokens(turns)
