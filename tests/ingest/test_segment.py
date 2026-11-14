"""segment(): splitting a session's turns into episodes by time gap."""

from datetime import datetime, timedelta, timezone

from mimir.ingest.models import Turn
from mimir.ingest.segment import segment

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def turn_at(minutes: int, role: str = "user", text: str = "x") -> Turn:
    return Turn(role=role, text=text, timestamp=BASE + timedelta(minutes=minutes))


def test_empty_session_yields_no_episodes():
    assert segment("claude-code", "s1", []) == []


def test_close_turns_stay_in_one_episode():
    turns = [turn_at(0), turn_at(1), turn_at(2)]
    episodes = segment("claude-code", "s1", turns)
    assert len(episodes) == 1
    assert len(episodes[0].turns) == 3


def test_a_large_gap_starts_a_new_episode():
    turns = [turn_at(0), turn_at(5), turn_at(200)]  # 200 - 5 = 195min > default 30min gap
    episodes = segment("claude-code", "s1", turns)
    assert len(episodes) == 2
    assert len(episodes[0].turns) == 2
    assert len(episodes[1].turns) == 1


def test_custom_gap_is_respected():
    turns = [turn_at(0), turn_at(10)]
    episodes = segment("claude-code", "s1", turns, gap=timedelta(minutes=5))
    assert len(episodes) == 2


def test_episode_carries_source_and_session_id():
    episodes = segment("cursor", "session-42", [turn_at(0)])
    assert episodes[0].source == "cursor"
    assert episodes[0].session_id == "session-42"


def test_episode_start_and_end_properties():
    turns = [turn_at(0), turn_at(1), turn_at(2)]
    episode = segment("claude-code", "s1", turns)[0]
    assert episode.start == turns[0].timestamp
    assert episode.end == turns[-1].timestamp
