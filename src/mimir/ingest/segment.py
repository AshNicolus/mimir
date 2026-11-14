"""Splits a session's turns into episodes by time gap.

A session file can span unrelated tasks hours apart; a gap this large marks a
new episode. This is a simple v1 heuristic; topic-shift detection (e.g. via
embeddings) is a natural upgrade once this proves too coarse in practice.
"""

from __future__ import annotations

from datetime import timedelta

from .models import Episode, Turn

DEFAULT_GAP = timedelta(minutes=30)


def segment(
    source: str, session_id: str, turns: list[Turn], gap: timedelta = DEFAULT_GAP
) -> list[Episode]:
    if not turns:
        return []
    groups: list[list[Turn]] = [[turns[0]]]
    for previous, current in zip(turns, turns[1:]):
        if current.timestamp - previous.timestamp > gap:
            groups.append([])
        groups[-1].append(current)
    return [Episode(source=source, session_id=session_id, turns=group) for group in groups]
