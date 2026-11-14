"""Reduces an episode to a size a small model can actually process in time.

Reader-level previews already avoid capturing full tool payloads; this stage
removes remaining redundancy and, if an episode is still oversized, drops the
middle, keeping the start (the original ask) and the end (how it landed).
"""

from __future__ import annotations

from .models import Episode, Turn

CHARS_PER_TOKEN = 4  # rough, model-agnostic; exact tokenization is not worth a new dependency
DEFAULT_BUDGET_TOKENS = 2000
OMITTED_NOTICE = "... {n} turns omitted ..."


def estimate_tokens(turns: list[Turn]) -> int:
    total_chars = sum(len(t.text) + sum(len(c.preview) for c in t.tool_calls) for t in turns)
    return total_chars // CHARS_PER_TOKEN


def dedupe(turns: list[Turn]) -> list[Turn]:
    """Drop empty turns and back-to-back exact repeats."""
    kept: list[Turn] = []
    for turn in turns:
        if not turn.text and not turn.tool_calls:
            continue
        if kept and kept[-1].role == turn.role and kept[-1].text == turn.text:
            continue
        kept.append(turn)
    return kept


def compress(episode: Episode, budget_tokens: int = DEFAULT_BUDGET_TOKENS) -> Episode:
    turns = dedupe(episode.turns)
    if len(turns) <= 4 or estimate_tokens(turns) <= budget_tokens:
        return Episode(source=episode.source, session_id=episode.session_id, turns=turns)

    head, tail = turns[:2], turns[-2:]
    middle = turns[2:-2]
    budget = budget_tokens - estimate_tokens(head) - estimate_tokens(tail)

    survivors: list[Turn] = []
    omitted = 0
    for turn in middle:
        cost = estimate_tokens([turn])
        if cost <= budget:
            survivors.append(turn)
            budget -= cost
        else:
            omitted += 1

    kept = head
    if omitted:
        text = OMITTED_NOTICE.format(n=omitted)
        notice = Turn(role="system", text=text, timestamp=middle[0].timestamp)
        kept = kept + [notice]
    kept = kept + survivors + tail
    return Episode(source=episode.source, session_id=episode.session_id, turns=kept)
