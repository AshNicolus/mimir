"""Scoring and ranking math shared by recall and recommend."""

from __future__ import annotations

import math

from .models import Experience, Outcome


def default_score(outcome: Outcome) -> float:
    return {Outcome.SUCCESS: 1.0, Outcome.PARTIAL: 0.5, Outcome.FAILURE: 0.0}[outcome]


def reciprocal_rank_fusion(
    *rankings: list[tuple[Experience, float]], c: int = 60
) -> list[tuple[Experience, float]]:
    """Merge ranked lists by summing 1 / (c + rank) per item.

    Rank-based, so keyword and vector lists fuse without their incompatible
    score scales fighting.
    """
    scores: dict[str, float] = {}
    experiences: dict[str, Experience] = {}
    for ranking in rankings:
        for rank, (exp, _) in enumerate(ranking):
            scores[exp.id] = scores.get(exp.id, 0.0) + 1.0 / (c + rank)
            experiences[exp.id] = exp
    fused = [(experiences[exp_id], score) for exp_id, score in scores.items()]
    fused.sort(key=lambda pair: pair[1], reverse=True)
    return fused


def wilson_lower_bound(successes: float, total: float, z: float = 1.96) -> float:
    """Lower bound of a Wilson score interval: rewards both a high success rate
    and a large sample. Accepts a relevance-weighted (non-integer) total."""
    if total <= 0:
        return 0.0
    phat = successes / total
    denom = 1 + z * z / total
    centre = phat + z * z / (2 * total)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total)
    return max(0.0, (centre - margin) / denom)
