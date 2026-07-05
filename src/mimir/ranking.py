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


def time_decay(age_days: float, half_life_days: float) -> float:
    """Halve an experience's weight every ``half_life_days``."""
    if half_life_days <= 0:
        return 1.0
    return 2.0 ** (-max(age_days, 0.0) / half_life_days)


def incomplete_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b), i.e. the Beta(a, b) CDF at x."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    front = math.exp(ln_front)
    # The continued fraction converges quickly below this pivot; past it, the
    # symmetry I_x(a, b) = 1 - I_(1-x)(b, a) keeps it in the fast region.
    if x < (a + 1.0) / (a + b + 2.0):
        return front * beta_cf(a, b, x) / a
    return 1.0 - front * beta_cf(b, a, 1.0 - x) / b


def beta_cf(a: float, b: float, x: float) -> float:
    # Lentz's algorithm for the incomplete-beta continued fraction.
    tiny = 1e-30
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (tiny if abs(d) < tiny else d)
    f = d
    for m in range(1, 200):
        m2 = 2 * m
        for aa in (
            m * (b - m) * x / ((a + m2 - 1.0) * (a + m2)),
            -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0)),
        ):
            d = 1.0 + aa * d
            d = 1.0 / (tiny if abs(d) < tiny else d)
            c = 1.0 + aa / c
            if abs(c) < tiny:
                c = tiny
            f *= d * c
        if abs(d * c - 1.0) < 1e-12:
            break
    return f


def beta_lower_bound(
    successes: float, failures: float, prior: float = 0.5, tail: float = 0.05
) -> float:
    """Lower end of a one-sided credible interval for the success rate: the
    ``tail`` quantile of the Beta(prior + successes, prior + failures) posterior.

    A Jeffreys prior (0.5) keeps small samples honest, and more evidence, whether
    counted raw or relevance/recency-weighted, concentrates the posterior and
    lifts the bound. Found by bisecting the Beta CDF, which is monotonic in x.
    """
    a = prior + successes
    b = prior + failures
    if successes <= 0.0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if incomplete_beta(a, b, mid) < tail:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0
