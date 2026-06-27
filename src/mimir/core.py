"""The Mimir public API.

Everything users touch goes through this class. All writes funnel through a
single chokepoint (`write`) — that one method is the seam where validation,
provenance tagging, and (future) a memory-firewall layer plug in without
touching the rest of the system.
"""

from __future__ import annotations

import json
import math
import threading

from .clustering import ActionClusterer
from .embeddings import Embedder, NullEmbedder
from .models import Experience, Outcome, Recommendation
from .storage import SQLiteStorage, Storage


class Mimir:
    def __init__(
        self,
        db_path: str = "mimir.db",
        *,
        storage: Storage | None = None,
        embedder: Embedder | None = None,
        clusterer: ActionClusterer | None = None,
        weight_by_relevance: bool = True,
    ) -> None:
        self._storage = storage or SQLiteStorage(db_path, clusterer=clusterer)
        self._embedder = embedder or NullEmbedder()
        self._lock = threading.Lock()
        self._weight_by_relevance = weight_by_relevance

    def record(
        self,
        task: str,
        action: str,
        outcome: str | Outcome = Outcome.SUCCESS,
        score: float | None = None,
        context: dict | None = None,
        supersedes: str | None = None,
    ) -> Experience:
        """Record an experience: a task, the action taken, and how it went.

        Pass ``supersedes`` with an existing id to mark that older experience
        as replaced by this one, hiding it from recall and recommendation.
        """
        outcome = Outcome(outcome) if not isinstance(outcome, Outcome) else outcome
        if score is None:
            score = default_score(outcome)
        exp = Experience(
            task=task,
            action=action,
            outcome=outcome,
            score=score,
            context=context or {},
        )
        new = self.write(exp)
        if supersedes is not None:
            self.supersede(supersedes, new.id)
        return new

    def record_failure(
        self,
        task: str,
        action: str,
        reason: str | None = None,
        score: float = 0.0,
        context: dict | None = None,
        supersedes: str | None = None,
    ) -> Experience:
        """Record a failure. Stored like any experience but with outcome=failure,
        so agents can recall what *didn't* work and stop repeating it."""
        ctx = dict(context or {})
        if reason:
            ctx["failure_reason"] = reason
        return self.record(
            task, action, outcome=Outcome.FAILURE, score=score, context=ctx, supersedes=supersedes
        )

    def supersede(self, old_id: str, new_id: str) -> bool:
        """Mark ``old_id`` as superseded by ``new_id``. Superseded experiences
        are hidden from recall and recommendation by default, while staying
        retrievable by id. Returns True if ``old_id`` existed."""
        return self._storage.set_superseded_by(old_id, new_id)

    def write(self, exp: Experience) -> Experience:
        """The single write chokepoint. Validation/provenance/firewall hooks go here."""
        # Fail fast with a clear message rather than crashing deep in storage.
        try:
            json.dumps(exp.context)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"context must be JSON-serializable: {exc}") from exc
        with self._lock:
            if self._embedder.enabled and exp.embedding is None:
                exp.embedding = self._embedder.embed(exp.text())
            self._storage.add(exp)
        return exp

    def recall(
        self,
        query: str,
        k: int = 5,
        outcome: str | Outcome | None = None,
        context: dict | None = None,
        include_superseded: bool = False,
    ) -> list[Experience]:
        """Return the most relevant past experiences for ``query``.

        With an embedder configured, recall is hybrid: keyword and vector
        candidates are fused, so an experience can be found by meaning even with
        no shared words. Without one, it is plain keyword search.

        Superseded experiences are excluded unless ``include_superseded`` is True.
        """
        outcome_val = outcome.value if isinstance(outcome, Outcome) else outcome
        width = max(k * 4, k)
        keyword = self._storage.search(
            query, k=width, outcome=outcome_val, context=context,
            include_superseded=include_superseded,
        )
        if not self._embedder.enabled:
            return [exp for exp, _ in keyword[:k]]
        vector = self._storage.vector_search(
            self._embedder.embed(query), k=width, outcome=outcome_val, context=context,
            include_superseded=include_superseded,
        )
        fused = reciprocal_rank_fusion(keyword, vector)
        return [exp for exp, _ in fused[:k]]

    def get(self, experience_id: str) -> Experience | None:
        """Fetch a single experience by id, or None if it doesn't exist."""
        return self._storage.get(experience_id)

    def delete(self, experience_id: str) -> bool:
        """Delete an experience by id. Returns True if it existed."""
        return self._storage.delete(experience_id)

    def recent(self, n: int = 10) -> list[Experience]:
        """Return the ``n`` most recently recorded experiences, newest first."""
        return self._storage.recent(n)

    def recommend(
        self,
        task: str,
        *,
        weight_by_relevance: bool | None = None,
        include_superseded: bool = False,
    ) -> Recommendation | None:
        """Suggest a strategy for a new task by aggregating similar past
        experiences. Returns None if there's nothing relevant to go on.

        Confidence is the Wilson lower bound of each action's success rate, so a
        9/10 action outranks a lucky 1/1. When relevance weighting is on (the
        default), each experience contributes its relevance to the query rather
        than a flat 1, so an action backed by more relevant evidence outranks an
        equally-successful but less relevant one. Pass ``weight_by_relevance``
        to override the instance default, which makes the weighting ablatable.

        Reported counts always cover the full matching population exactly;
        weighting only affects ranking.
        """
        weighted = self._weight_by_relevance if weight_by_relevance is None else weight_by_relevance
        best_stat = None
        best_confidence = -1.0
        for stat in self._storage.aggregate_actions(task, include_superseded=include_superseded):
            if stat.success + 0.5 * stat.partial == 0:
                continue  # never recommend an action with no wins
            if weighted:
                successes = stat.weighted_success + 0.5 * stat.weighted_partial
                total = stat.weighted_total
            else:
                successes = stat.success + 0.5 * stat.partial
                total = stat.total
            confidence = wilson_lower_bound(successes, total)
            if confidence > best_confidence:
                best_confidence, best_stat = confidence, stat
        if best_stat is None:
            return None
        return Recommendation(
            task=task,
            recommended_action=best_stat.action,
            confidence=best_confidence,
            success_count=best_stat.success,
            failure_count=best_stat.failure,
            partial_count=best_stat.partial,
            based_on=best_stat.total,
            supporting_ids=self._storage.supporting_ids(
                task, best_stat.key, include_superseded=include_superseded
            ),
        )

    def count(self) -> int:
        return self._storage.count()

    def close(self) -> None:
        self._storage.close()

    def __enter__(self) -> "Mimir":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def default_score(outcome: Outcome) -> float:
    return {Outcome.SUCCESS: 1.0, Outcome.PARTIAL: 0.5, Outcome.FAILURE: 0.0}[outcome]


def reciprocal_rank_fusion(
    *rankings: list[tuple[Experience, float]], c: int = 60
) -> list[tuple[Experience, float]]:
    """Merge ranked candidate lists into one, scoring each item by the sum of
    1 / (c + rank) across the lists it appears in. Rank-based, so it fuses the
    keyword and vector lists without their incompatible score scales fighting.
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
    """Lower bound of a Wilson score interval for a binomial proportion.

    Rewards both a high success rate and a large sample size. ``total`` may be a
    relevance-weighted (non-integer) effective count, which turns this into a
    weighted-evidence estimator: a heuristic, not a strict binomial interval.
    """
    if total <= 0:
        return 0.0
    phat = successes / total
    denom = 1 + z * z / total
    centre = phat + z * z / (2 * total)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total)
    return max(0.0, (centre - margin) / denom)
