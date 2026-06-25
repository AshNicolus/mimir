"""The Mimir public API.

Everything users touch goes through this class. All writes funnel through a
single chokepoint (`_write`) — that one method is the seam where validation,
provenance tagging, and (future) a memory-firewall layer plug in without
touching the rest of the system.
"""

from __future__ import annotations

import json
import math
import threading

from .embeddings import Embedder, NullEmbedder, cosine_similarity
from .models import Experience, Outcome, Recommendation
from .storage import SQLiteStorage, Storage


class Mimir:
    def __init__(
        self,
        db_path: str = "mimir.db",
        *,
        storage: Storage | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self._storage = storage or SQLiteStorage(db_path)
        self._embedder = embedder or NullEmbedder()
        self._lock = threading.Lock()

    def record(
        self,
        task: str,
        action: str,
        outcome: str | Outcome = Outcome.SUCCESS,
        score: float | None = None,
        context: dict | None = None,
    ) -> Experience:
        """Record an experience: a task, the action taken, and how it went."""
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
        return self._write(exp)

    def record_failure(
        self,
        task: str,
        action: str,
        reason: str | None = None,
        score: float = 0.0,
        context: dict | None = None,
    ) -> Experience:
        """Record a failure. Stored like any experience but with outcome=failure,
        so agents can recall what *didn't* work and stop repeating it."""
        ctx = dict(context or {})
        if reason:
            ctx["failure_reason"] = reason
        return self.record(task, action, outcome=Outcome.FAILURE, score=score, context=ctx)

    def _write(self, exp: Experience) -> Experience:
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
    ) -> list[Experience]:
        """Return the most relevant past experiences for ``query``."""
        outcome_val = outcome.value if isinstance(outcome, Outcome) else outcome
        # Pull a wider candidate set so optional embedding rerank has room to work.
        candidates = self._storage.search(
            query, k=max(k * 4, k), outcome=outcome_val, context=context
        )
        if self._embedder.enabled:
            candidates = self._rerank(query, candidates)
        return [exp for exp, _ in candidates[:k]]

    def get(self, experience_id: str) -> Experience | None:
        """Fetch a single experience by id, or None if it doesn't exist."""
        return self._storage.get(experience_id)

    def delete(self, experience_id: str) -> bool:
        """Delete an experience by id. Returns True if it existed."""
        return self._storage.delete(experience_id)

    def recent(self, n: int = 10) -> list[Experience]:
        """Return the ``n`` most recently recorded experiences, newest first."""
        return self._storage.recent(n)

    def recommend(self, task: str) -> Recommendation | None:
        """Suggest a strategy for a new task by aggregating similar past
        experiences. Returns None if there's nothing relevant to go on.

        Confidence is the Wilson lower bound of each action's success rate, so a
        9/10 action outranks a lucky 1/1. Counts cover all matches, not a sample.
        """
        best: Recommendation | None = None
        for stat in self._storage.aggregate_actions(task):
            effective_successes = stat.success + 0.5 * stat.partial
            if effective_successes == 0:
                continue  # never recommend an action with no wins
            rec = Recommendation(
                task=task,
                recommended_action=stat.action,
                confidence=wilson_lower_bound(effective_successes, stat.total),
                success_count=stat.success,
                failure_count=stat.failure,
                partial_count=stat.partial,
                based_on=stat.total,
                supporting_ids=stat.supporting_ids,
            )
            if best is None or rec.confidence > best.confidence:
                best = rec
        return best

    def _rerank(
        self, query: str, candidates: list[tuple[Experience, float]]
    ) -> list[tuple[Experience, float]]:
        qvec = self._embedder.embed(query)
        rescored = []
        for exp, kw_score in candidates:
            sem = cosine_similarity(qvec, exp.embedding) if exp.embedding else 0.0
            # Blend keyword and semantic relevance.
            rescored.append((exp, 0.5 * kw_score + 0.5 * sem))
        rescored.sort(key=lambda pair: pair[1], reverse=True)
        return rescored

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


def wilson_lower_bound(successes: float, total: int, z: float = 1.96) -> float:
    """Lower bound of a Wilson score interval for a binomial proportion.

    Rewards both a high success rate and a large sample size.
    """
    if total == 0:
        return 0.0
    phat = successes / total
    denom = 1 + z * z / total
    centre = phat + z * z / (2 * total)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total)
    return max(0.0, (centre - margin) / denom)
