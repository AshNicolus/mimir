"""The Mimir public API. Every write funnels through one chokepoint, write(),
where validation and future provenance hooks live."""

from __future__ import annotations

import json
import threading
from collections import OrderedDict

from .clustering import ActionClusterer
from .embeddings import Embedder, NullEmbedder
from .models import Experience, Outcome, Recommendation, utcnow
from .ranking import beta_lower_bound, default_score, reciprocal_rank_fusion, time_decay
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
        half_life_days: float | None = None,
        query_cache_size: int = 256,
    ) -> None:
        self.storage = storage or SQLiteStorage(db_path, clusterer=clusterer)
        self.embedder = embedder or NullEmbedder()
        self.weight_by_relevance = weight_by_relevance
        # Age at which past evidence counts for half. None keeps all evidence equal.
        self.half_life_days = half_life_days
        # Serializes writes: embedders aren't guaranteed thread-safe.
        self.lock = threading.Lock()
        # LRU of query text -> embedding: agents repeat queries on retries, and
        # re-embedding costs a model pass or an API call. Set 0 to disable.
        self.query_cache_size = query_cache_size
        self.query_cache: OrderedDict[str, list[float]] = OrderedDict()
        self.query_cache_lock = threading.Lock()

    def record(
        self,
        task: str,
        action: str,
        outcome: str | Outcome = Outcome.SUCCESS,
        score: float | None = None,
        context: dict | None = None,
        supersedes: str | None = None,
    ) -> Experience:
        """Record an experience. Pass supersedes=<id> to mark an older
        experience as replaced by this one."""
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
        """Record what didn't work, so agents can stop repeating it."""
        ctx = dict(context or {})
        if reason:
            ctx["failure_reason"] = reason
        return self.record(
            task, action, outcome=Outcome.FAILURE, score=score, context=ctx, supersedes=supersedes
        )

    def supersede(self, old_id: str, new_id: str) -> bool:
        """Mark old_id as replaced by new_id, hiding it from recall and
        recommendation. Returns True if old_id existed."""
        return self.storage.set_superseded_by(old_id, new_id)

    def write(self, exp: Experience) -> Experience:
        """Store an experience. Returns the stored object; when an embedder
        fills in the embedding this is a copy, the caller's object is untouched."""
        try:
            json.dumps(exp.context)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"context must be JSON-serializable: {exc}") from exc
        with self.lock:
            if self.embedder.enabled and exp.embedding is None:
                exp = exp.model_copy(update={"embedding": self.embedder.embed(exp.text())})
            self.storage.add(exp)
        return exp

    def recall(
        self,
        query: str,
        k: int = 5,
        outcome: str | Outcome | None = None,
        context: dict | None = None,
        include_superseded: bool = False,
    ) -> list[Experience]:
        """Return the most relevant past experiences for the query.

        With an embedder configured, keyword and vector candidates are fused so
        an experience can match by meaning alone; otherwise it is keyword search.
        With ``half_life_days`` set, recency reweights candidates so a fresh
        experience outranks an equally relevant but staler one.
        """
        outcome_val = outcome.value if isinstance(outcome, Outcome) else outcome
        width = k * 4  # over-fetch candidates before fusing and trimming to k
        keyword = self.storage.search(
            query, k=width, outcome=outcome_val, context=context,
            include_superseded=include_superseded,
        )
        if self.embedder.enabled:
            vector = self.storage.vector_search(
                self.embed_query(query), k=width, outcome=outcome_val, context=context,
                include_superseded=include_superseded,
            )
            candidates = reciprocal_rank_fusion(keyword, vector)
        else:
            candidates = keyword
        if self.half_life_days:
            candidates = self.apply_recency(candidates)
        return [exp for exp, _ in candidates[:k]]

    def apply_recency(self, candidates: list[tuple[Experience, float]]):
        now = utcnow()
        scored = []
        for exp, score in candidates:
            age_days = (now - exp.created_at).total_seconds() / 86400
            scored.append((exp, score * time_decay(age_days, self.half_life_days)))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored

    def embed_query(self, query: str) -> list[float]:
        """Embed a query, reusing a cached vector for a repeated one. Embeddings
        are pure functions of text, so the cache never needs invalidating."""
        if self.query_cache_size <= 0:
            return self.embedder.embed(query)
        with self.query_cache_lock:
            cached = self.query_cache.get(query)
            if cached is not None:
                self.query_cache.move_to_end(query)
                return cached
        vector = self.embedder.embed(query)
        with self.query_cache_lock:
            self.query_cache[query] = vector
            self.query_cache.move_to_end(query)
            while len(self.query_cache) > self.query_cache_size:
                self.query_cache.popitem(last=False)
        return vector

    def get(self, experience_id: str) -> Experience | None:
        return self.storage.get(experience_id)

    def delete(self, experience_id: str) -> bool:
        return self.storage.delete(experience_id)

    def recent(self, n: int = 10) -> list[Experience]:
        return self.storage.recent(n)

    def recommend(
        self,
        task: str,
        *,
        weight_by_relevance: bool | None = None,
        include_superseded: bool = False,
    ) -> Recommendation | None:
        """Suggest the action with the strongest track record on similar tasks.

        Confidence is the Beta-posterior lower bound of the action's success rate
        from its raw counts, so it reads as a success rate: a 9/10 action beats a
        lucky 1/1, and the number means the same whatever the query. Relevance
        (and recency, when ``half_life_days`` is set) only steers which action
        wins, not the confidence: with weighting on, an action proven on closely
        matching tasks outranks an equally confident one proven on loosely related
        ones. Turn weighting off to rank on track record alone.
        """
        weighted = self.weight_by_relevance if weight_by_relevance is None else weight_by_relevance
        best_stat = None
        best_confidence = 0.0
        best_rank = -1.0
        stats = self.storage.aggregate_actions(
            task, include_superseded=include_superseded, half_life_days=self.half_life_days
        )
        for stat in stats:
            if stat.success + 0.5 * stat.partial == 0:
                continue  # never recommend an action with no wins
            # Partial counts as half a success and half a failure.
            confidence = beta_lower_bound(
                stat.success + 0.5 * stat.partial, stat.failure + 0.5 * stat.partial
            )
            # Rank by confidence scaled by how relevant/recent the evidence is.
            rank = confidence * (stat.weighted_total / stat.total) if weighted else confidence
            if rank > best_rank:
                best_rank, best_confidence, best_stat = rank, confidence, stat
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
            supporting_ids=self.storage.supporting_ids(
                task, best_stat.key, include_superseded=include_superseded
            ),
        )

    def count(self) -> int:
        return self.storage.count()

    def close(self) -> None:
        self.storage.close()

    def __enter__(self) -> "Mimir":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
