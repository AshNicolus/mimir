"""Retrieval and recommendation quality eval for Mimir's default (keyword) path.

Seeds a labeled corpus, then scores recall and recommendation against known-good
answers. Unlike the speed benchmark this measures correctness, so it doubles as a
regression gate: tests/test_eval.py asserts these metrics stay above a floor.

    python -m benchmarks.eval          # print metrics as JSON
"""

from __future__ import annotations

import json
from datetime import timedelta

from mimir import Experience, Mimir
from mimir.models import utcnow

from .eval_dataset import DECAY_CASE, DECAY_SEEDS, RECALL_CASES, RECOMMEND_CASES, SEEDS


def seed_store() -> tuple[Mimir, dict[str, str]]:
    """Record the corpus and return the store plus a label -> id map."""
    memory = Mimir(":memory:")
    labels = {}
    for s in SEEDS:
        exp = memory.record(s.task, s.action, outcome=s.outcome)
        labels[exp.id] = s.label
    return memory, labels


def recall_metrics(memory: Mimir, labels: dict[str, str], k: int = 5) -> dict[str, float]:
    """recall@k (share of relevant labels retrieved) and MRR (mean reciprocal
    rank of the first relevant hit), averaged over the recall cases."""
    recall_at_k = 0.0
    reciprocal_rank = 0.0
    for case in RECALL_CASES:
        hits = [labels[e.id] for e in memory.recall(case.query, k=k)]
        found = case.relevant.intersection(hits)
        recall_at_k += len(found) / len(case.relevant)
        rank = next((i for i, label in enumerate(hits, 1) if label in case.relevant), 0)
        reciprocal_rank += 1.0 / rank if rank else 0.0
    n = len(RECALL_CASES)
    return {"recall_at_k": recall_at_k / n, "mrr": reciprocal_rank / n}


def recommend_accuracy(memory: Mimir) -> float:
    """Share of recommend cases whose top action matches the expected one."""
    correct = 0
    for case in RECOMMEND_CASES:
        rec = memory.recommend(case.task)
        if rec is not None and rec.recommended_action == case.expected_action:
            correct += 1
    return correct / len(RECOMMEND_CASES)


def decay_recency_correct(half_life_days: float = 30) -> float:
    """1.0 if decay surfaces the recent action over the larger stale sample."""
    memory = Mimir(":memory:", half_life_days=half_life_days)
    try:
        for s in DECAY_SEEDS:
            created = utcnow() - timedelta(days=s.days_ago)
            memory.write(Experience(task=s.task, action=s.action, created_at=created))
        rec = memory.recommend(DECAY_CASE.task)
        return 1.0 if rec and rec.recommended_action == DECAY_CASE.recent_action else 0.0
    finally:
        memory.close()


def run_eval(k: int = 5) -> dict[str, float]:
    memory, labels = seed_store()
    try:
        metrics = recall_metrics(memory, labels, k=k)
        metrics["recommend_accuracy"] = recommend_accuracy(memory)
    finally:
        memory.close()
    metrics["decay_recency_correct"] = decay_recency_correct()
    return {key: round(value, 4) for key, value in metrics.items()}


def main() -> None:
    print(json.dumps(run_eval()))


if __name__ == "__main__":
    main()
