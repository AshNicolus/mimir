"""Quality gate: retrieval and recommendation metrics must stay above a floor.

The floors sit below currently observed values, so this fails on a regression
rather than tracking an aspirational target. Raise them when a change earns it.
"""

from benchmarks.eval import run_eval


def test_quality_metrics_hold():
    metrics = run_eval()
    assert metrics["recall_at_k"] >= 0.80, metrics
    assert metrics["mrr"] >= 0.90, metrics
    assert metrics["recommend_accuracy"] == 1.0, metrics
    assert metrics["decay_recency_correct"] == 1.0, metrics
