"""Micro-benchmark for Mimir's hot paths. Prints JSON metrics to stdout.

Deterministic workload (no randomness) so runs are comparable.
"""

import json
import time

from mimir import Mimir

N_RECORD = 2000
N_RECALL = 200
N_RECOMMEND = 50

ACTIONS = [
    "add a redis cache",
    "add a database index",
    "rewrite the slow query",
    "increase the pool size",
    "batch the outbound requests",
]


def build_store():
    m = Mimir(":memory:")
    start = time.perf_counter()
    for i in range(N_RECORD):
        m.record(
            task=f"fix latency in service {i % 100} under load",
            action=ACTIONS[i % len(ACTIONS)],
            outcome="success" if i % 4 else "failure",
        )
    return m, time.perf_counter() - start


def percentiles(samples_ms):
    samples_ms.sort()
    last = len(samples_ms) - 1
    p50 = samples_ms[min(last, int(len(samples_ms) * 0.50))]
    p95 = samples_ms[min(last, int(len(samples_ms) * 0.95))]
    return p50, p95


def time_ms(fn, n):
    samples = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return percentiles(samples)


def main():
    store, record_seconds = build_store()
    recall_p50, recall_p95 = time_ms(lambda: store.recall("latency under load service", k=5), N_RECALL)
    recommend_p50, _ = time_ms(lambda: store.recommend("latency under load"), N_RECOMMEND)
    store.close()

    print(json.dumps({
        "record_ops_per_sec": round(N_RECORD / record_seconds),
        "recall_p50_ms": round(recall_p50, 3),
        "recall_p95_ms": round(recall_p95, 3),
        "recommend_p50_ms": round(recommend_p50, 3),
    }))


if __name__ == "__main__":
    main()
