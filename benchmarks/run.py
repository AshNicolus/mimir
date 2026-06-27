"""Micro-benchmark for Mimir's hot paths. Prints JSON metrics to stdout.

Deterministic workload (no randomness) so runs are comparable.
"""

import json
import os
import tempfile
import threading
import time

from mimir import Mimir

N_RECORD = 2000
N_RECALL = 200
N_RECOMMEND = 50
N_READ_EACH = 400  # reads per thread in the concurrency benchmark
READ_QUERY = "latency under load service"

ACTIONS = [
    "add a redis cache",
    "add a database index",
    "rewrite the slow query",
    "increase the pool size",
    "batch the outbound requests",
]


def seed(store):
    for i in range(N_RECORD):
        store.record(
            task=f"fix latency in service {i % 100} under load",
            action=ACTIONS[i % len(ACTIONS)],
            outcome="success" if i % 4 else "failure",
        )


def build_store():
    m = Mimir(":memory:")
    start = time.perf_counter()
    seed(m)
    return m, time.perf_counter() - start


def read_ops_per_sec(store, n_threads):
    # Fixed work per thread, measured by wall clock: if reads run concurrently,
    # n_threads finish in about the time of one, so throughput scales with cores.
    ready = threading.Barrier(n_threads + 1)

    def worker():
        ready.wait()
        for _ in range(N_READ_EACH):
            store.recall(READ_QUERY, k=5)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    ready.wait()  # release all workers together
    start = time.perf_counter()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - start
    return round(n_threads * N_READ_EACH / elapsed)


def read_scaling():
    # File-backed so WAL applies: in-memory stores share one connection and
    # can't show reader concurrency.
    path = os.path.join(tempfile.mkdtemp(), "bench.db")
    store = Mimir(db_path=path)
    try:
        seed(store)
        return read_ops_per_sec(store, 1), read_ops_per_sec(store, 4)
    finally:
        store.close()


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

    read_1t, read_4t = read_scaling()

    print(json.dumps({
        "record_ops_per_sec": round(N_RECORD / record_seconds),
        "recall_p50_ms": round(recall_p50, 3),
        "recall_p95_ms": round(recall_p95, 3),
        "recommend_p50_ms": round(recommend_p50, 3),
        "read_1thread_ops_per_sec": read_1t,
        "read_4thread_ops_per_sec": read_4t,
    }))


if __name__ == "__main__":
    main()
