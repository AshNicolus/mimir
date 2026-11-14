"""Compare two benchmark JSON files and print a markdown table.

Usage: python benchmarks/compare.py base.json head.json
"""

import json
import sys

# (key, label, higher_is_better)
METRICS = [
    ("record_ops_per_sec", "record (ops/sec)", True),
    ("recall_p50_ms", "recall p50 (ms)", False),
    ("recall_p95_ms", "recall p95 (ms)", False),
    ("recommend_p50_ms", "recommend p50 (ms)", False),
    ("read_1thread_ops_per_sec", "concurrent read, 1 thread (ops/sec)", True),
    ("read_4thread_ops_per_sec", "concurrent read, 4 threads (ops/sec)", True),
]

# Shared CI runners swing by several percent run to run; smaller changes are
# reported as noise rather than as a win or a regression.
NOISE_PCT = 10.0


def status(delta, higher):
    if abs(delta) < NOISE_PCT:
        return "~ same"
    return "better" if (delta if higher else -delta) > 0 else "**worse**"


def get(data, key):
    # Tolerate older runs that predate a metric so the table still renders.
    return data.get(key, 0)


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    base = load(sys.argv[1])
    head = load(sys.argv[2])

    rows = [
        "### Performance: this PR vs base",
        "",
        "| metric | base | this PR | change |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label, higher in METRICS:
        b, h = get(base, key), get(head, key)
        if b:
            delta = (h - b) / b * 100
            change = f"{status(delta, higher)} {delta:+.1f}%"
        else:
            change = "n/a"  # metric missing from the base run
        rows.append(f"| {label} | {b:,} | {h:,} | {change} |")

    rows += [
        "",
        f"_Same-runner comparison, informational only. Changes under {NOISE_PCT:.0f}% "
        "read as noise. CI does not fail on regressions (CI timing is noisy)._",
    ]
    print("\n".join(rows))


if __name__ == "__main__":
    main()
