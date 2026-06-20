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
]


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
        b, h = base[key], head[key]
        delta = (h - b) / b * 100 if b else 0.0
        status = "better" if (delta if higher else -delta) >= 0 else "worse"
        rows.append(f"| {label} | {b} | {h} | {status} {delta:+.1f}% |")

    rows += [
        "",
        "_Same-runner comparison, informational only. CI does not fail on regressions (CI timing is noisy)._",
    ]
    print("\n".join(rows))


if __name__ == "__main__":
    main()
