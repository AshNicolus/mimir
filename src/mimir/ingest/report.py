"""Aggregate insights over stored, bucketed episodes: real counts from the
store, not a model-generated narrative.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from ..core import Mimir
from ..models import Experience

UNCATEGORIZED = "uncategorized"


def build_report(memory: Mimir, limit: int = 10_000) -> str:
    experiences = memory.recent(limit)
    if not experiences:
        return "No experiences stored yet."

    by_category: dict[str, list[Experience]] = defaultdict(list)
    for exp in experiences:
        by_category[exp.context.get("category", UNCATEGORIZED)].append(exp)

    lines = [f"{len(experiences)} experiences across {len(by_category)} categories:", ""]
    for category, exps in sorted(by_category.items(), key=lambda kv: -len(kv[1])):
        successes = sum(1 for e in exps if e.outcome.value == "success")
        failures = sum(1 for e in exps if e.outcome.value == "failure")
        partials = sum(1 for e in exps if e.outcome.value == "partial")
        top_action, top_count = Counter(e.action for e in exps).most_common(1)[0]
        top_agent, _ = Counter(e.context.get("agent", "unknown") for e in exps).most_common(1)[0]
        lines.append(
            f"[{category}] {len(exps)} episodes "
            f"({successes} success / {failures} failure / {partials} partial), "
            f"most common action: {top_action!r} ({top_count}x), mostly from {top_agent}"
        )
    return "\n".join(lines)
