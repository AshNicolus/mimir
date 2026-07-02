"""Helpers that turn queries and context filters into SQL-ready pieces."""

from __future__ import annotations

import re

TOKEN = re.compile(r"[a-z0-9]+")
SIMPLE_KEY = re.compile(r"[A-Za-z0-9_]+")

STOPWORDS = frozenset(
    "a an the is are was were be been to of in on for and or it this that with "
    "as at by from how what i my we".split()
)


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def query_terms(query: str) -> list[str]:
    terms = tokenize(query)
    meaningful = [t for t in terms if t not in STOPWORDS]
    return meaningful or terms


def bm25_to_score(rank: float) -> float:
    # bm25 ranks are unbounded and negative-leaning; map them into (0, 1].
    return 1.0 / (1.0 + max(rank, 0.0))


def context_matches(stored: dict, wanted: dict) -> bool:
    return all(stored.get(key) == value for key, value in wanted.items())


def context_sql_filters(context: dict | None) -> list[tuple[str, object]]:
    # Only plain keys with scalar values can become json_extract equality checks;
    # everything else is left for context_matches to verify in Python.
    if not context:
        return []
    filters = []
    for key, value in context.items():
        if isinstance(value, (str, int, float)) and SIMPLE_KEY.fullmatch(key):
            filters.append((f"$.{key}", value))
    return filters
