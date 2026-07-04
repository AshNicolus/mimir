"""record() and recall(): keyword, hybrid, filters, and scaling."""

import pytest

from mimir import Mimir, Outcome
from mimir.embeddings import Embedder


def test_record_and_count(memory):
    memory.record("Fix login latency", "Added Redis cache", outcome="success", score=0.9)
    assert memory.count() == 1


def test_record_defaults_score_from_outcome(memory):
    exp = memory.record("task", "action", outcome="failure")
    assert exp.outcome is Outcome.FAILURE
    assert exp.score == 0.0


def test_recall_finds_relevant_experience(memory):
    memory.record("Fix authentication timeout", "Implemented Redis caching", score=0.95)
    memory.record("Render a chart", "Used matplotlib", score=0.8)

    results = memory.recall("authentication latency", k=5)

    assert results, "expected at least one recalled experience"
    assert "Redis" in results[0].action


def test_recall_returns_nothing_for_unrelated_query(memory):
    memory.record("Fix authentication timeout", "Implemented Redis caching")
    assert memory.recall("how to bake sourdough bread") == []


class TopicEmbedder(Embedder):
    """Maps text to a topic vector by keyword, so synonyms with no shared words
    still embed to the same vector."""

    TOPICS = {
        0: ("cat", "feline", "kitten"),
        1: ("dog", "canine", "puppy"),
    }

    def embed(self, text):
        t = text.lower()
        vec = [0.0, 0.0, 1.0]  # default topic, distinct from the two below
        for axis, words in self.TOPICS.items():
            if any(w in t for w in words):
                vec = [0.0, 0.0, 0.0]
                vec[axis] = 1.0
                break
        return vec


def test_recall_finds_semantic_match_without_keyword_overlap():
    m = Mimir(":memory:", embedder=TopicEmbedder())
    try:
        m.record("adopt a feline companion", "visit the shelter", outcome="success")
        m.record("buy a canine leash", "go to the pet store", outcome="success")

        results = m.recall("kitten care tips")
        assert results
        assert results[0].task == "adopt a feline companion"
    finally:
        m.close()


class CountingEmbedder(TopicEmbedder):
    """TopicEmbedder that records how many times embed() ran."""

    def __init__(self):
        self.calls = 0

    def embed(self, text):
        self.calls += 1
        return super().embed(text)


def test_repeated_query_is_embedded_once():
    embedder = CountingEmbedder()
    m = Mimir(":memory:", embedder=embedder)
    try:
        m.record("adopt a feline companion", "visit the shelter")
        base = embedder.calls  # embedding the stored experience
        m.recall("kitten care")
        m.recall("kitten care")
        assert embedder.calls == base + 1  # second recall hits the cache
    finally:
        m.close()


def test_query_cache_can_be_disabled():
    embedder = CountingEmbedder()
    m = Mimir(":memory:", embedder=embedder, query_cache_size=0)
    try:
        m.record("adopt a feline companion", "visit the shelter")
        base = embedder.calls
        m.recall("kitten care")
        m.recall("kitten care")
        assert embedder.calls == base + 2  # every recall re-embeds
    finally:
        m.close()


def test_query_cache_evicts_least_recently_used():
    embedder = CountingEmbedder()
    m = Mimir(":memory:", embedder=embedder, query_cache_size=2)
    try:
        m.embed_query("a")
        m.embed_query("b")
        m.embed_query("a")  # touch "a" so "b" is now least recent
        m.embed_query("c")  # evicts "b"
        assert set(m.query_cache) == {"a", "c"}
    finally:
        m.close()


def test_recall_without_embeddings_misses_semantic_only_match(memory):
    memory.record("adopt a feline companion", "visit the shelter")
    assert memory.recall("kitten care tips") == []


def test_search_scores_rank_stronger_matches_higher(memory):
    memory.record("fix login latency under load", "add a redis cache")
    memory.record("unrelated gardening chores with latency", "water the plants")

    scores = {exp.task: score for exp, score in memory.storage.search("login latency", k=5)}
    assert scores["fix login latency under load"] > scores["unrelated gardening chores with latency"]


def test_search_with_zero_limit_returns_empty(memory):
    memory.record("a task", "an action")
    assert memory.storage.search("task", k=0) == []


def test_recall_ignores_common_stopwords(memory):
    memory.record("improve database latency", "add an index")
    memory.record("the weather is nice today", "go outside")
    results = memory.recall("what is the latency")
    assert len(results) == 1
    assert results[0].action == "add an index"


def test_record_failure_is_queryable_separately(memory):
    memory.record_failure(
        "Throttle abusive clients",
        "Fixed-window rate limiter",
        reason="WebSocket traffic wasn't handled",
    )
    failures = memory.recall("rate limiter", outcome="failure")
    assert len(failures) == 1
    assert failures[0].context["failure_reason"].startswith("WebSocket")


def test_recall_filter_by_context(memory):
    memory.record("speed up api", "add cache", context={"service": "auth"})
    memory.record("speed up api", "add index", context={"service": "billing"})

    results = memory.recall("speed up api", context={"service": "auth"})
    assert len(results) == 1
    assert results[0].action == "add cache"


def test_recall_filter_by_nested_context(memory):
    # Values SQL can't compare still have to filter correctly, in Python.
    memory.record("speed up api", "add cache", context={"tags": ["auth", "cache"]})
    memory.record("speed up api", "add index", context={"tags": ["billing"]})

    results = memory.recall("speed up api", context={"tags": ["auth", "cache"]})
    assert len(results) == 1
    assert results[0].action == "add cache"


def test_recall_hides_superseded_experience(memory):
    old = memory.record("fix login latency", "add a write cache", outcome="failure")
    new = memory.record("fix login latency", "add a read cache", outcome="success")
    memory.supersede(old.id, new.id)

    results = memory.recall("login latency", k=5)
    ids = [e.id for e in results]
    assert new.id in ids
    assert old.id not in ids


def test_recall_can_include_superseded(memory):
    old = memory.record("fix login latency", "add a write cache")
    new = memory.record("fix login latency", "add a read cache")
    memory.supersede(old.id, new.id)

    ids = [e.id for e in memory.recall("login latency", k=5, include_superseded=True)]
    assert old.id in ids and new.id in ids


def hydration_count(memory, query, k):
    """Count how many rows recall turns into Experience objects."""
    storage = memory.storage
    original = storage.row_to_experience
    calls = 0

    def counting(row):
        nonlocal calls
        calls += 1
        return original(row)

    storage.row_to_experience = counting
    try:
        memory.recall(query, k=k)
    finally:
        storage.row_to_experience = original
    return calls


def test_recall_does_not_scale_with_store_size():
    # Recall must bound how many rows it hydrates as the store grows.
    small, big = Mimir(":memory:"), Mimir(":memory:")
    if not small.storage.fts_enabled:
        small.close()
        big.close()
        pytest.skip("FTS5 not available; fallback search scans the full table")
    try:
        for i in range(100):
            small.record(f"fix latency in service {i}", "add cache")
        for i in range(2000):
            big.record(f"fix latency in service {i}", "add cache")

        assert hydration_count(small, "latency service", 5) == hydration_count(
            big, "latency service", 5
        )
    finally:
        small.close()
        big.close()
