"""recommend(): ranking, counts, ablation, action clustering, and time decay."""

import random
from datetime import timedelta

import pytest

from mimir import Experience, Mimir
from mimir.models import utcnow
from mimir.ranking import beta_lower_bound


def test_recommend_prefers_more_proven_action(memory):
    # "Redis caching" succeeds many times; "rewrite in rust" succeeds once.
    for _ in range(9):
        memory.record("auth is slow", "Redis caching", outcome="success")
    memory.record("auth is slow", "Redis caching", outcome="failure")
    memory.record("auth is slow", "rewrite in rust", outcome="success")

    rec = memory.recommend("authentication is slow")

    assert rec is not None
    assert rec.recommended_action == "Redis caching"
    assert rec.success_count == 9
    assert rec.failure_count == 1
    assert 0.0 < rec.confidence <= 1.0


def test_recommend_counts_full_population_not_a_sample(memory):
    for _ in range(30):
        memory.record("auth is slow", "Redis caching", outcome="success")

    rec = memory.recommend("auth is slow")

    assert rec is not None
    assert rec.success_count == 30
    assert rec.based_on == 30


def test_recommend_returns_none_without_data(memory):
    assert memory.recommend("anything at all") is None


def test_recommend_caps_supporting_ids(memory):
    # supporting_ids is a bounded sample; the counts stay exact.
    for _ in range(250):
        memory.record("auth is slow", "Redis caching", outcome="success")

    rec = memory.recommend("auth is slow")
    assert rec is not None
    assert rec.based_on == 250
    assert rec.success_count == 250
    assert 0 < len(rec.supporting_ids) <= 100


def test_recommend_prefers_more_relevant_action(memory):
    # Identical track records; the more relevant evidence should rank higher.
    for _ in range(5):
        memory.record("fix authentication login latency", "add a read cache", outcome="success")
    for _ in range(5):
        memory.record("authentication note", "rewrite the query", outcome="success")

    rec = memory.recommend("authentication login latency is slow")
    assert rec is not None
    assert rec.recommended_action == "add a read cache"


def test_relevance_weighting_changes_which_action_wins(memory):
    # "rewrite the query" is more proven but barely matches; "add a read cache"
    # is less proven but closely matches the query.
    for _ in range(6):
        memory.record("slow batch job cleanup", "rewrite the query", outcome="success")
    for _ in range(3):
        memory.record("login latency is slow under load", "add a read cache", outcome="success")

    on = memory.recommend("login latency is slow", weight_by_relevance=True)
    off = memory.recommend("login latency is slow", weight_by_relevance=False)
    # Weighting picks the relevant action; without it, the more-proven one wins.
    assert on.recommended_action == "add a read cache"
    assert off.recommended_action == "rewrite the query"


def test_confidence_is_an_interpretable_success_rate(memory):
    # 8/8 successes reads high regardless of how the query matches, and weighting
    # does not change the reported confidence, only the ranking.
    for _ in range(8):
        memory.record("api latency under load", "add a redis cache", outcome="success")
    on = memory.recommend("latency spikes", weight_by_relevance=True)
    off = memory.recommend("latency spikes", weight_by_relevance=False)
    assert on.confidence == off.confidence
    assert on.confidence > 0.6


def test_default_clusterer_keeps_phrasings_separate(memory):
    memory.record("auth is slow", "Added Redis cache", outcome="success")
    memory.record("auth is slow", "use redis caching", outcome="success")

    rec = memory.recommend("auth is slow")
    assert rec is not None
    assert rec.based_on == 1


def test_custom_clusterer_merges_equivalent_actions():
    from mimir.clustering import ActionClusterer, normalize_action

    class RedisClusterer(ActionClusterer):
        def key(self, action, known):
            return "redis" if "redis" in action.lower() else normalize_action(action)

    m = Mimir(":memory:", clusterer=RedisClusterer())
    try:
        m.record("auth is slow", "Added Redis cache", outcome="success")
        m.record("auth is slow", "use redis caching", outcome="success")
        m.record("auth is slow", "add a redis layer", outcome="failure")

        rec = m.recommend("auth is slow")
        assert rec is not None
        assert rec.based_on == 3
        assert rec.success_count == 2
        assert rec.failure_count == 1
    finally:
        m.close()


def test_embedding_clusterer_merges_similar_actions():
    from mimir.clustering import EmbeddingClusterer
    from mimir.embeddings import Embedder

    class FakeEmbedder(Embedder):
        def embed(self, text):
            t = text.lower()
            return [1.0, 0.0] if ("redis" in t or "cache" in t or "caching" in t) else [0.0, 1.0]

    m = Mimir(":memory:", clusterer=EmbeddingClusterer(FakeEmbedder(), threshold=0.9))
    try:
        m.record("auth is slow", "Added Redis cache", outcome="success")
        m.record("auth is slow", "use redis caching", outcome="success")
        m.record("auth is slow", "rewrite in rust", outcome="success")

        stats = m.storage.aggregate_actions("auth is slow")
        keys = {s.key for s in stats}
        assert len(keys) == 2  # the two redis phrasings collapsed into one cluster
        redis = max(stats, key=lambda s: s.total)
        assert redis.total == 2
    finally:
        m.close()


def test_recommend_groups_actions_instead_of_scanning_every_row():
    # The rows pulled back are bounded by distinct actions, not store size.
    actions = ["add a cache", "add an index", "rewrite the query"]
    small, big = Mimir(":memory:"), Mimir(":memory:")
    try:
        for i in range(60):
            small.record(f"slow service {i}", actions[i % len(actions)], outcome="success")
        for i in range(1200):
            big.record(f"slow service {i}", actions[i % len(actions)], outcome="success")

        small_stats = small.storage.aggregate_actions("slow service")
        big_stats = big.storage.aggregate_actions("slow service")
        assert len(small_stats) == len(big_stats) == len(actions)

        # Counts stay exact across the full population (1200 / 3 actions).
        rec = big.recommend("slow service")
        assert rec is not None
        assert rec.based_on == 400
        assert rec.success_count == 400
    finally:
        small.close()
        big.close()


def test_recommend_ignores_actions_that_only_failed(memory):
    memory.record_failure("deploy fails", "force push", reason="broke prod")
    assert memory.recommend("deploy fails") is None


def test_recommend_skips_failed_action_for_a_proven_one(memory):
    memory.record_failure("deploy fails", "force push")
    memory.record("deploy fails", "run migrations first", outcome="success")
    rec = memory.recommend("deploy fails")
    assert rec is not None
    assert rec.recommended_action == "run migrations first"


def test_recommendation_str_is_readable(memory):
    memory.record("auth slow", "Redis caching", outcome="success")
    rec = memory.recommend("auth slow")
    text = str(rec)
    assert "Redis caching" in text
    assert "confidence" in text


def test_recommend_excludes_superseded_from_counts(memory):
    old = memory.record("auth is slow", "Redis caching", outcome="success")
    memory.record("auth is slow", "rewrite the query", outcome="success")
    memory.supersede(old.id, memory.record("auth is slow", "rewrite the query").id)

    rec = memory.recommend("auth is slow")
    assert rec is not None
    assert rec.recommended_action == "rewrite the query"
    assert old.id not in rec.supporting_ids


def test_recommend_can_include_superseded(memory):
    # Same action on both, so the pair groups together when superseded rows count.
    old = memory.record("auth is slow", "Redis caching", outcome="success")
    new = memory.record("auth is slow", "Redis caching", outcome="success")
    memory.supersede(old.id, new.id)

    assert memory.recommend("auth is slow").based_on == 1
    assert memory.recommend("auth is slow", include_superseded=True).based_on == 2


def record_aged(memory, task, action, days_ago):
    exp = Experience(
        task=task, action=action, created_at=utcnow() - timedelta(days=days_ago)
    )
    memory.write(exp)


def seed_stale_vs_recent(memory):
    # An old action with more wins, and a fresh action with fewer.
    for _ in range(5):
        record_aged(memory, "api latency", "old approach", days_ago=200)
    for _ in range(2):
        record_aged(memory, "api latency", "new approach", days_ago=2)


def test_recommend_without_decay_prefers_the_larger_sample(memory):
    seed_stale_vs_recent(memory)
    assert memory.recommend("api latency").recommended_action == "old approach"


def test_recommend_with_decay_prefers_the_recent_action(memory):
    seed_stale_vs_recent(memory)
    memory.half_life_days = 30
    assert memory.recommend("api latency").recommended_action == "new approach"


def test_decay_reranks_but_leaves_counts_exact(memory):
    seed_stale_vs_recent(memory)
    memory.half_life_days = 30
    rec = memory.recommend("api latency")
    assert rec.recommended_action == "new approach"
    assert rec.success_count == 2  # decay only reweights ranking, not the counts


def test_decay_works_without_sql_math_functions(memory):
    memory.storage.math_enabled = False  # force the Python decay path
    seed_stale_vs_recent(memory)
    memory.half_life_days = 30
    assert memory.recommend("api latency").recommended_action == "new approach"


def test_decay_works_without_fts(memory):
    memory.storage.fts_enabled = False  # force the no-FTS Python path
    seed_stale_vs_recent(memory)
    memory.half_life_days = 30
    assert memory.recommend("api latency").recommended_action == "new approach"


def seed_proven_vs_newcomer(memory):
    for _ in range(20):
        memory.record("api latency", "proven fix", outcome="success")
    memory.record("api latency", "newcomer fix", outcome="success")
    memory.record("api latency", "broken fix", outcome="failure")


def explore_picks(memory, n=100):
    return [
        memory.recommend("api latency", explore=True, rng=random.Random(i)).recommended_action
        for i in range(n)
    ]


def test_explore_samples_the_underdog_but_never_a_pure_failure(memory):
    seed_proven_vs_newcomer(memory)
    picks = explore_picks(memory)
    assert set(picks) == {"proven fix", "newcomer fix"}


def test_explore_still_favors_the_proven_action(memory):
    seed_proven_vs_newcomer(memory)
    picks = explore_picks(memory)
    assert picks.count("proven fix") > picks.count("newcomer fix")


def test_explore_reports_the_honest_bound_not_the_draw(memory):
    for _ in range(5):
        memory.record("api latency", "proven fix", outcome="success")
    rec = memory.recommend("api latency", explore=True, rng=random.Random(0))
    assert rec.confidence == pytest.approx(beta_lower_bound(5, 0))


def test_default_mode_stays_deterministic(memory):
    seed_proven_vs_newcomer(memory)
    picks = {memory.recommend("api latency").recommended_action for _ in range(5)}
    assert picks == {"proven fix"}
