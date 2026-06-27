"""recommend(): ranking, counts, ablation, and action clustering."""

from mimir import Mimir


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
    # Regression: counts used to cap at k=20; must reflect the real total.
    for _ in range(30):
        memory.record("auth is slow", "Redis caching", outcome="success")

    rec = memory.recommend("auth is slow")

    assert rec is not None
    assert rec.success_count == 30
    assert rec.based_on == 30


def test_recommend_returns_none_without_data(memory):
    assert memory.recommend("anything at all") is None


def test_recommend_caps_supporting_ids(memory):
    # supporting_ids is a bounded sample, even when many experiences match,
    # while the counts stay exact over the whole population.
    for _ in range(250):
        memory.record("auth is slow", "Redis caching", outcome="success")

    rec = memory.recommend("auth is slow")
    assert rec is not None
    assert rec.based_on == 250
    assert rec.success_count == 250
    assert 0 < len(rec.supporting_ids) <= 100


def test_recommend_prefers_more_relevant_action(memory):
    # Two actions with identical track records (5/5 success each). The one whose
    # supporting experiences are more relevant to the query should rank higher.
    for _ in range(5):
        memory.record("fix authentication login latency", "add a read cache", outcome="success")
    for _ in range(5):
        memory.record("authentication note", "rewrite the query", outcome="success")

    rec = memory.recommend("authentication login latency is slow")
    assert rec is not None
    assert rec.recommended_action == "add a read cache"


def test_relevance_weighting_is_ablatable(memory):
    # The ablation hook must actually change the estimator: weighting on vs off
    # produces different confidence for the same action.
    for _ in range(5):
        memory.record("authentication note", "rewrite the query", outcome="success")

    on = memory.recommend("authentication login latency is slow", weight_by_relevance=True)
    off = memory.recommend("authentication login latency is slow", weight_by_relevance=False)
    assert on is not None and off is not None
    assert on.recommended_action == off.recommended_action == "rewrite the query"
    assert on.confidence != off.confidence


def test_default_clusterer_keeps_phrasings_separate(memory):
    # The cheap default clusters by exact text, so differently worded phrasings
    # of the same strategy do not merge.
    memory.record("auth is slow", "Added Redis cache", outcome="success")
    memory.record("auth is slow", "use redis caching", outcome="success")

    rec = memory.recommend("auth is slow")
    assert rec is not None
    assert rec.based_on == 1


def test_custom_clusterer_merges_equivalent_actions():
    # A swappable clusterer collapses semantically equivalent but textually
    # different actions into one recommendation with summed counts.
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
    # The shipped embedding backend merges actions whose vectors are close and
    # keeps unrelated ones apart.
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

        stats = m._storage.aggregate_actions("auth is slow")
        keys = {s.key for s in stats}
        assert len(keys) == 2  # the two redis phrasings collapsed into one cluster
        redis = max(stats, key=lambda s: s.total)
        assert redis.total == 2
    finally:
        m.close()


def test_recommend_groups_actions_instead_of_scanning_every_row():
    # recommend aggregates in the backend, so the work it pulls back is bounded
    # by the number of distinct actions, not the size of the store.
    actions = ["add a cache", "add an index", "rewrite the query"]
    small, big = Mimir(":memory:"), Mimir(":memory:")
    try:
        for i in range(60):
            small.record(f"slow service {i}", actions[i % len(actions)], outcome="success")
        for i in range(1200):
            big.record(f"slow service {i}", actions[i % len(actions)], outcome="success")

        small_stats = small._storage.aggregate_actions("slow service")
        big_stats = big._storage.aggregate_actions("slow service")
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
    # An older strategy, even with a winning record, drops out of recommend once
    # superseded; its outcomes no longer count toward any action.
    old = memory.record("auth is slow", "Redis caching", outcome="success")
    memory.record("auth is slow", "rewrite the query", outcome="success")
    memory.supersede(old.id, memory.record("auth is slow", "rewrite the query").id)

    rec = memory.recommend("auth is slow")
    assert rec is not None
    assert rec.recommended_action == "rewrite the query"
    assert old.id not in rec.supporting_ids


def test_recommend_can_include_superseded(memory):
    # Same action on both, so the superseded one and its replacement group
    # together: by default only the replacement counts, but opting in counts both.
    old = memory.record("auth is slow", "Redis caching", outcome="success")
    new = memory.record("auth is slow", "Redis caching", outcome="success")
    memory.supersede(old.id, new.id)

    assert memory.recommend("auth is slow").based_on == 1
    assert memory.recommend("auth is slow", include_superseded=True).based_on == 2
