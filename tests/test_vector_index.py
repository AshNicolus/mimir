"""Tests for the optional sqlite-vec ANN backend.

These exercise the real vector index. They skip when sqlite-vec isn't installed
or this Python's sqlite3 can't load extensions, where recall uses the cosine
fallback covered in test_mimir.py.
"""

import pytest

from mimir import Mimir
from mimir.embeddings import Embedder


class TopicEmbedder(Embedder):
    """Maps text to a topic vector by keyword, so synonyms with no shared words
    still embed to the same vector."""

    TOPICS = {
        0: ("cat", "feline", "kitten"),
        1: ("dog", "canine", "puppy"),
    }

    def embed(self, text):
        t = text.lower()
        vec = [0.0, 0.0, 1.0]
        for axis, words in self.TOPICS.items():
            if any(w in t for w in words):
                vec = [0.0, 0.0, 0.0]
                vec[axis] = 1.0
                break
        return vec


def vec_memory(**kwargs):
    """A Mimir whose store actually loaded the ANN index, or skip."""
    m = Mimir(":memory:", embedder=TopicEmbedder(), **kwargs)
    if not m._storage._vec:
        m.close()
        pytest.skip("sqlite-vec not available; covered by the cosine fallback")
    return m


def test_ann_index_is_used_when_available():
    m = vec_memory()
    try:
        m.record("adopt a feline companion", "visit the shelter", outcome="success")
        assert m._storage._vec_dim == 3
        count = m._storage._conn.execute("SELECT COUNT(*) FROM vec_experiences").fetchone()[0]
        assert count == 1
    finally:
        m.close()


def test_ann_recall_finds_semantic_match_without_keyword_overlap():
    m = vec_memory()
    try:
        m.record("adopt a feline companion", "visit the shelter", outcome="success")
        m.record("buy a canine leash", "go to the pet store", outcome="success")

        results = m.recall("kitten care tips")
        assert results
        assert results[0].task == "adopt a feline companion"
    finally:
        m.close()


def test_ann_vector_search_ranks_by_similarity():
    m = vec_memory()
    try:
        m.record("adopt a feline companion", "visit the shelter")
        m.record("buy a canine leash", "go to the pet store")

        hits = m._storage.vector_search(TopicEmbedder().embed("kitten"), k=2)
        assert [e.task for e, _ in hits][0] == "adopt a feline companion"
        assert all(0.0 < score <= 1.0 for _, score in hits)
    finally:
        m.close()


def test_ann_respects_outcome_filter():
    m = vec_memory()
    try:
        m.record("adopt a feline companion", "visit the shelter", outcome="success")
        m.record_failure("rescue a feline stray", "left food out")

        failures = m.recall("kitten", outcome="failure")
        assert [e.task for e in failures] == ["rescue a feline stray"]
    finally:
        m.close()


def test_ann_hides_superseded():
    m = vec_memory()
    try:
        old = m.record("adopt a feline companion", "old shelter")
        new = m.record("adopt a feline companion", "new shelter", supersedes=old.id)

        ids = [e.id for e in m.recall("kitten")]
        assert new.id in ids and old.id not in ids
    finally:
        m.close()


def test_ann_respects_context_filter():
    m = vec_memory()
    try:
        m.record("adopt a feline companion", "shelter A", context={"region": "north"})
        m.record("adopt a feline companion", "shelter B", context={"region": "south"})

        results = m.recall("kitten", context={"region": "north"})
        assert [e.action for e in results] == ["shelter A"]
    finally:
        m.close()


def test_ann_dropped_on_delete():
    m = vec_memory()
    try:
        exp = m.record("adopt a feline companion", "visit the shelter")
        assert m.delete(exp.id) is True
        count = m._storage._conn.execute("SELECT COUNT(*) FROM vec_experiences").fetchone()[0]
        assert count == 0
        assert m.recall("kitten") == []
    finally:
        m.close()


def test_ann_rerecord_does_not_duplicate():
    m = vec_memory()
    try:
        exp = m.record("adopt a feline companion", "first attempt")
        exp.action = "second attempt"
        m.write(exp)

        count = m._storage._conn.execute("SELECT COUNT(*) FROM vec_experiences").fetchone()[0]
        assert count == 1
        results = m.recall("kitten")
        assert len(results) == 1
        assert results[0].action == "second attempt"
    finally:
        m.close()


def test_ann_backfills_existing_json_embeddings(tmp_path):
    # An embedded store reopened later should rebuild the index from JSON and
    # still answer semantic recall.
    db = str(tmp_path / "vec.db")
    first = Mimir(db, embedder=TopicEmbedder())
    if not first._storage._vec:
        first.close()
        pytest.skip("sqlite-vec not available")
    first.record("adopt a feline companion", "visit the shelter")
    # Drop the index but keep the JSON embeddings, simulating a pre-extension db.
    first._storage._conn.execute("DROP TABLE vec_experiences")
    first._storage._conn.commit()
    first.close()

    reopened = Mimir(db, embedder=TopicEmbedder())
    try:
        assert reopened._storage._vec_dim == 3
        count = reopened._storage._conn.execute(
            "SELECT COUNT(*) FROM vec_experiences"
        ).fetchone()[0]
        assert count == 1
        results = reopened.recall("kitten")
        assert results and results[0].task == "adopt a feline companion"
    finally:
        reopened.close()
