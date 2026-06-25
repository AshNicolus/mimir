"""Action clustering seam.

recommend() groups experiences by a canonical cluster key so the same strategy
in different words accumulates evidence instead of fragmenting into separate
recommendations. The default clusters by exact normalized text (cheap, no
dependencies). An embedding-backed clusterer merges semantically equivalent
phrasings, and any other strategy can implement the same interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import NamedTuple

from .embeddings import Embedder, cosine_similarity


class Cluster(NamedTuple):
    key: str  # the stored cluster key
    action: str  # a representative phrasing of the cluster


def normalize_action(action: str) -> str:
    """Collapse case and whitespace so identical actions group together."""
    return " ".join(action.lower().split())


class ActionClusterer(ABC):
    @abstractmethod
    def key(self, action: str, known: Callable[[], list[Cluster]]) -> str:
        """Return the canonical cluster key to store for ``action``.

        ``known`` lazily loads the clusters already in the store, so a semantic
        backend can merge ``action`` into an existing one. Stateless backends
        ignore it and pay nothing.
        """


class ExactClusterer(ActionClusterer):
    """Clusters by normalized text only. The cheap default: no embeddings, so
    differently worded phrasings of the same strategy stay separate."""

    def key(self, action: str, known: Callable[[], list[Cluster]]) -> str:
        return normalize_action(action)


class EmbeddingClusterer(ActionClusterer):
    """Merges an action into the nearest existing cluster whose representative is
    within ``threshold`` cosine similarity, otherwise starts a new cluster."""

    def __init__(self, embedder: Embedder, threshold: float = 0.85) -> None:
        self._embedder = embedder
        self._threshold = threshold
        self._cache: dict[str, list[float]] = {}

    def key(self, action: str, known: Callable[[], list[Cluster]]) -> str:
        vec = self.embed(action)
        best_key = None
        best_sim = self._threshold
        for cluster in known():
            sim = cosine_similarity(vec, self.embed(cluster.action))
            if sim >= best_sim:
                best_key, best_sim = cluster.key, sim
        return best_key or normalize_action(action)

    def embed(self, text: str) -> list[float]:
        if text not in self._cache:
            self._cache[text] = self._embedder.embed(text)
        return self._cache[text]
