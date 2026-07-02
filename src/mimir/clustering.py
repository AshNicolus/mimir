"""Action clustering: groups equivalent actions so their evidence pools
instead of fragmenting into separate recommendations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import NamedTuple

from .embeddings import Embedder, cosine_similarity


class Cluster(NamedTuple):
    key: str
    action: str  # a representative phrasing of the cluster


def normalize_action(action: str) -> str:
    return " ".join(action.lower().split())


class ActionClusterer(ABC):
    @abstractmethod
    def key(self, action: str, known: Callable[[], list[Cluster]]) -> str:
        """Return the cluster key to store for this action. ``known`` lazily
        loads existing clusters so semantic backends can merge into one."""


class ExactClusterer(ActionClusterer):
    """Clusters by normalized text only; the dependency-free default."""

    def key(self, action: str, known: Callable[[], list[Cluster]]) -> str:
        return normalize_action(action)


class EmbeddingClusterer(ActionClusterer):
    """Merges an action into the nearest cluster within the cosine threshold,
    otherwise starts a new one."""

    def __init__(self, embedder: Embedder, threshold: float = 0.85) -> None:
        self.embedder = embedder
        self.threshold = threshold
        self.cache: dict[str, list[float]] = {}

    def key(self, action: str, known: Callable[[], list[Cluster]]) -> str:
        vec = self.embed(action)
        best_key = None
        best_sim = self.threshold
        for cluster in known():
            sim = cosine_similarity(vec, self.embed(cluster.action))
            if sim >= best_sim:
                best_key, best_sim = cluster.key, sim
        return best_key or normalize_action(action)

    def embed(self, text: str) -> list[float]:
        if text not in self.cache:
            self.cache[text] = self.embedder.embed(text)
        return self.cache[text]
