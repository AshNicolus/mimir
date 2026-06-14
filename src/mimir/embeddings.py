"""Embedding provider seam.

The default is a no-op: recall works on keywords alone, so v1 has no heavy
dependencies and no API cost. Plug in a real embedder (local or API-backed) to
enable semantic recall — it just needs to implement ``Embedder``.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod


class Embedder(ABC):
    enabled: bool = True

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Return a vector for ``text``."""


class NullEmbedder(Embedder):
    """Does nothing. Recall falls back to keyword search."""

    enabled = False

    def embed(self, text: str) -> list[float]:  # pragma: no cover - never called
        raise RuntimeError("NullEmbedder cannot produce embeddings")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
