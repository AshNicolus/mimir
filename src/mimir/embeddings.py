"""Embedding provider seam. The default is a no-op, so v1 needs no heavy
dependencies; plug in any Embedder to enable semantic recall."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

try:
    import numpy
except ImportError:  # optional: part of the embeddings extra
    numpy = None


class Embedder(ABC):
    enabled: bool = True

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Return a vector for the text."""


class NullEmbedder(Embedder):
    """Does nothing; recall stays keyword-only."""

    enabled = False

    def embed(self, text: str) -> list[float]:  # pragma: no cover - never called
        raise RuntimeError("NullEmbedder cannot produce embeddings")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    if numpy is not None:
        va, vb = numpy.asarray(a), numpy.asarray(b)
        norm = numpy.linalg.norm(va) * numpy.linalg.norm(vb)
        return float(va @ vb / norm) if norm else 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
