"""Storage interface — the seam that lets Mimir swap backends without a rewrite.

v1 ships a SQLite implementation. A Postgres/pgvector backend (for concurrent
multi-agent writes) and adapters for external memory stores can implement this
same interface later.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Experience


class Storage(ABC):
    @abstractmethod
    def add(self, exp: Experience) -> None:
        """Persist a single experience."""

    @abstractmethod
    def get(self, experience_id: str) -> Experience | None:
        """Fetch one experience by id, or None."""

    @abstractmethod
    def search(
        self,
        query: str,
        k: int = 5,
        outcome: str | None = None,
        context: dict | None = None,
    ) -> list[tuple[Experience, float]]:
        """Return up to ``k`` (experience, relevance_score) pairs, best first.

        ``relevance_score`` is in [0, 1]. ``outcome`` and ``context`` are
        optional equality filters applied before ranking.
        """

    @abstractmethod
    def count(self) -> int:
        """Total number of stored experiences."""

    @abstractmethod
    def close(self) -> None:
        """Release any underlying resources."""
