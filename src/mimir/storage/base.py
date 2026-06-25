"""Storage interface — the seam that lets Mimir swap backends without a rewrite.

v1 ships a SQLite implementation. A Postgres/pgvector backend (for concurrent
multi-agent writes) and adapters for external memory stores can implement this
same interface later.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

from ..models import Experience


class ActionStat(NamedTuple):
    """Outcome counts for one action across all experiences matching a query.

    The grouping (by normalized action) and counting happen in the backend so
    recommend() never has to hydrate the whole matching population.

    The raw counts (success/failure/partial/total) are exact integers for
    reporting. The weighted_* fields sum each experience's relevance to the
    query instead of counting it as 1, so recommend() can rank on how relevant
    the supporting evidence is, not just its success rate.
    """

    action: str  # a representative phrasing of the action
    key: str  # normalized action, the group key
    success: int
    failure: int
    partial: int
    total: int
    weighted_success: float
    weighted_partial: float
    weighted_total: float


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
        k: int | None = 5,
        outcome: str | None = None,
        context: dict | None = None,
    ) -> list[tuple[Experience, float]]:
        """Return up to ``k`` (experience, relevance_score) pairs, best first.

        ``relevance_score`` is in [0, 1]. ``outcome`` and ``context`` are
        optional equality filters. ``k=None`` returns all matches.
        """

    @abstractmethod
    def vector_search(
        self,
        embedding: list[float],
        k: int | None = 5,
        outcome: str | None = None,
        context: dict | None = None,
    ) -> list[tuple[Experience, float]]:
        """Return up to ``k`` (experience, similarity) pairs by vector similarity
        to ``embedding``, best first, over experiences that have an embedding.

        This is the vector half of hybrid recall. The SQLite backend computes
        cosine in Python (no dependency); a real vector-index backend overrides
        this method. ``outcome`` and ``context`` filter as in ``search``.
        """

    @abstractmethod
    def aggregate_actions(self, query: str) -> list[ActionStat]:
        """Group all experiences matching ``query`` by normalized action and
        return outcome counts per action, so recommend() can rank without
        hydrating every matching row."""

    @abstractmethod
    def supporting_ids(self, query: str, action_key: str, limit: int = 100) -> list[str]:
        """Return up to ``limit`` ids of experiences matching ``query`` whose
        normalized action equals ``action_key``."""

    @abstractmethod
    def delete(self, experience_id: str) -> bool:
        """Remove one experience. Returns True if it existed."""

    @abstractmethod
    def recent(self, n: int = 10) -> list[Experience]:
        """Return the ``n`` most recently recorded experiences, newest first."""

    @abstractmethod
    def count(self) -> int:
        """Total number of stored experiences."""

    @abstractmethod
    def close(self) -> None:
        """Release any underlying resources."""
