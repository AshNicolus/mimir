"""Storage interface: the seam that lets Mimir swap backends without a rewrite."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

from ..models import Experience


class ActionStat(NamedTuple):
    """Outcome counts for one action across the experiences matching a query.

    Raw counts are exact integers for reporting; the weighted fields sum each
    experience's relevance to the query instead of counting it as 1.
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
        include_superseded: bool = False,
    ) -> list[tuple[Experience, float]]:
        """Return up to k (experience, relevance) pairs, best first, relevance
        in [0, 1]. k=None returns all matches."""

    @abstractmethod
    def vector_search(
        self,
        embedding: list[float],
        k: int | None = 5,
        outcome: str | None = None,
        context: dict | None = None,
        include_superseded: bool = False,
    ) -> list[tuple[Experience, float]]:
        """Return up to k (experience, similarity) pairs by vector similarity,
        best first, over experiences that have an embedding."""

    @abstractmethod
    def aggregate_actions(self, query: str, include_superseded: bool = False) -> list[ActionStat]:
        """Group experiences matching the query by normalized action and return
        per-action outcome counts, without hydrating every row."""

    @abstractmethod
    def supporting_ids(
        self, query: str, action_key: str, limit: int = 100, include_superseded: bool = False
    ) -> list[str]:
        """Return up to limit ids of matching experiences with this action key."""

    @abstractmethod
    def set_superseded_by(self, experience_id: str, superseded_by: str | None) -> bool:
        """Mark an experience as superseded (None clears it). True if it existed."""

    @abstractmethod
    def delete(self, experience_id: str) -> bool:
        """Remove one experience. True if it existed."""

    @abstractmethod
    def recent(self, n: int = 10) -> list[Experience]:
        """Return the n most recent experiences, newest first."""

    @abstractmethod
    def count(self) -> int:
        """Total number of stored experiences."""

    @abstractmethod
    def close(self) -> None:
        """Release underlying resources."""
