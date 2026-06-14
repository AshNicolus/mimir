"""Core data models for Mimir.

The whole point of Mimir is that it stores *experiences* (problem -> action ->
outcome), not documents. These models are that contract.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex


class Outcome(str, Enum):
    """How an attempt turned out."""

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"


class Experience(BaseModel):
    """A single recorded experience: what was attempted and how it went.

    This is the atomic unit Mimir stores and learns from.
    """

    id: str = Field(default_factory=_new_id)
    task: str  # the problem being solved
    action: str  # what was actually done
    outcome: Outcome = Outcome.SUCCESS
    score: float = Field(default=1.0, ge=0.0, le=1.0)  # quality/confidence of the outcome
    context: dict = Field(default_factory=dict)  # env, tags, agent_id, domain, ...
    embedding: list[float] | None = None  # set only when an embedder is configured
    created_at: datetime = Field(default_factory=_now)
    superseded_by: str | None = None  # for staleness/versioning (future use)

    @field_validator("task", "action")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be empty or whitespace")
        return cleaned

    def text(self) -> str:
        """The text Mimir indexes and embeds for retrieval."""
        return f"{self.task}\n{self.action}"


class Recommendation(BaseModel):
    """A strategy suggested for a new task, derived from past experiences.

    In v1 this is computed on the fly by aggregating recalled experiences
    (no LLM). Later phases will materialize these as first-class Strategy rows.
    """

    task: str  # the query this recommendation answers
    recommended_action: str
    confidence: float = Field(ge=0.0, le=1.0)
    success_count: int = 0
    failure_count: int = 0
    partial_count: int = 0
    based_on: int = 0  # number of experiences considered
    supporting_ids: list[str] = Field(default_factory=list)

    @property
    def total(self) -> int:
        return self.success_count + self.failure_count + self.partial_count

    def __str__(self) -> str:  # nice console output, as shown in the README
        return (
            f"Recommended strategy: {self.recommended_action!r}\n"
            f"  confidence: {self.confidence:.2f}\n"
            f"  based on {self.total} experiences "
            f"({self.success_count} success / {self.failure_count} failure"
            + (f" / {self.partial_count} partial" if self.partial_count else "")
            + ")"
        )
