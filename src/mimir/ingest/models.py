"""Data types shared across the ingest pipeline.

Turn and Episode are plain internal plumbing, created once per line of a raw
session log, so they stay lightweight dataclasses. Bucketed is the pipeline's
output contract and gets the same validation as mimir.Draft.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from ..models import Outcome

DEFAULT_CATEGORIES = ("feature", "fix", "refactor", "other")


@dataclass
class ToolCall:
    name: str
    preview: str  # a short preview only; the full payload is never kept


@dataclass
class Turn:
    role: str  # "user", "assistant", or "system" for an omission notice
    text: str
    timestamp: datetime
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class Episode:
    source: str
    session_id: str
    turns: list[Turn]

    @property
    def start(self) -> datetime:
        return self.turns[0].timestamp

    @property
    def end(self) -> datetime:
        return self.turns[-1].timestamp


class Bucketed(BaseModel):
    """One episode's classification: which bucket it belongs to and a short
    summary of what happened, in the same shape mimir-learn's Draft expects."""

    category: str
    task: str
    action: str
    outcome: Outcome | None = None
    score: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("category", "task", "action")
    @classmethod
    def not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be empty or whitespace")
        return cleaned
