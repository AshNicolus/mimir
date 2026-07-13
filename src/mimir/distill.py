"""Distill finished conversations into experiences.

A distiller summarizes what happened in a transcript; ground truth passed by
the caller always outranks what it inferred, and returning None abstains.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Callable

from pydantic import BaseModel, Field, field_validator

from .models import Outcome


class Draft(BaseModel):
    """A distilled experience awaiting ground truth and provenance."""

    task: str
    action: str
    outcome: Outcome | None = None  # only when the transcript makes it evident
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    context: dict = Field(default_factory=dict)

    @field_validator("task", "action")
    @classmethod
    def not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be empty or whitespace")
        return cleaned


class Distiller(ABC):
    """Turns one finished conversation into a draft experience."""

    name: str = "distiller"  # stored as provenance on every distilled row

    @abstractmethod
    def distill(self, messages: list[dict]) -> Draft | None:
        """Summarize one completed task from a transcript, or None to abstain."""


class CallableDistiller(Distiller):
    """Adapts any ``messages -> Draft | None`` function into a Distiller."""

    def __init__(self, fn: Callable[[list[dict]], Draft | None], name: str = "callable") -> None:
        self.fn = fn
        self.name = name

    def distill(self, messages: list[dict]) -> Draft | None:
        return self.fn(messages)


def transcript_id(messages: list[dict]) -> str:
    """Deterministic id for a transcript, so re-recording replaces the row."""
    canonical = json.dumps(messages, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]
