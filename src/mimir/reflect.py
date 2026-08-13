"""Synthesize patterns across accumulated experiences.

A reflector looks at a set of experiences and writes down what they have in
common; returning None abstains. Reflections are derived knowledge: always
rebuildable from the raw experiences, never a source of truth themselves.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from .models import Experience, new_id, utcnow


class ReflectionDraft(BaseModel):
    """What a reflector produces: the synthesis, nothing about provenance."""

    summary: str
    pattern: str

    @field_validator("summary", "pattern")
    @classmethod
    def not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be empty or whitespace")
        return cleaned


class Reflection(BaseModel):
    """A synthesized pattern across a set of experiences."""

    id: str = Field(default_factory=new_id)
    summary: str
    pattern: str
    supporting_experience_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)


class Reflector(ABC):
    """Turns a set of experiences into a reflection."""

    name: str = "reflector"  # stored as provenance is left to the caller

    @abstractmethod
    def reflect(self, experiences: list[Experience]) -> ReflectionDraft | None:
        """Synthesize a pattern across these experiences, or None to abstain."""


class CallableReflector(Reflector):
    """Adapts any ``experiences -> ReflectionDraft | None`` function into a Reflector."""

    def __init__(
        self, fn: Callable[[list[Experience]], ReflectionDraft | None], name: str = "callable"
    ) -> None:
        self.fn = fn
        self.name = name

    def reflect(self, experiences: list[Experience]) -> ReflectionDraft | None:
        return self.fn(experiences)


def reflection_id(experience_ids: list[str]) -> str:
    """Deterministic id from the supporting experiences, so reflecting on an
    unchanged set replaces the earlier reflection instead of duplicating it."""
    canonical = ",".join(sorted(experience_ids))
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]
