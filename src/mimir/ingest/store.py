"""Stores a bucketed episode into a Mimir store via record_conversation, so
the same transcript-hash dedupe and provenance stamping already built into
Mimir apply here too, and re-running ingestion on the same session replaces
rather than duplicates.
"""

from __future__ import annotations

from ..core import Mimir
from ..distill import Distiller, Draft
from ..models import Experience

from .fallback import infer_outcome
from .models import Bucketed, Episode


class PrecomputedDistiller(Distiller):
    """Wraps a classification already computed upstream as a Distiller, so
    record_conversation's dedupe and provenance apply without reclassifying."""

    def __init__(self, bucketed: Bucketed, name: str) -> None:
        self.bucketed = bucketed
        self.name = name

    def distill(self, messages: list[dict]) -> Draft:
        return Draft(
            task=self.bucketed.task,
            action=self.bucketed.action,
            outcome=self.bucketed.outcome,
            score=self.bucketed.score,
            context={"category": self.bucketed.category},
        )


def episode_to_messages(episode: Episode) -> list[dict]:
    return [{"role": turn.role, "content": turn.text} for turn in episode.turns]


def store_episode(
    memory: Mimir, episode: Episode, bucketed: Bucketed, extractor_name: str
) -> Experience | None:
    # We infer the outcome, we do not observe it, so it belongs on the draft
    # (mimir-learn's "best guess" side), never asserted as ground truth.
    resolved = (
        bucketed
        if bucketed.outcome is not None
        else bucketed.model_copy(update={"outcome": infer_outcome(episode)})
    )
    messages = episode_to_messages(episode)
    return memory.record_conversation(
        messages,
        distiller=PrecomputedDistiller(resolved, name=extractor_name),
        # "agent", not "source": record_conversation reserves context["source"]
        # for its own provenance ("transcript") and overwrites it unconditionally.
        context={"agent": episode.source, "session_id": episode.session_id},
    )
