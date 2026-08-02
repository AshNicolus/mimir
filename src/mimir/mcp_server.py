"""An MCP server that gives any MCP client a shared experience memory.

Claude Code, Codex, Cursor, and other MCP clients can record what they tried and
consult that track record later. Several clients can point at one database: the
store runs in SQLite WAL mode, whose locking is per-process, so concurrent
readers and serialized writers work across separate processes too.

Run it with the ``mimir-mcp`` console script, installed by the ``mcp`` extra.

The tool logic lives in :class:`MimirTools`, which never imports the MCP SDK, so
it can be tested and reused on its own. :func:`build_server` is the thin layer
that exposes those methods as MCP tools.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

from .core import Mimir
from .models import Experience, Outcome, OutcomeScoreWarning, Recommendation

DEFAULT_DB_PATH = Path.home() / ".mimir" / "memory.db"
MAX_RECALL = 50  # keep a runaway k from flooding the client's context
MAX_RECENT = 100
OUTCOMES = tuple(o.value for o in Outcome)


def default_db_path() -> str:
    """Where the shared memory lives: ``MIMIR_DB_PATH`` or ~/.mimir/memory.db.

    One path per machine by default, so every client shares what the others learn.
    """
    return os.environ.get("MIMIR_DB_PATH") or str(DEFAULT_DB_PATH)


def parse_outcome(value: str) -> Outcome:
    # Clients are language models, so name the valid options in the error.
    try:
        return Outcome(value)
    except ValueError:
        raise ValueError(f"outcome must be one of {', '.join(OUTCOMES)}, got {value!r}") from None


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def require_text(field: str, value: str) -> str:
    """Validate a required text field in the caller's terms.

    Pydantic would also catch this, but its message names internal models and
    links to its own docs, which is noise to the language model that has to
    correct the call.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty description, got {value!r}")
    return value


def check_score(score: float | None) -> float | None:
    if score is None:
        return None
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        raise ValueError(f"score must be a number between 0 and 1, got {score!r}")
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"score must be between 0 and 1, got {score}")
    return float(score)


def check_context(context: dict | None) -> dict | None:
    if context is None:
        return None
    if not isinstance(context, dict):
        raise ValueError(f"context must be an object of tags, got {type(context).__name__}")
    return context


def experience_to_dict(exp: Experience, warnings_caught: list | None = None) -> dict[str, Any]:
    # The embedding is deliberately left out: hundreds of floats no client can use.
    payload = {
        "id": exp.id,
        "task": exp.task,
        "action": exp.action,
        "outcome": exp.outcome.value,
        "score": exp.score,
        "context": exp.context,
        "created_at": exp.created_at.isoformat(),
        "superseded": exp.superseded_by is not None,
    }
    if warnings_caught:
        payload["warnings"] = [str(w.message) for w in warnings_caught]
    return payload


def recommendation_to_dict(rec: Recommendation) -> dict[str, Any]:
    return {
        "recommended_action": rec.recommended_action,
        "confidence": round(rec.confidence, 4),
        "success_count": rec.success_count,
        "failure_count": rec.failure_count,
        "partial_count": rec.partial_count,
        "based_on": rec.based_on,
        "supporting_ids": rec.supporting_ids,
    }


class MimirTools:
    """The MCP tool bodies, as plain Python over a Mimir store."""

    def __init__(self, memory: Mimir) -> None:
        self.memory = memory

    def record_experience(
        self,
        task: str,
        action: str,
        outcome: str = "success",
        score: float | None = None,
        context: dict | None = None,
    ) -> dict[str, Any]:
        parsed = parse_outcome(outcome)
        # Warnings go to the server's stderr, which no client ever sees, so
        # catch them and hand them back where the caller can act on them.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", OutcomeScoreWarning)
            exp = self.memory.record(
                task=require_text("task", task),
                action=require_text("action", action),
                outcome=parsed,
                score=check_score(score),
                context=check_context(context),
            )
        return experience_to_dict(exp, warnings_caught=caught)

    def record_failure(
        self,
        task: str,
        action: str,
        reason: str | None = None,
        context: dict | None = None,
    ) -> dict[str, Any]:
        exp = self.memory.record_failure(
            task=require_text("task", task),
            action=require_text("action", action),
            reason=reason,
            context=check_context(context),
        )
        return experience_to_dict(exp)

    def recall_experiences(
        self,
        query: str,
        k: int = 5,
        outcome: str | None = None,
    ) -> dict[str, Any]:
        hits = self.memory.recall(
            query,
            k=clamp(k, 1, MAX_RECALL),
            outcome=parse_outcome(outcome) if outcome else None,
        )
        found = [experience_to_dict(e) for e in hits]
        return {"query": query, "count": len(found), "experiences": found}

    def recommend_action(self, task: str, explore: bool = False) -> dict[str, Any]:
        rec = self.memory.recommend(task, explore=explore)
        if rec is None:
            return {
                "task": task,
                "found": False,
                "detail": "no relevant experience yet; decide freely and record the outcome",
            }
        return {"task": task, "found": True, **recommendation_to_dict(rec)}

    def recent_experiences(self, n: int = 10) -> dict[str, Any]:
        hits = self.memory.recent(clamp(n, 1, MAX_RECENT))
        return {"count": len(hits), "experiences": [experience_to_dict(e) for e in hits]}

    def memory_stats(self) -> dict[str, Any]:
        return {"total_experiences": self.memory.count(), "db_path": self.memory.storage.db_path}


def build_server(tools: MimirTools, name: str = "mimir"):
    """Register the tools on an MCP server and return it.

    Each docstring below is what the client's model reads to decide when to call
    the tool, so they are written as guidance rather than as API notes.
    """
    try:
        from mcp.server import MCPServer
        from mcp.types import ToolAnnotations
    except ImportError:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "the MCP server needs the mcp extra: pip install 'mimir-learn[mcp]'"
        ) from None

    server = MCPServer(
        name=name,
        instructions=(
            "Experience memory shared across sessions and tools. Before starting a "
            "task, call recommend_action to see what has worked and recall_experiences "
            "to see what has failed. After finishing, record what you did and how it "
            "went, so future sessions benefit."
        ),
    )
    read_only = ToolAnnotations(read_only_hint=True)
    writes = ToolAnnotations(read_only_hint=False, destructive_hint=False)

    @server.tool(annotations=writes)
    def record_experience(
        task: str,
        action: str,
        outcome: str = "success",
        score: float | None = None,
        context: dict | None = None,
    ) -> dict[str, Any]:
        """Record what was tried on a task and how it turned out.

        Call this after finishing a task. ``outcome`` is success, failure, or
        partial. ``score`` (0 to 1) defaults from the outcome. Put stable tags
        like repo, language, or service in ``context`` so it can be filtered later.
        """
        return tools.record_experience(task, action, outcome, score, context)

    @server.tool(annotations=writes)
    def record_failure(
        task: str,
        action: str,
        reason: str | None = None,
        context: dict | None = None,
    ) -> dict[str, Any]:
        """Record an approach that did not work, with the reason it failed.

        Worth doing every time: this is what stops a future session repeating
        the same dead end.
        """
        return tools.record_failure(task, action, reason, context)

    @server.tool(annotations=read_only)
    def recall_experiences(query: str, k: int = 5, outcome: str | None = None) -> dict[str, Any]:
        """Find past experiences relevant to a query, most relevant first.

        Pass ``outcome="failure"`` to see only what has gone wrong before, which
        is the fastest way to avoid a known dead end.
        """
        return tools.recall_experiences(query, k, outcome)

    @server.tool(annotations=read_only)
    def recommend_action(task: str, explore: bool = False) -> dict[str, Any]:
        """Suggest the action with the best track record for a task.

        Confidence is a conservative success rate from past outcomes, so it is
        safe to threshold on (for example, reuse above 0.7). Actions that have
        only ever failed are never suggested. Set ``explore`` to sample a
        promising but less proven action instead of always the safest one.
        """
        return tools.recommend_action(task, explore)

    @server.tool(annotations=read_only)
    def recent_experiences(n: int = 10) -> dict[str, Any]:
        """List the most recently recorded experiences, newest first."""
        return tools.recent_experiences(n)

    @server.tool(annotations=read_only)
    def memory_stats() -> dict[str, Any]:
        """Report how many experiences are stored and where the database lives."""
        return tools.memory_stats()

    return server


def main() -> None:
    """Console-script entry point: serve Mimir over stdio."""
    memory = Mimir(db_path=default_db_path())
    try:
        build_server(MimirTools(memory)).run(transport="stdio")
    finally:
        memory.close()


if __name__ == "__main__":
    main()
