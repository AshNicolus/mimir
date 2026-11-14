"""A deterministic bucketer and outcome guess: used when no model is
configured, or when the configured one abstains, times out, or fails to
parse. Always succeeds, so a run never stalls on one bad episode.
"""

from __future__ import annotations

from ..models import Outcome

from .models import Bucketed, Episode

FIX_WORDS = ("fix", "bug", "error", "broken", "fail", "crash", "issue")
FEATURE_WORDS = ("add", "implement", "build", "create", "feature", "new")
REFACTOR_WORDS = ("refactor", "clean up", "cleanup", "rename", "reorganize", "restructure")
POSITIVE_WORDS = ("thanks", "works now", "perfect", "great", "looks good", "that works")
ACTION_CHARS = 200
NEGATIVE_WORDS = ("still broken", "doesn't work", "not working", "still fails", "worse")


def heuristic_bucket(episode: Episode) -> Bucketed:
    text = " ".join(t.text.lower() for t in episode.turns if t.role == "user")
    category = "other"
    if any(w in text for w in FIX_WORDS):
        category = "fix"
    elif any(w in text for w in FEATURE_WORDS):
        category = "feature"
    elif any(w in text for w in REFACTOR_WORDS):
        category = "refactor"

    first_user = next((t.text for t in episode.turns if t.role == "user"), "")
    task = first_user[:200].strip() or "unlabeled session"
    action = summarize_action(episode, task)
    outcome = infer_outcome(episode)

    return Bucketed(category=category, task=task, action=action, outcome=outcome)


def summarize_action(episode: Episode, task: str) -> str:
    """What the agent said it did, so the action is specific to this episode.

    Mimir's recommend() pools evidence by action text across all tasks, so any
    constant action ("used tools: Bash") collects every episode's wins and
    dominates. Prefers the assistant's last words before the user's final
    turn: what follows "thanks, works now" is usually a pleasantry.
    """
    turns = episode.turns
    said = [(i, t.text) for i, t in enumerate(turns) if t.role == "assistant" and t.text.strip()]
    last_user = max(
        (i for i, t in enumerate(turns) if t.role == "user" and t.text.strip()), default=-1
    )
    before = [text for i, text in said if i < last_user]
    chosen = (before or [text for _, text in said] or [""])[-1]
    summary = " ".join(chosen.split())[:ACTION_CHARS]
    if summary:
        return summary
    # No assistant text at all: scope the tool list to the task so it can't pool.
    tools_used = sorted({c.name for t in turns for c in t.tool_calls} - {"tool_result"})
    done = f"used tools: {', '.join(tools_used)}" if tools_used else "discussion only"
    return f"{done} (for: {task[:80]})"


def infer_outcome(episode: Episode) -> Outcome:
    """Reads the human's own reaction in the trailing turns, since it is
    genuinely present in the transcript: a stronger signal than guessing."""
    trailing = " ".join(t.text.lower() for t in episode.turns[-2:] if t.role == "user")
    if any(w in trailing for w in NEGATIVE_WORDS):
        return Outcome.FAILURE
    if any(w in trailing for w in POSITIVE_WORDS):
        return Outcome.SUCCESS
    return Outcome.PARTIAL
