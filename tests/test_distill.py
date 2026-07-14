"""record_conversation(): distilling transcripts into experiences."""

import pytest

from mimir import CallableDistiller, Draft, Outcome
from mimir.distill import Distiller

TRANSCRIPT = [
    {"role": "user", "content": "login is timing out under load"},
    {"role": "assistant", "content": "Added a redis cache in front of session lookups."},
    {"role": "user", "content": "works now, thanks"},
]


class FixedDistiller(Distiller):
    name = "fixed"

    def __init__(self, draft):
        self.draft = draft

    def distill(self, messages):
        return self.draft


def make_draft(**overrides):
    fields = {"task": "fix login timeout", "action": "add a redis cache"}
    fields.update(overrides)
    return Draft(**fields)


def test_distilled_conversation_is_recorded_with_provenance(memory):
    exp = memory.record_conversation(
        TRANSCRIPT, distiller=FixedDistiller(make_draft()), outcome="success"
    )
    assert exp.task == "fix login timeout"
    assert exp.action == "add a redis cache"
    assert exp.context["source"] == "transcript"
    assert exp.context["distiller"] == "fixed"
    assert memory.count() == 1


def test_distilled_experience_is_recallable(memory):
    memory.record_conversation(
        TRANSCRIPT, distiller=FixedDistiller(make_draft()), outcome="success"
    )
    assert memory.recall("login timeout")[0].action == "add a redis cache"


def test_ground_truth_overrides_the_draft(memory):
    draft = make_draft(outcome=Outcome.SUCCESS, score=0.9)
    exp = memory.record_conversation(TRANSCRIPT, distiller=FixedDistiller(draft), outcome="failure")
    assert exp.outcome is Outcome.FAILURE
    assert exp.score == 0.0  # the draft's score graded a different verdict


def test_draft_outcome_is_used_without_ground_truth(memory):
    draft = make_draft(outcome=Outcome.FAILURE)
    exp = memory.record_conversation(TRANSCRIPT, distiller=FixedDistiller(draft))
    assert exp.outcome is Outcome.FAILURE
    assert exp.score == 0.0


def test_missing_outcome_raises_instead_of_assuming_success(memory):
    with pytest.raises(ValueError, match="outcome"):
        memory.record_conversation(TRANSCRIPT, distiller=FixedDistiller(make_draft()))
    assert memory.count() == 0


def test_abstaining_distiller_records_nothing(memory):
    result = memory.record_conversation(TRANSCRIPT, distiller=CallableDistiller(lambda m: None))
    assert result is None
    assert memory.count() == 0


def test_same_transcript_replaces_instead_of_duplicating(memory):
    distiller = FixedDistiller(make_draft())
    first = memory.record_conversation(TRANSCRIPT, distiller=distiller, outcome="success")
    second = memory.record_conversation(TRANSCRIPT, distiller=distiller, outcome="failure")
    assert first.id == second.id
    assert memory.count() == 1
    assert memory.get(first.id).outcome is Outcome.FAILURE


def test_caller_context_wins_over_draft_context(memory):
    draft = make_draft(context={"env": "draft", "failure_reason": "cache stampede"})
    exp = memory.record_conversation(
        TRANSCRIPT, distiller=FixedDistiller(draft), outcome="failure", context={"env": "prod"}
    )
    assert exp.context["env"] == "prod"
    assert exp.context["failure_reason"] == "cache stampede"


def test_blank_draft_fields_are_rejected():
    with pytest.raises(ValueError):
        Draft(task="  ", action="add a cache")
