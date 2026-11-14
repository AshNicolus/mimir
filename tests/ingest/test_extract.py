"""Extractor: the pluggable seam, its prompt/response shape, and abstention."""

from datetime import datetime, timezone

from mimir.ingest.extract import CallableExtractor, OllamaExtractor, build_prompt, parse_response
from mimir.ingest.models import Bucketed, Episode, Turn

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_episode(text: str = "add a health check endpoint") -> Episode:
    turns = [Turn(role="user", text=text, timestamp=NOW)]
    return Episode(source="claude-code", session_id="s1", turns=turns)


def test_callable_extractor_delegates_to_the_function():
    bucketed = Bucketed(category="feature", task="t", action="a")
    extractor = CallableExtractor(lambda ep: bucketed, name="fixed")
    assert extractor.extract(make_episode()) is bucketed
    assert extractor.name == "fixed"


def test_callable_extractor_can_abstain():
    extractor = CallableExtractor(lambda ep: None)
    assert extractor.extract(make_episode()) is None


def test_build_prompt_includes_task_text_and_categories():
    prompt = build_prompt(make_episode(), ("feature", "fix"))
    assert "add a health check endpoint" in prompt
    assert "feature" in prompt and "fix" in prompt


def test_build_prompt_includes_tool_names():
    from mimir.ingest.models import ToolCall

    calls = [ToolCall(name="Edit", preview="x")]
    turns = [Turn(role="assistant", text="done", timestamp=NOW, tool_calls=calls)]
    episode = Episode(source="claude-code", session_id="s1", turns=turns)
    assert "Edit" in build_prompt(episode, ("feature",))


def test_parse_response_reads_clean_json():
    raw = '{"category": "fix", "task": "fix the bug", "action": "patched it"}'
    bucketed = parse_response(raw)
    assert bucketed is not None
    assert bucketed.category == "fix"


def test_parse_response_tolerates_surrounding_prose():
    raw = 'Sure, here you go:\n{"category": "feature", "task": "t", "action": "a"}\nDone!'
    bucketed = parse_response(raw)
    assert bucketed is not None and bucketed.category == "feature"


def test_parse_response_returns_none_on_garbage():
    assert parse_response("not json at all") is None
    assert parse_response("") is None
    assert parse_response('{"category": "fix"}') is None  # missing required fields


def test_parse_response_rejects_a_category_outside_the_requested_set():
    # A small model can echo the prompt's own placeholder text back as the
    # category instead of picking a real one; that must abstain, not stick.
    raw = '{"category": "one short sentence", "task": "t", "action": "a"}'
    assert parse_response(raw, categories=("feature", "fix", "refactor", "other")) is None


def test_ollama_extractor_abstains_when_unreachable():
    # No server is running on this port in the test environment: this must
    # abstain (return None), not raise, so the caller can fall back cleanly.
    extractor = OllamaExtractor(model="does-not-matter", host="http://127.0.0.1:1", timeout=1.0)
    assert extractor.extract(make_episode()) is None
