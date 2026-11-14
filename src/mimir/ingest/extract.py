"""Classifies an episode into a bucket with a short task/action summary.

Extractor is the pluggable seam, the same pattern the core uses for Embedder
and Distiller: implement one method, and any backend, local or hosted, works
the same way to the rest of the pipeline. OllamaExtractor takes a plain model
name string, so swapping the small model is a runtime argument, not a new
integration.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable

from .compress import compress
from .models import Bucketed, DEFAULT_CATEGORIES, Episode


class Extractor(ABC):
    @abstractmethod
    def extract(self, episode: Episode) -> Bucketed | None:
        """Classify one episode, or return None to abstain."""


class CallableExtractor(Extractor):
    """Adapts any episode -> Bucketed | None function into an Extractor."""

    def __init__(self, fn: Callable[[Episode], Bucketed | None], name: str = "callable") -> None:
        self.fn = fn
        self.name = name

    def extract(self, episode: Episode) -> Bucketed | None:
        return self.fn(episode)


class OllamaExtractor(Extractor):
    """Talks to any model already pulled in a local Ollama install, by name.

    Never raises: a network error, timeout, or unparseable reply all abstain
    (return None), so the caller can fall back to the heuristic bucketer.
    """

    name = "ollama"

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        timeout: float = 30.0,
        categories: tuple[str, ...] = DEFAULT_CATEGORIES,
        budget_tokens: int = 2000,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.categories = categories
        self.budget_tokens = budget_tokens

    def extract(self, episode: Episode) -> Bucketed | None:
        compact = compress(episode, budget_tokens=self.budget_tokens)
        prompt = build_prompt(compact, self.categories)
        try:
            raw = self.call_ollama(prompt)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            return None
        return parse_response(raw, self.categories)

    def call_ollama(self, prompt: str) -> str:
        body = json.dumps(
            {"model": self.model, "prompt": prompt, "stream": False, "format": "json"}
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}/api/generate", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload.get("response", "")


def build_prompt(episode: Episode, categories: tuple[str, ...]) -> str:
    lines = [
        f"Classify this coding agent session episode into exactly one of: {', '.join(categories)}.",
        'Reply with only JSON: {"category": ..., "task": "one short sentence", '
        '"action": "one short sentence"}.',
        "",
    ]
    for turn in episode.turns:
        tools = ", ".join(c.name for c in turn.tool_calls)
        line = f"{turn.role}: {turn.text}"
        if tools:
            line += f" [tools: {tools}]"
        lines.append(line)
    return "\n".join(lines)


def parse_response(raw: str, categories: tuple[str, ...] = DEFAULT_CATEGORIES) -> Bucketed | None:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        data = json.loads(raw[start : end + 1])
        bucketed = Bucketed(category=data["category"], task=data["task"], action=data["action"])
    except (json.JSONDecodeError, KeyError, ValueError):
        return None
    # A small model can echo the prompt's own placeholder text (e.g. "one
    # short sentence") back as the category instead of picking a real one;
    # treat anything outside the requested set as an abstention, not data.
    if bucketed.category not in categories:
        return None
    return bucketed
