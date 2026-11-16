"""README code samples must show what the code actually does."""

import contextlib
import io
import re
from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"


def section(heading: str) -> str:
    return README.read_text(encoding="utf-8").split(f"## {heading}", 1)[1].split("\n## ", 1)[0]


def test_see_it_work_demo_output_matches_readme():
    code, shown = re.findall(r"```(?:python|text)\n(.*?)```", section("See it work"), re.S)[:2]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(code, {})
    assert out.getvalue() == shown


def test_the_idea_snippet_matches_its_comments():
    code = re.search(r"```python\n(.*?)```", section("The idea"), re.S)[1]
    ns = {}
    exec(code.replace("Mimir()", 'Mimir(":memory:")'), ns)  # don't write mimir.db
    action = re.search(r'rec\.recommended_action\s+# "(.+)"', code)[1]
    confidence = float(re.search(r"rec\.confidence\s+# ([0-9.]+)", code)[1])
    assert ns["rec"].recommended_action == action
    assert round(ns["rec"].confidence, 2) == confidence
    ns["memory"].close()
