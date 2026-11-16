"""The README's "See it work" demo must print exactly the output it shows."""

import contextlib
import io
import re
from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"


def test_see_it_work_demo_output_matches_readme():
    section = README.read_text(encoding="utf-8").split("## See it work", 1)[1].split("\n## ", 1)[0]
    code, shown = re.findall(r"```(?:python|text)\n(.*?)```", section, re.S)[:2]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(code, {})
    assert out.getvalue() == shown
