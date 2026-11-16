"""server.json must stay publishable: the MCP registry rejects a version that
isn't on PyPI and a name the package README doesn't claim."""

import json
from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")  # stdlib from 3.11

ROOT = Path(__file__).resolve().parent.parent


def test_server_json_matches_package():
    server = json.loads((ROOT / "server.json").read_text())
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    assert server["version"] == version
    assert [p["version"] for p in server["packages"]] == [version]
    assert f"<!-- mcp-name: {server['name']} -->" in (ROOT / "README.md").read_text()
