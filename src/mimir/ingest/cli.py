"""mimir-ingest: turn raw agent session logs into Mimir experiences.

    mimir-ingest run --model qwen2.5:1.5b
    mimir-ingest run --dry-run
    mimir-ingest report
"""

from __future__ import annotations

import argparse
import sys

from ..core import Mimir
from ..mcp_server import default_db_path  # one store path for both CLIs
from .extract import Extractor, OllamaExtractor
from .fallback import heuristic_bucket
from .readers import claude_code, cursor
from .report import build_report
from .segment import segment
from .store import store_episode

READERS = {
    "claude-code": claude_code.read_all_sessions,
    "cursor": cursor.read_all_sessions,
}


def build_extractor(model: str | None) -> Extractor | None:
    return OllamaExtractor(model=model) if model else None


def run(args: argparse.Namespace) -> None:
    sessions = READERS[args.source]()
    if args.limit:
        sessions = dict(list(sessions.items())[: args.limit])
    extractor = build_extractor(args.model)
    extractor_name = extractor.name if extractor else "heuristic"

    memory = None if args.dry_run else Mimir(args.db)
    before = memory.count() if memory else 0
    classified = skipped = 0
    try:
        for session_id, turns in sessions.items():
            for episode in segment(args.source, session_id, turns):
                bucketed = extractor.extract(episode) if extractor else None
                if bucketed is None:
                    bucketed = heuristic_bucket(episode)
                if args.dry_run:
                    print(f"[{bucketed.category}] {bucketed.task} -> {bucketed.action}")
                    classified += 1
                    continue
                exp = store_episode(memory, episode, bucketed, extractor_name)
                classified += 1 if exp is not None else 0
                skipped += 1 if exp is None else 0
        after = memory.count() if memory else 0
    finally:
        if memory is not None:
            memory.close()
    if args.dry_run:
        print(f"{classified} episodes classified, nothing written (dry run)", file=sys.stderr)
    else:
        # classified can exceed the count delta: re-ingesting a session replaces
        # its earlier row by transcript hash rather than adding a new one.
        added = after - before
        print(f"{classified} classified, {added} new, {skipped} skipped", file=sys.stderr)


def report(args: argparse.Namespace) -> None:
    memory = Mimir(args.db)
    try:
        print(build_report(memory))
    finally:
        memory.close()


def main() -> None:
    # Real session content can hold arbitrary Unicode; a console locked to a
    # narrow codepage (cp1252 on Windows) must not crash a run over it.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(prog="mimir-ingest")
    parser.add_argument("--db", default=default_db_path(), help="Mimir store path")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="ingest raw session logs into the store")
    run_parser.add_argument("--source", choices=sorted(READERS), default="claude-code")
    run_parser.add_argument("--model", default=None, help="Ollama model tag; omit for heuristic")
    run_parser.add_argument("--limit", type=int, default=None, help="cap sessions processed")
    run_parser.add_argument("--dry-run", action="store_true", help="print, don't write anything")
    run_parser.set_defaults(func=run)

    report_parser = sub.add_parser("report", help="print aggregate insights from stored episodes")
    report_parser.set_defaults(func=report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
