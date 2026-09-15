#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from incremental_kb.analyzers import get_analyzer
from incremental_kb.core import (
    bootstrap_baseline, connect, decide_review, export_kb_registry, ingest,
    list_reviews, publish, rollback, validate_release,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_STATE = ROOT / ".state" / "registry.sqlite3"
BASELINE = ROOT / "output" / "kb_entries_official_v3.jsonl"
KB_REGISTRY = ROOT / "registry" / "kb_revisions.jsonl"


def print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def open_state(path: Path, *, dry_run: bool = False):
    if dry_run and not path.exists():
        db = connect(Path(":memory:"))
    else:
        db = connect(path)
    bootstrap_baseline(db, BASELINE)
    return db


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Incremental offline knowledge pipeline")
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE, help="local SQLite registry")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("init", help="initialize local state from frozen 645-entry baseline")

    ingest_parser = commands.add_parser("ingest", help="plan or ingest new/changed chat Markdown")
    ingest_parser.add_argument("incoming_path", type=Path)
    ingest_parser.add_argument("--dry-run", action="store_true")
    ingest_parser.add_argument("--overlap-messages", type=int, default=30)
    ingest_parser.add_argument("--source-id", help="explicit identity for a renamed and changed single file")
    ingest_parser.add_argument("--analyze", action="store_true", help="allow selected analyzer to run")
    ingest_parser.add_argument("--analyzer", choices=["deterministic", "mock", "openai-compatible"], default="deterministic")

    review_parser = commands.add_parser("review", help="list or decide review items")
    review_parser.add_argument("--status", default="pending")
    decision = review_parser.add_mutually_exclusive_group()
    decision.add_argument("--approve", metavar="REVIEW_ID")
    decision.add_argument("--reject", metavar="REVIEW_ID")
    review_parser.add_argument("--note", default="")

    publish_parser = commands.add_parser("publish", help="validate and atomically publish an immutable snapshot")
    publish_parser.add_argument("--version", required=True)
    publish_parser.add_argument("--embedding-model", default="embedding-3")
    publish_parser.add_argument("--embedding-dimension", type=int, default=2048)

    validate_parser = commands.add_parser("validate-release")
    validate_parser.add_argument("release", type=Path)

    rollback_parser = commands.add_parser("rollback")
    rollback_parser.add_argument("--version")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate-release":
            print_json(validate_release(args.release.resolve()))
            return 0
        if args.command == "rollback":
            print_json({"current_release": rollback(ROOT, args.version)})
            return 0

        db = open_state(args.state.resolve(), dry_run=args.command == "ingest" and args.dry_run)
        if args.command == "init":
            export_kb_registry(db, KB_REGISTRY)
            print_json({"state": str(args.state.resolve()), "baseline_entries": 645, "kb_registry": str(KB_REGISTRY)})
        elif args.command == "ingest":
            if args.overlap_messages < 1:
                raise ValueError("--overlap-messages must be >= 1")
            if args.analyzer != "deterministic" and not args.analyze:
                raise ValueError("external/mock analysis requires explicit --analyze")
            result = ingest(
                db, args.incoming_path, get_analyzer(args.analyzer), dry_run=args.dry_run,
                overlap=args.overlap_messages, explicit_source_id=args.source_id,
            )
            print_json(result)
        elif args.command == "review":
            if args.approve or args.reject:
                result = decide_review(db, args.approve or args.reject, "approved" if args.approve else "rejected", args.note)
                export_kb_registry(db, KB_REGISTRY)
                print_json(result)
            else:
                print_json({"status": args.status, "items": list_reviews(db, args.status)})
        elif args.command == "publish":
            export_kb_registry(db, KB_REGISTRY)
            release = publish(db, ROOT, args.version, embedding_model=args.embedding_model,
                              embedding_dimension=args.embedding_dimension)
            print_json({"published": str(release), "current": release.name})
        return 0
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
