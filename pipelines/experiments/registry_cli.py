#!/usr/bin/env python3
"""Manage the CAPA experiment registry and generated reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from capa.experiments.registry import (  # noqa: E402
    RegistryError,
    append_entry,
    atomic_write_jsonl,
    legacy_entry_to_v2,
    load_registry,
    read_json,
    validate_registry,
)
from capa.experiments.reporting import render_current, render_leaderboard  # noqa: E402


REGISTRY = ROOT / "experiments" / "registry.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--registry", type=Path, default=REGISTRY)
    validate.add_argument("--allow-incomplete", action="store_true")

    migrate = subparsers.add_parser("migrate-legacy")
    migrate.add_argument("--legacy", type=Path, default=ROOT / "experiments" / "manifest.jsonl")
    migrate.add_argument("--registry", type=Path, default=REGISTRY)
    migrate.add_argument("--overwrite", action="store_true")

    add = subparsers.add_parser("add")
    add.add_argument("entry", type=Path)
    add.add_argument("--registry", type=Path, default=REGISTRY)
    add.add_argument("--allow-incomplete", action="store_true")

    render = subparsers.add_parser("render")
    render.add_argument("--registry", type=Path, default=REGISTRY)
    render.add_argument("--status", type=Path, default=ROOT / "experiments" / "project_status.json")
    render.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "datasets")
    render.add_argument("--current", type=Path, default=ROOT / "reports" / "CURRENT.md")
    render.add_argument("--leaderboard", type=Path, default=ROOT / "reports" / "leaderboard.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "validate":
        errors = validate_registry(load_registry(args.registry), strict=not args.allow_incomplete)
        if errors:
            raise RegistryError("\n".join(errors))
        print(json.dumps({"status": "ok", "entries": len(load_registry(args.registry))}))
        return
    if args.command == "migrate-legacy":
        if args.registry.exists() and not args.overwrite:
            raise RegistryError(f"refusing to overwrite {args.registry}")
        rows = [legacy_entry_to_v2(row) for row in load_registry(args.legacy)]
        errors = validate_registry(rows, strict=True)
        if errors:
            raise RegistryError("\n".join(errors))
        atomic_write_jsonl(args.registry, rows)
        print(json.dumps({"status": "migrated", "entries": len(rows)}))
        return
    if args.command == "add":
        append_entry(args.registry, read_json(args.entry), strict=not args.allow_incomplete)
        print(json.dumps({"status": "added", "entry": str(args.entry)}))
        return
    if args.command == "render":
        render_current(
            registry_path=args.registry,
            status_path=args.status,
            dataset_root=args.dataset_root,
            output_path=args.current,
        )
        render_leaderboard(registry_path=args.registry, output_path=args.leaderboard)
        print(json.dumps({"status": "rendered", "current": str(args.current)}))


if __name__ == "__main__":
    main()
