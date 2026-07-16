#!/usr/bin/env python3
"""Combine non-overlapping Planner rollout prediction shards in case-file order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected object")
        rows.append(value)
    return rows


def combine(cases: list[dict[str, Any]], shards: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    case_order = [str(row.get("case_id") or "") for row in cases]
    if not all(case_order) or len(case_order) != len(set(case_order)):
        raise ValueError("case file must contain unique non-empty case IDs")
    expected = set(case_order)
    predictions: dict[str, dict[str, Any]] = {}
    for shard_index, rows in enumerate(shards):
        for row in rows:
            case_id = str(row.get("case_id") or "")
            if case_id not in expected:
                raise ValueError(f"shard {shard_index}: unknown case_id {case_id!r}")
            if case_id in predictions:
                raise ValueError(f"overlapping prediction for {case_id}")
            predictions[case_id] = row
    missing = [case_id for case_id in case_order if case_id not in predictions]
    if missing:
        raise ValueError(f"missing {len(missing)} predictions; first={missing[:5]}")
    return [predictions[case_id] for case_id in case_order]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_jsonl(args.cases)
    shards = [load_jsonl(path) for path in args.predictions]
    combined = combine(cases, shards)
    write_jsonl(args.out, combined)
    print(
        json.dumps(
            {
                "cases": len(cases),
                "shards": len(shards),
                "shard_rows": [len(rows) for rows in shards],
                "out": str(args.out),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
