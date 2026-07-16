#!/usr/bin/env python3
"""Combine disjoint Planner rollout shards with exact case coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: all rows must be JSON objects")
    return rows


def combine(
    *, cases: list[dict[str, Any]], prediction_shards: list[list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    case_ids = [str(row.get("case_id") or "") for row in cases]
    if "" in case_ids or len(case_ids) != len(set(case_ids)):
        raise ValueError("case IDs must be non-empty and unique")
    predictions = [row for shard in prediction_shards for row in shard]
    prediction_ids = [str(row.get("case_id") or "") for row in predictions]
    if "" in prediction_ids or len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("prediction IDs must be non-empty and disjoint across shards")
    expected = set(case_ids)
    observed = set(prediction_ids)
    if observed != expected:
        raise ValueError(
            f"prediction coverage mismatch: missing={sorted(expected - observed)[:3]}, "
            f"extra={sorted(observed - expected)[:3]}"
        )
    indexed = {str(row["case_id"]): row for row in predictions}
    return [indexed[case_id] for case_id in case_ids]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = combine(
        cases=load_jsonl(args.cases),
        prediction_shards=[load_jsonl(path) for path in args.predictions],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
