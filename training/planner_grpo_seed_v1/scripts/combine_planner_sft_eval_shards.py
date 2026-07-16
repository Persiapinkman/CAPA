#!/usr/bin/env python3
"""Combine disjoint Planner SFT generation-evaluation shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--predictions-out", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=260)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: report must be an object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: prediction must be an object")
        rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if len(args.reports) != len(args.predictions):
        raise ValueError("reports and predictions must have the same number of shards")

    reports = [load_json(path) for path in args.reports]
    common_fields = (
        "model_name_or_path",
        "adapter_path",
        "eval_file",
        "score_first_json_only",
        "num_shards",
    )
    for field in common_fields:
        values = {json.dumps(report.get(field), sort_keys=True) for report in reports}
        if len(values) != 1:
            raise ValueError(f"shard reports disagree on {field}: {values}")

    declared_shards = int(reports[0]["num_shards"])
    shard_indices = sorted(int(report["shard_index"]) for report in reports)
    if declared_shards != len(reports) or shard_indices != list(range(declared_shards)):
        raise ValueError(
            f"incomplete shard set: declared={declared_shards}, indices={shard_indices}"
        )

    predictions = [row for path in args.predictions for row in load_jsonl(path)]
    keys = [(str(row["case_id"]), int(row["step_index"])) for row in predictions]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate (case_id, step_index) predictions across shards")
    if len(predictions) != args.expected_rows:
        raise ValueError(f"expected {args.expected_rows} predictions, got {len(predictions)}")
    predictions.sort(key=lambda row: (str(row["case_id"]), int(row["step_index"])))

    # Import after validation so this lightweight tool fails cleanly on malformed inputs.
    from training.planner_grpo_seed_v1.scripts.eval_planner_sft_outputs import summarize

    report = {
        "model_name_or_path": reports[0]["model_name_or_path"],
        "adapter_path": reports[0]["adapter_path"],
        "eval_file": reports[0]["eval_file"],
        "limit": 0,
        "shard_index": None,
        "num_shards": declared_shards,
        "score_first_json_only": reports[0]["score_first_json_only"],
        "combined": True,
        "source_reports": [str(path) for path in args.reports],
        "source_predictions": [str(path) for path in args.predictions],
        **summarize(predictions),
    }
    write_json(args.out, report)
    write_jsonl(args.predictions_out, predictions)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
