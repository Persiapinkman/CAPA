#!/usr/bin/env python3
"""Combine disjoint generation-evaluation shards into one canonical run record."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from capa.experiments.registry import current_git_commit, validate_entry  # noqa: E402
from pipelines.eval.run_generation_eval import aggregate_repeats, utc_now  # noqa: E402
from training.planner_grpo_seed_v1.scripts.eval_planner_sft_outputs import (  # noqa: E402
    load_jsonl,
    summarize,
)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--parent-run-id", default="")
    parser.add_argument("--shard-run-record", action="append", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    return parser.parse_args()


def comparable_signature(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "study_id": record["study_id"],
        "dataset_id": record["data"]["dataset_id"],
        "split": record["data"]["split"],
        "files": record["data"]["files"],
        "sha256": record["data"]["sha256"],
        "total_source_rows": record["data"]["total_source_rows"],
        "model": record["method"]["model"],
        "adapter_path": record["method"]["adapter_path"],
        "prompt_format": record["method"]["prompt_format"],
        "generation": record["method"]["generation"],
        "seed": record["provenance"]["seed"],
    }


def validate_and_sort_shards(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(records) < 2:
        raise ValueError("at least two shard records are required")
    ordered = sorted(records, key=lambda row: int(row["data"].get("offset", -1)))
    reference = comparable_signature(ordered[0])
    expected_offset = 0
    seen_run_ids: set[str] = set()
    for record in ordered:
        if comparable_signature(record) != reference:
            raise ValueError(f"shard metadata mismatch: {record['run_id']}")
        if record["run_id"] in seen_run_ids:
            raise ValueError(f"duplicate shard run id: {record['run_id']}")
        seen_run_ids.add(record["run_id"])
        offset = int(record["data"].get("offset", -1))
        if offset != expected_offset:
            raise ValueError(f"shards are not contiguous: expected offset {expected_offset}, got {offset}")
        expected_offset += int(record["data"]["rows"])
    if expected_offset != int(reference["total_source_rows"]):
        raise ValueError(
            f"shards cover {expected_offset} rows, expected {reference['total_source_rows']}"
        )
    return ordered


def combine_predictions(
    ordered: list[dict[str, Any]], repeat_index: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for record in ordered:
        paths = record["artifacts"]["predictions"]
        if repeat_index >= len(paths):
            raise ValueError(f"missing repeat {repeat_index + 1} in {record['run_id']}")
        for row in load_jsonl(Path(paths[repeat_index])):
            key = (str(row["case_id"]), int(row["step_index"]), int(row["repeat"]))
            if key in seen:
                raise ValueError(f"overlapping prediction key: {key}")
            seen.add(key)
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    shard_paths = [resolve(path) for path in args.shard_run_record]
    ordered = validate_and_sort_shards([load_json(path) for path in shard_paths])
    reference = ordered[0]
    repeat_count = int(reference["method"]["generation"]["repeats"])
    if any(int(row["method"]["generation"]["repeats"]) != repeat_count for row in ordered):
        raise ValueError("shards have different repeat counts")

    run_dir = resolve(args.run_dir)
    artifact_dir = resolve(args.artifact_dir)
    all_predictions: list[list[dict[str, Any]]] = []
    summaries: list[dict[str, Any]] = []
    prediction_paths: list[str] = []
    for repeat_index in range(repeat_count):
        predictions = combine_predictions(ordered, repeat_index)
        summary = summarize(predictions)
        prediction_path = artifact_dir / f"predictions_run{repeat_index + 1}.jsonl"
        write_jsonl(prediction_path, predictions)
        write_json(run_dir / f"metrics_run{repeat_index + 1}.json", summary)
        all_predictions.append(predictions)
        summaries.append(summary)
        prediction_paths.append(str(prediction_path))

    aggregate = aggregate_repeats(summaries, all_predictions)
    runtimes = [float(row["metrics"]["aggregate"].get("runtime_seconds", 0.0)) for row in ordered]
    peaks = [float(row["metrics"]["aggregate"].get("peak_gpu_memory_gb", 0.0)) for row in ordered]
    aggregate["runtime_seconds"] = max(runtimes, default=0.0)
    aggregate["shard_runtime_seconds_sum"] = sum(runtimes)
    aggregate["peak_gpu_memory_gb"] = max(peaks, default=0.0)
    aggregate["shard_count"] = len(ordered)
    write_json(run_dir / "metrics.json", aggregate)

    command = " ".join(shlex.quote(value) for value in [sys.executable, *sys.argv])
    started_at = min(str(row["provenance"]["started_at"]) for row in ordered)
    finished_at = max(str(row["provenance"]["finished_at"]) for row in ordered)
    record = {
        "schema_version": "2.0",
        "run_id": args.run_id,
        "study_id": reference["study_id"],
        "date": started_at[:10],
        "kind": "eval_generation_repeated",
        "status": "completed",
        "purpose": args.purpose,
        "hypothesis": args.hypothesis,
        "parent_run_id": args.parent_run_id or None,
        "provenance": {
            "git_commit": current_git_commit(ROOT),
            "git_dirty": True,
            "command": command,
            "seed": reference["provenance"]["seed"],
            "started_at": started_at,
            "finished_at": finished_at,
            "environment": {
                **reference["provenance"]["environment"],
                "sharded": True,
                "shard_count": len(ordered),
                "shard_cuda_visible_devices": [
                    row["provenance"]["environment"].get("cuda_visible_devices", "")
                    for row in ordered
                ],
            },
            "shard_run_ids": [row["run_id"] for row in ordered],
            "environments": [row["provenance"]["environment"] for row in ordered],
        },
        "data": {
            **reference["data"],
            "rows": sum(int(row["data"]["rows"]) for row in ordered),
            "offset": 0,
        },
        "method": reference["method"],
        "metrics": {
            "primary": {
                "name": "step_mean_verifier_score",
                "value": aggregate["overall"]["mean_score"]["mean"],
                "higher_is_better": True,
            },
            "aggregate": aggregate,
        },
        "artifacts": {
            "run_dir": str(run_dir),
            "metrics": str(run_dir / "metrics.json"),
            "predictions": prediction_paths,
            "shard_run_records": [str(path) for path in shard_paths],
        },
        "decision": {
            "outcome": "pending_comparison",
            "rationale": "Sharded run completed; paired study comparison has not yet been applied.",
        },
    }
    validation_errors = validate_entry(record, strict=True)
    if validation_errors:
        raise ValueError("invalid combined run record: " + "; ".join(validation_errors))
    write_json(run_dir / "run_record.json", record)
    write_json(
        run_dir / "config.json",
        {
            "run_id": args.run_id,
            "study_id": reference["study_id"],
            "shard_run_records": [str(path) for path in shard_paths],
            "rows": record["data"]["rows"],
            "repeats": repeat_count,
        },
    )
    print(json.dumps({"status": "combined", "run_record": str(run_dir / "run_record.json")}))


if __name__ == "__main__":
    main()
