#!/usr/bin/env python3
"""Combine disjoint GRPO sampling-support audit shards."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.eval.audit_grpo_sampling_support import aggregate  # noqa: E402
from training.planner_grpo_seed_v1.scripts.train_planner_grpo import (  # noqa: E402
    score_step_completion,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dirs = [path if path.is_absolute() else ROOT / path for path in args.input_dir]
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    summaries = [load_json(path / "summary.json") for path in input_dirs]
    reference = summaries[0]
    for summary in summaries[1:]:
        for key in ("model", "adapter_path", "data", "filters", "sampling", "reward_weights"):
            left = reference[key]
            right = summary[key]
            if key == "sampling":
                left = {item: value for item, value in left.items() if item != "seed"}
                right = {item: value for item, value in right.items() if item != "seed"}
            if left != right:
                raise ValueError(f"audit shard mismatch for {key}: {left!r} != {right!r}")
    groups = [row for path in input_dirs for row in load_jsonl(path / "groups.jsonl")]
    samples = [row for path in input_dirs for row in load_jsonl(path / "samples.jsonl")]
    group_keys = {(row["case_id"], int(row["step_index"])) for row in groups}
    if len(group_keys) != len(groups):
        raise ValueError("audit shards overlap in case_id/step_index")
    data_path = Path(reference["data"])
    data_rows = {
        (str(row["case_id"]), int(row["step_index"])): row
        for row in load_jsonl(data_path)
    }
    enriched_by_group: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        key = (str(sample["case_id"]), int(sample["step_index"]))
        expected = data_rows[key]
        sample["task_reward"] = score_step_completion(
            completion=sample["completion"],
            expected_step=expected["expected_step"],
            forbidden_actions=expected.get("forbidden_actions", "[]"),
            reward_spec=expected.get("reward_spec", "{}"),
            previous_action=expected.get("previous_action", ""),
            full_expected_actions=expected.get("full_expected_actions", "[]"),
            step_index=int(expected["step_index"]),
        )
        enriched_by_group[key].append(sample)
    enriched_groups = []
    for group in groups:
        key = (str(group["case_id"]), int(group["step_index"]))
        group_samples = enriched_by_group[key]
        task_rewards = [float(item["task_reward"]) for item in group_samples]
        actions = [str(item["action"]) for item in group_samples]
        enriched_groups.append(
            {
                **group,
                "mean_task_reward": statistics.mean(task_rewards),
                "max_task_reward": max(task_rewards),
                "min_task_reward": min(task_rewards),
                "task_reward_std": statistics.pstdev(task_rewards),
                "action_counts": dict(Counter(actions)),
            }
        )
    payload = {
        "schema_version": "1.0",
        "model": reference["model"],
        "adapter_path": reference["adapter_path"],
        "data": reference["data"],
        "filters": reference["filters"],
        "rows": len(enriched_groups),
        "shards": [str(path) for path in input_dirs],
        "sampling": reference["sampling"],
        "reward_weights": reference["reward_weights"],
        "support": aggregate(enriched_groups),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_jsonl(output_dir / "groups.jsonl", enriched_groups)
    write_jsonl(output_dir / "samples.jsonl", samples)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
