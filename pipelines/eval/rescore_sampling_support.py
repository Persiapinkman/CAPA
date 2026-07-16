#!/usr/bin/env python3
"""Rescore saved GRPO samples against a frozen step file and strict actions."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from capa.tools.registry import normalize_tool_action  # noqa: E402
from training.planner_grpo_seed_v1.scripts.train_planner_grpo import (  # noqa: E402
    parse_completion,
    score_step_completion,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _action(value: Any) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return "invalid"
    if not isinstance(value, dict):
        return "invalid"
    decision_type = str(value.get("decision_type") or "")
    if decision_type in {"clarify", "end"}:
        return decision_type
    return normalize_tool_action(str(value.get("action") or "missing_action"))


def _aggregate(groups: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        by_category[str(group["category"])].append(group)
        by_step[int(group["step_index"])].append(group)

    def stats(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "groups": len(items),
            "mean_task_reward": statistics.mean(
                float(item["mean_task_reward"]) for item in items
            ),
            "mean_group_std": statistics.mean(
                float(item["task_reward_std"]) for item in items
            ),
            "nonzero_std_rate": statistics.mean(
                float(item["task_reward_std"] > 1e-6) for item in items
            ),
            "exact_action_support_rate": statistics.mean(
                float(item["exact_action_samples"] > 0) for item in items
            ),
            "near_exact_task_support_rate": statistics.mean(
                float(item["max_task_reward"] >= 0.95) for item in items
            ),
            "fully_saturated_rate": statistics.mean(
                float(item["min_task_reward"] >= 0.95) for item in items
            ),
            "mean_distinct_actions": statistics.mean(
                float(item["distinct_actions"]) for item in items
            ),
        }

    return {
        "overall": stats(groups),
        "categories": {
            key: stats(items) for key, items in sorted(by_category.items())
        },
        "steps": {
            str(key): stats(items) for key, items in sorted(by_step.items())
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = {
        (str(row["case_id"]), int(row["step_index"])): row
        for row in _read_jsonl(args.data)
    }
    samples = _read_jsonl(args.samples)
    rescored: list[dict[str, Any]] = []
    for sample in samples:
        key = (str(sample["case_id"]), int(sample["step_index"]))
        row = data.get(key)
        if row is None:
            raise KeyError(f"sample key not found in step data: {key}")
        completion = str(sample.get("completion") or "")
        parsed = parse_completion(completion)
        expected_action = _action(row["expected_step"])
        actual_action = _action(parsed)
        task_reward = score_step_completion(
            completion=completion,
            expected_step=row["expected_step"],
            forbidden_actions=row.get("forbidden_actions", "[]"),
            reward_spec=row.get("reward_spec", "{}"),
            previous_action=row.get("previous_action", ""),
            full_expected_actions=row.get("full_expected_actions", "[]"),
            step_index=int(row["step_index"]),
            first_json_only=True,
        )
        rescored.append(
            {
                "case_id": key[0],
                "category": str(row["category"]),
                "step_index": key[1],
                "entity_id": str(row.get("entity_id") or ""),
                "sample_index": int(sample.get("sample_index") or 0),
                "expected_action": expected_action,
                "actual_action": actual_action,
                "exact_action_match": actual_action == expected_action,
                "task_reward": task_reward,
            }
        )

    by_group: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rescored:
        by_group[(row["case_id"], row["step_index"])].append(row)
    groups: list[dict[str, Any]] = []
    for (case_id, step_index), items in sorted(by_group.items()):
        rewards = [float(item["task_reward"]) for item in items]
        actions = [str(item["actual_action"]) for item in items]
        groups.append(
            {
                "case_id": case_id,
                "category": items[0]["category"],
                "step_index": step_index,
                "entity_id": items[0]["entity_id"],
                "samples": len(items),
                "mean_task_reward": statistics.mean(rewards),
                "task_reward_std": statistics.pstdev(rewards),
                "min_task_reward": min(rewards),
                "max_task_reward": max(rewards),
                "exact_action_samples": sum(
                    int(item["exact_action_match"]) for item in items
                ),
                "distinct_actions": len(set(actions)),
                "action_counts": dict(Counter(actions)),
            }
        )
    payload = {
        "schema_version": "1.0",
        "samples": str(args.samples),
        "data": str(args.data),
        "sample_rows": len(rescored),
        "groups": len(groups),
        "support": _aggregate(groups),
        "group_results": groups,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["support"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
