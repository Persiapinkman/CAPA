#!/usr/bin/env python3
"""Apply a preregistered GRPO support gate to a sampling audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    study = load(args.study)
    audit = load(args.audit)
    gate = study["support_gate"]
    support = audit["support"]
    category_metrics = support["categories"]
    scope_categories = gate.get("scope_categories") or list(category_metrics)
    scope_steps = [int(value) for value in gate.get("scope_steps") or []]
    if scope_steps:
        category_step_metrics = support["category_steps"]
        scoped_metrics = [
            category_step_metrics[f"{category}::step{step_index}"]
            for category in scope_categories
            for step_index in scope_steps
        ]
    else:
        scoped_metrics = [category_metrics[category] for category in scope_categories]

    def weighted_metric(name: str) -> float:
        if any(name not in item for item in scoped_metrics):
            return 0.0
        denominator = sum(int(item["groups"]) for item in scoped_metrics)
        return (
            sum(float(item[name]) * int(item["groups"]) for item in scoped_metrics) / denominator
            if denominator
            else 0.0
        )

    observed = {
        "nonzero_std_rate": weighted_metric("nonzero_std_rate"),
        "usable_support_rate": weighted_metric("usable_support_rate"),
        "near_exact_task_support_rate": weighted_metric("near_exact_task_support_rate"),
        "exact_task_support_rate": weighted_metric("exact_task_support_rate"),
        "exact_action_support_rate": weighted_metric("exact_action_support_rate"),
        "fully_saturated_rate": weighted_metric("fully_saturated_rate"),
        "mean_distinct_actions": weighted_metric("mean_distinct_actions"),
        "mean_distinct_valid_actions": weighted_metric("mean_distinct_valid_actions"),
    }
    checks = {
        "nonzero_reward_std_rate": {
            "observed": observed["nonzero_std_rate"],
            "operator": ">=",
            "threshold": gate["minimum_nonzero_reward_std_rate"],
            "passed": observed["nonzero_std_rate"]
            >= gate["minimum_nonzero_reward_std_rate"],
        },
        "usable_support_rate": {
            "observed": observed["usable_support_rate"],
            "operator": ">=",
            "threshold": gate["minimum_usable_support_rate"],
            "passed": observed["usable_support_rate"]
            >= gate["minimum_usable_support_rate"],
        },
        "near_exact_task_support_rate": {
            "observed": observed["near_exact_task_support_rate"],
            "operator": ">=",
            "threshold": gate["minimum_near_exact_task_support_rate"],
            "passed": observed["near_exact_task_support_rate"]
            >= gate["minimum_near_exact_task_support_rate"],
        },
        "fully_saturated_rate": {
            "observed": observed["fully_saturated_rate"],
            "operator": "<=",
            "threshold": gate["maximum_fully_saturated_rate"],
            "passed": observed["fully_saturated_rate"]
            <= gate["maximum_fully_saturated_rate"],
        },
    }
    optional_minimums = {
        "exact_task_support_rate": "minimum_exact_task_support_rate",
        "exact_action_support_rate": "minimum_exact_action_support_rate",
        "mean_distinct_actions": "minimum_mean_distinct_actions",
        "mean_distinct_valid_actions": "minimum_mean_distinct_valid_actions",
    }
    for metric, threshold_key in optional_minimums.items():
        if threshold_key not in gate:
            continue
        checks[metric] = {
            "observed": observed[metric],
            "operator": ">=",
            "threshold": gate[threshold_key],
            "passed": observed[metric] >= gate[threshold_key],
        }
    payload = {
        "schema_version": "1.0",
        "study_id": study["study_id"],
        "model": audit["model"],
        "audit": str(args.audit),
        "scope_categories": scope_categories,
        "scope_steps": scope_steps,
        "passed": all(item["passed"] for item in checks.values()),
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
