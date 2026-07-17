#!/usr/bin/env python3
"""Apply the preregistered V11 task-support and safety-signal gate."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.planner_grpo_seed_v1.scripts.gate_planner_retry_safe_end_residual_v8_support import (  # noqa: E402
    apply_gate as apply_task_gate,
    check,
    load_json,
    load_jsonl,
)
from training.planner_grpo_seed_v1.scripts.reward_planner_grpo import (  # noqa: E402
    normalize_action,
)


TASK_THRESHOLD_ALIASES = {
    "minimum_primary_nonzero_reward_variance_rate": "minimum_primary_nonzero_task_reward_variance_rate",
    "minimum_control_nonzero_reward_variance_rate": "minimum_control_nonzero_task_reward_variance_rate",
    "minimum_nonzero_reward_variance_groups_per_scenario_detector": "minimum_nonzero_task_reward_variance_groups_per_scenario_detector",
    "minimum_nonzero_reward_variance_rate_per_support_block": "minimum_nonzero_task_reward_variance_rate_per_support_block",
}


def _adapt_task_preregistration(preregistration: dict[str, Any]) -> dict[str, Any]:
    adapted = copy.deepcopy(preregistration)
    spec = adapted["support_audit"]
    spec["include_stability_controls"] = True
    thresholds = spec["hard_gates"]
    for expected, actual in TASK_THRESHOLD_ALIASES.items():
        thresholds[expected] = thresholds[actual]
    return adapted


def apply_gate(
    *,
    preregistration: dict[str, Any],
    data_rows: list[dict[str, Any]],
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = apply_task_gate(
        preregistration=_adapt_task_preregistration(preregistration),
        data_rows=data_rows,
        samples=samples,
    )
    spec = preregistration["support_audit"]
    thresholds = spec["hard_gates"]
    primary_scenarios = set(preregistration["design"]["primary_scenarios"])
    data = {str(row["case_id"]): row for row in data_rows}

    grouped_safety_rewards: dict[str, list[float]] = defaultdict(list)
    grouped_forbidden: dict[str, list[bool]] = defaultdict(list)
    for sample in samples:
        case_id = str(sample.get("case_id") or "")
        row = data[case_id]
        raw_forbidden = row.get("forbidden_actions") or "[]"
        forbidden_values = (
            json.loads(raw_forbidden)
            if isinstance(raw_forbidden, str)
            else raw_forbidden
        )
        forbidden = {
            normalize_action(str(action))
            for action in forbidden_values
            if str(action).strip()
        }
        valid = bool(sample.get("json_valid"))
        action = normalize_action(str(sample.get("actual_action") or ""))
        forbidden_hit = valid and action in forbidden
        grouped_forbidden[case_id].append(forbidden_hit)
        grouped_safety_rewards[case_id].append(
            float(valid and not forbidden_hit)
        )

    group_rows: list[dict[str, Any]] = []
    for case_id, row in data.items():
        safety_values = grouped_safety_rewards.get(case_id, [])
        forbidden_values = grouped_forbidden.get(case_id, [])
        group_rows.append(
            {
                "case_id": case_id,
                "scenario_id": str(row["scenario_id"]),
                "detector_family": str(row["detector_family"]),
                "support_block": str(row.get("support_block") or ""),
                "safety_variance": len(set(safety_values)) > 1,
                "forbidden_samples": sum(forbidden_values),
                "samples": len(safety_values),
            }
        )
    total_samples = sum(group["samples"] for group in group_rows)
    forbidden_samples = sum(group["forbidden_samples"] for group in group_rows)
    forbidden_rate = forbidden_samples / total_samples if total_samples else 0.0
    safety_variance_groups = sum(group["safety_variance"] for group in group_rows)
    primary_safety_variance = {
        scenario: sum(
            group["safety_variance"]
            for group in group_rows
            if group["scenario_id"] == scenario
        )
        for scenario in sorted(primary_scenarios)
    }
    safety_checks = [
        check(
            "forbidden_action_sample_rate_minimum",
            forbidden_rate,
            ">=",
            float(thresholds["minimum_forbidden_action_sample_rate"]),
        ),
        check(
            "forbidden_action_sample_rate_maximum",
            forbidden_rate,
            "<=",
            float(thresholds["maximum_forbidden_action_sample_rate"]),
        ),
        check(
            "safety_variance_groups_overall",
            safety_variance_groups,
            ">=",
            int(thresholds["minimum_safety_variance_groups_overall"]),
        ),
    ]
    for scenario, observed in primary_safety_variance.items():
        safety_checks.append(
            check(
                f"{scenario}_safety_variance_groups",
                observed,
                ">=",
                int(thresholds["minimum_safety_variance_groups_per_primary_scenario"]),
            )
        )
    payload["hard_checks"].extend(safety_checks)
    passed = all(item["passed"] for item in payload["hard_checks"])
    payload["status"] = "pass" if passed else "fail"
    payload["optimizer_authorized"] = passed
    payload["optimizer_scenarios"] = (
        list(preregistration["design"]["primary_scenarios"])
        + list(preregistration["design"]["stability_controls"])
        if passed
        else []
    )
    payload["observed"]["safety"] = {
        "forbidden_action_samples": forbidden_samples,
        "forbidden_action_sample_rate": forbidden_rate,
        "safety_variance_groups": safety_variance_groups,
        "safety_variance_rate": (
            safety_variance_groups / len(group_rows) if group_rows else 0.0
        ),
        "primary_safety_variance_groups": primary_safety_variance,
        "by_scenario": {
            scenario: {
                "groups": sum(group["scenario_id"] == scenario for group in group_rows),
                "forbidden_action_samples": sum(
                    group["forbidden_samples"]
                    for group in group_rows
                    if group["scenario_id"] == scenario
                ),
                "safety_variance_groups": sum(
                    group["safety_variance"]
                    for group in group_rows
                    if group["scenario_id"] == scenario
                ),
            }
            for scenario in sorted({group["scenario_id"] for group in group_rows})
        },
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--step-data", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accepted-scenarios-out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = apply_gate(
        preregistration=load_json(args.preregistration),
        data_rows=load_jsonl(args.step_data),
        samples=load_jsonl(args.samples),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.accepted_scenarios_out.parent.mkdir(parents=True, exist_ok=True)
    args.accepted_scenarios_out.write_text(
        "".join(f"{scenario}\n" for scenario in payload["optimizer_scenarios"]),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
