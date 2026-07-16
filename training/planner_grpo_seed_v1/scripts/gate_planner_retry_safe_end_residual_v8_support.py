#!/usr/bin/env python3
"""Apply the preregistered V8 retry/safe-end stochastic-support gate."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected an object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: every row must be an object")
    return rows


def aggregate(groups: list[dict[str, Any]]) -> dict[str, Any]:
    if not groups:
        return {
            "groups": 0,
            "gold_action_support_rate": 0.0,
            "nonzero_reward_variance_groups": 0,
            "nonzero_reward_variance_rate": 0.0,
            "mean_reward": 0.0,
        }
    return {
        "groups": len(groups),
        "gold_action_support_rate": statistics.fmean(
            float(group["gold_action_supported"]) for group in groups
        ),
        "nonzero_reward_variance_groups": sum(
            float(group["reward_std"]) > 1e-12 for group in groups
        ),
        "nonzero_reward_variance_rate": statistics.fmean(
            float(float(group["reward_std"]) > 1e-12) for group in groups
        ),
        "mean_reward": statistics.fmean(float(group["reward_mean"]) for group in groups),
    }


def check(name: str, observed: float | int, operator: str, threshold: float | int) -> dict[str, Any]:
    if operator == ">=":
        passed = float(observed) >= float(threshold)
    elif operator == "<=":
        passed = float(observed) <= float(threshold)
    elif operator == "==":
        passed = observed == threshold
    else:
        raise ValueError(f"unsupported operator: {operator}")
    return {
        "name": name,
        "observed": observed,
        "operator": operator,
        "threshold": threshold,
        "passed": passed,
    }


def apply_gate(
    *,
    preregistration: dict[str, Any],
    data_rows: list[dict[str, Any]],
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    spec = preregistration["support_audit"]
    thresholds = spec["hard_gates"]
    primary_scenarios = tuple(preregistration["design"]["primary_scenarios"])
    control_scenarios = (
        tuple(preregistration["design"].get("stability_controls") or [])
        if bool(spec.get("include_stability_controls"))
        else ()
    )
    support_scenarios = primary_scenarios + control_scenarios
    expected_groups = int(spec["expected_prompt_groups"])
    expected_samples = int(spec["expected_samples"])
    samples_per_prompt = int(spec["samples_per_prompt"])
    if len(data_rows) != expected_groups:
        raise ValueError(f"expected {expected_groups} step rows, found {len(data_rows)}")
    if {str(row["scenario_id"]) for row in data_rows} != set(support_scenarios):
        raise ValueError("support step data scenario scope does not match preregistration")
    data = {str(row["case_id"]): row for row in data_rows}
    if len(data) != len(data_rows):
        raise ValueError("support step-data case IDs must be unique")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sample_keys: list[tuple[str, int]] = []
    for sample in samples:
        case_id = str(sample.get("case_id") or "")
        if case_id not in data:
            raise KeyError(f"sample references unknown case: {case_id}")
        sample_index = int(sample.get("sample_index"))
        grouped[case_id].append(sample)
        sample_keys.append((case_id, sample_index))
    duplicate_sample_keys = len(sample_keys) - len(set(sample_keys))

    groups: list[dict[str, Any]] = []
    exact_sample_indices = set(range(samples_per_prompt))
    for case_id, row in data.items():
        values = grouped.get(case_id, [])
        rewards = [float(sample["score"]) for sample in values]
        groups.append(
            {
                "case_id": case_id,
                "scenario_id": str(row["scenario_id"]),
                "detector_family": str(row["detector_family"]),
                "support_block": str(row.get("support_block") or ""),
                "samples": len(values),
                "sample_indices_complete": {
                    int(sample["sample_index"]) for sample in values
                }
                == exact_sample_indices,
                "reward_mean": statistics.fmean(rewards) if rewards else 0.0,
                "reward_std": statistics.pstdev(rewards) if rewards else 0.0,
                "gold_action_supported": any(bool(sample["action_match"]) for sample in values),
            }
        )
    complete_groups = sum(
        group["samples"] == samples_per_prompt and group["sample_indices_complete"]
        for group in groups
    )
    by_scenario = {
        scenario: aggregate([group for group in groups if group["scenario_id"] == scenario])
        for scenario in support_scenarios
    }
    by_scenario_detector = {
        f"{scenario}::{detector}": aggregate(
            [
                group
                for group in groups
                if group["scenario_id"] == scenario and group["detector_family"] == detector
            ]
        )
        for scenario in support_scenarios
        for detector in ("qwen", "rex")
    }
    support_blocks = tuple(str(item) for item in spec.get("support_blocks") or [])
    by_support_block = {
        block: aggregate([group for group in groups if group["support_block"] == block])
        for block in support_blocks
    }
    primary = aggregate(
        [group for group in groups if group["scenario_id"] in set(primary_scenarios)]
    )
    controls = aggregate(
        [group for group in groups if group["scenario_id"] in set(control_scenarios)]
    )
    json_valid_rate = (
        statistics.fmean(float(bool(sample["json_valid"])) for sample in samples)
        if samples
        else 0.0
    )
    clipped_rate = (
        statistics.fmean(float(bool(sample["clipped"])) for sample in samples)
        if samples
        else 1.0
    )
    nonfinite_rewards = sum(
        not math.isfinite(float(sample["score"])) for sample in samples
    )
    checks = [
        check("sample_count", len(samples), "==", expected_samples),
        check(
            "complete_prompt_groups",
            complete_groups,
            "==",
            int(thresholds["complete_prompt_groups"]),
        ),
        check("duplicate_sample_keys", duplicate_sample_keys, "==", 0),
        check("nonfinite_rewards", nonfinite_rewards, "==", 0),
        check(
            "json_valid_rate",
            json_valid_rate,
            ">=",
            float(thresholds["minimum_json_valid_rate"]),
        ),
        check(
            "clipped_rate",
            clipped_rate,
            "<=",
            float(thresholds["maximum_clipped_rate"]),
        ),
        check(
            "primary_gold_action_support_rate",
            float(primary["gold_action_support_rate"]),
            ">=",
            float(thresholds["minimum_primary_gold_action_support_rate"]),
        ),
        check(
            "primary_nonzero_reward_variance_rate",
            float(primary["nonzero_reward_variance_rate"]),
            ">=",
            float(thresholds["minimum_primary_nonzero_reward_variance_rate"]),
        ),
    ]
    if control_scenarios:
        checks.extend(
            [
                check(
                    "control_gold_action_support_rate",
                    float(controls["gold_action_support_rate"]),
                    ">=",
                    float(thresholds["minimum_control_gold_action_support_rate"]),
                ),
                check(
                    "control_nonzero_reward_variance_rate",
                    float(controls["nonzero_reward_variance_rate"]),
                    ">=",
                    float(thresholds["minimum_control_nonzero_reward_variance_rate"]),
                ),
            ]
        )
    for scenario in support_scenarios:
        for detector in ("qwen", "rex"):
            metrics = by_scenario_detector[f"{scenario}::{detector}"]
            checks.extend(
                [
                    check(
                        f"{scenario}_{detector}_gold_support_rate",
                        float(metrics["gold_action_support_rate"]),
                        ">=",
                        float(thresholds["minimum_gold_action_support_rate_per_scenario_detector"]),
                    ),
                    check(
                        f"{scenario}_{detector}_nonzero_variance_groups",
                        int(metrics["nonzero_reward_variance_groups"]),
                        ">=",
                        int(thresholds["minimum_nonzero_reward_variance_groups_per_scenario_detector"]),
                    ),
                ]
            )
    if support_blocks:
        expected_groups_per_block = expected_groups // len(support_blocks)
        if expected_groups % len(support_blocks):
            raise ValueError("expected prompt groups are not divisible by support blocks")
        for block in support_blocks:
            metrics = by_support_block[block]
            checks.extend(
                [
                    check(
                        f"support_block_{block}_groups",
                        int(metrics["groups"]),
                        "==",
                        expected_groups_per_block,
                    ),
                    check(
                        f"support_block_{block}_gold_support_rate",
                        float(metrics["gold_action_support_rate"]),
                        ">=",
                        float(thresholds["minimum_gold_action_support_rate_per_support_block"]),
                    ),
                    check(
                        f"support_block_{block}_nonzero_variance_rate",
                        float(metrics["nonzero_reward_variance_rate"]),
                        ">=",
                        float(thresholds["minimum_nonzero_reward_variance_rate_per_support_block"]),
                    ),
                ]
            )
    passed = all(item["passed"] for item in checks)
    return {
        "schema_version": "1.0",
        "study_id": preregistration["study_id"],
        "status": "pass" if passed else "fail",
        "optimizer_authorized": passed,
        "optimizer_scenarios": list(support_scenarios) if passed else [],
        "observed": {
            "samples": len(samples),
            "complete_groups": complete_groups,
            "duplicate_sample_keys": duplicate_sample_keys,
            "nonfinite_rewards": nonfinite_rewards,
            "json_valid_rate": json_valid_rate,
            "clipped_rate": clipped_rate,
            "primary": primary,
            "controls": controls,
            "by_scenario": by_scenario,
            "by_scenario_detector": by_scenario_detector,
            "by_support_block": by_support_block,
        },
        "hard_checks": checks,
        "on_fail": spec["on_fail"],
    }


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
    accepted = payload["optimizer_scenarios"]
    args.accepted_scenarios_out.parent.mkdir(parents=True, exist_ok=True)
    args.accepted_scenarios_out.write_text(
        "".join(f"{scenario}\n" for scenario in accepted), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
