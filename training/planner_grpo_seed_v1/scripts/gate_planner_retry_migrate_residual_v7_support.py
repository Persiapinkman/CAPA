#!/usr/bin/env python3
"""Apply the preregistered residual V7 stochastic-support gate."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


PRIMARY_ORDER = (
    "fresh_retry_step2",
    "post_retry_success_step3",
    "post_retry_error_step3",
    "post_retry_metric_veto_step3",
    "current_success_step2",
    "conflicting_state_step2",
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: every row must be an object")
    return rows


def aggregate(groups: list[dict[str, Any]]) -> dict[str, Any]:
    if not groups:
        return {
            "groups": 0,
            "nonzero_reward_variance_groups": 0,
            "nonzero_reward_variance_rate": 0.0,
            "gold_action_support_rate": 0.0,
            "fully_saturated_rate": 0.0,
            "mean_reward": 0.0,
        }
    return {
        "groups": len(groups),
        "nonzero_reward_variance_groups": sum(
            float(group["reward_std"]) > 1e-12 for group in groups
        ),
        "nonzero_reward_variance_rate": statistics.fmean(
            float(float(group["reward_std"]) > 1e-12) for group in groups
        ),
        "gold_action_support_rate": statistics.fmean(
            float(bool(group["gold_action_supported"])) for group in groups
        ),
        "fully_saturated_rate": statistics.fmean(
            float(float(group["reward_min"]) >= 0.95) for group in groups
        ),
        "mean_reward": statistics.fmean(float(group["reward_mean"]) for group in groups),
    }


def check(name: str, observed: float | int | bool, operator: str, threshold: float | int | bool) -> dict[str, Any]:
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
    prereg = load_json(args.preregistration)
    spec = prereg["support_audit"]
    thresholds = spec["hard_gates"]
    data_rows = load_jsonl(args.step_data)
    samples = load_jsonl(args.samples)
    data = {str(row["case_id"]): row for row in data_rows}
    expected_groups = int(spec["expected_prompt_groups"])
    expected_samples = int(spec["expected_samples"])
    samples_per_prompt = int(spec["samples_per_prompt"])
    if len(data) != expected_groups:
        raise ValueError(f"expected {expected_groups} step rows, found {len(data)}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        case_id = str(sample.get("case_id") or "")
        if case_id not in data:
            raise KeyError(f"sample references unknown case: {case_id}")
        grouped[case_id].append(sample)
    groups: list[dict[str, Any]] = []
    for case_id, row in data.items():
        values = grouped.get(case_id, [])
        rewards = [float(sample["score"]) for sample in values]
        groups.append(
            {
                "case_id": case_id,
                "scenario_id": str(row["scenario_id"]),
                "optimization_scope": str(row["optimization_scope"]),
                "detector_family": str(row["detector_family"]),
                "target_action_class": str(row["target_action_class"]),
                "samples": len(values),
                "reward_mean": statistics.fmean(rewards) if rewards else 0.0,
                "reward_std": statistics.pstdev(rewards) if rewards else 0.0,
                "reward_min": min(rewards) if rewards else 0.0,
                "reward_max": max(rewards) if rewards else 0.0,
                "gold_action_supported": any(bool(sample["action_match"]) for sample in values),
            }
        )

    complete_groups = sum(group["samples"] == samples_per_prompt for group in groups)
    primary = [group for group in groups if group["scenario_id"] in set(PRIMARY_ORDER)]
    controls = [group for group in groups if group["scenario_id"] not in set(PRIMARY_ORDER)]
    by_scenario = {
        scenario: aggregate([group for group in groups if group["scenario_id"] == scenario])
        for scenario in sorted({str(group["scenario_id"]) for group in groups})
    }
    by_scenario_detector = {
        f"{scenario}::{detector}": aggregate(
            [
                group
                for group in groups
                if group["scenario_id"] == scenario and group["detector_family"] == detector
            ]
        )
        for scenario in sorted({str(group["scenario_id"]) for group in groups})
        for detector in ("qwen", "rex")
    }
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
    primary_metrics = aggregate(primary)
    fresh_by_detector = {
        detector: by_scenario_detector[f"fresh_retry_step2::{detector}"]
        for detector in ("qwen", "rex")
    }
    hard_checks = [
        check("sample_count", len(samples), "==", expected_samples),
        check("complete_prompt_groups", complete_groups, "==", expected_groups),
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
            float(primary_metrics["gold_action_support_rate"]),
            ">=",
            float(thresholds["minimum_primary_gold_action_support_rate"]),
        ),
        check(
            "primary_nonzero_reward_variance_rate",
            float(primary_metrics["nonzero_reward_variance_rate"]),
            ">=",
            float(thresholds["minimum_primary_nonzero_reward_variance_rate"]),
        ),
    ]
    for detector, metrics in fresh_by_detector.items():
        hard_checks.extend(
            [
                check(
                    f"fresh_retry_{detector}_nonzero_groups",
                    int(metrics["nonzero_reward_variance_groups"]),
                    ">=",
                    int(thresholds["minimum_fresh_retry_nonzero_groups_per_detector"]),
                ),
                check(
                    f"fresh_retry_{detector}_gold_support_rate",
                    float(metrics["gold_action_support_rate"]),
                    ">=",
                    float(thresholds["minimum_fresh_retry_gold_support_rate_per_detector"]),
                ),
            ]
        )

    eligible: list[str] = []
    eligibility: dict[str, Any] = {}
    for scenario in PRIMARY_ORDER:
        combined = by_scenario[scenario]
        detectors = {
            detector: by_scenario_detector[f"{scenario}::{detector}"]
            for detector in ("qwen", "rex")
        }
        passed = (
            float(combined["nonzero_reward_variance_rate"]) >= 0.10
            and all(int(metrics["nonzero_reward_variance_groups"]) >= 2 for metrics in detectors.values())
        )
        eligibility[scenario] = {
            "passed": passed,
            "combined": combined,
            "by_detector": detectors,
        }
        if passed:
            eligible.append(scenario)

    passed = all(item["passed"] for item in hard_checks) and "fresh_retry_step2" in eligible
    if not passed:
        eligible = []
    payload = {
        "schema_version": "1.0",
        "study_id": prereg["study_id"],
        "status": "pass" if passed else "fail",
        "optimizer_authorized": passed,
        "optimizer_scenarios": eligible,
        "inputs": {
            "preregistration": str(args.preregistration),
            "step_data": str(args.step_data),
            "samples": str(args.samples),
        },
        "observed": {
            "samples": len(samples),
            "groups": len(groups),
            "complete_groups": complete_groups,
            "json_valid_rate": json_valid_rate,
            "clipped_rate": clipped_rate,
            "primary": primary_metrics,
            "stability_controls": aggregate(controls),
            "by_scenario": by_scenario,
            "by_scenario_detector": by_scenario_detector,
        },
        "hard_checks": hard_checks,
        "adaptive_scenario_eligibility": eligibility,
        "nonfinite_rewards": sum(
            not math.isfinite(float(sample["score"])) for sample in samples
        ),
        "on_fail": spec["on_fail"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.accepted_scenarios_out.parent.mkdir(parents=True, exist_ok=True)
    args.accepted_scenarios_out.write_text(
        "".join(f"{scenario}\n" for scenario in eligible),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
