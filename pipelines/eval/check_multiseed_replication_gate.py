#!/usr/bin/env python3
"""Apply an adaptive, preregistered multi-seed GRPO replication gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_gate(
    arm: dict[str, Any], full: dict[str, Any], primary: dict[str, Any]
) -> dict[str, Any]:
    gate = arm["development_replication_gate"]
    mean_primary = primary["mean_policy"]["comparison"]
    action_delta = float(mean_primary["action_match_delta"])
    action_ci = [float(value) for value in mean_primary["action_match_ci95"]]
    score_delta = float(mean_primary["case_macro_delta"])
    seed_action_deltas = {
        label: float(item["action_match_delta"])
        for label, item in primary["seed_comparisons"].items()
    }
    positive_seed_count = sum(value > 0 for value in seed_action_deltas.values())

    action_mean_passed = action_delta >= gate["minimum_three_seed_mean_action_delta"]
    positive_seeds_passed = positive_seed_count >= gate["minimum_positive_seed_count"]
    action_ci_passed = (
        action_ci[0] > 0
        if gate["require_entity_clustered_action_ci_lower_above_zero"]
        else True
    )
    score_passed = score_delta >= gate["minimum_three_seed_mean_score_delta"]

    category_deltas = full["mean_policy"]["category_deltas"]
    guardrail_deltas = {
        category: float(category_deltas[category])
        for category in gate["guardrail_categories"]
    }
    guardrails_passed = all(
        delta >= -gate["maximum_mean_guardrail_regression"]
        for delta in guardrail_deltas.values()
    )
    anti_shortcut = gate["anti_shortcut_category"]
    anti_shortcut_delta = float(category_deltas[anti_shortcut])
    anti_shortcut_passed = (
        anti_shortcut_delta >= -gate["maximum_mean_anti_shortcut_regression"]
    )

    return {
        "schema_version": "1.0",
        "study_id": arm["study_id"],
        "arm_id": arm["arm_id"],
        "passed": (
            action_mean_passed
            and positive_seeds_passed
            and action_ci_passed
            and score_passed
            and guardrails_passed
            and anti_shortcut_passed
        ),
        "primary": {
            "category": gate["primary_category"],
            "mean_action_delta": action_delta,
            "minimum_mean_action_delta": gate["minimum_three_seed_mean_action_delta"],
            "mean_action_passed": action_mean_passed,
            "action_ci95": action_ci,
            "action_ci_lower_above_zero_passed": action_ci_passed,
            "seed_action_deltas": seed_action_deltas,
            "positive_seed_count": positive_seed_count,
            "minimum_positive_seed_count": gate["minimum_positive_seed_count"],
            "positive_seed_count_passed": positive_seeds_passed,
            "mean_score_delta": score_delta,
            "minimum_mean_score_delta": gate["minimum_three_seed_mean_score_delta"],
            "mean_score_passed": score_passed,
        },
        "guardrails": {
            "category_deltas": guardrail_deltas,
            "maximum_mean_regression": gate["maximum_mean_guardrail_regression"],
            "passed": guardrails_passed,
        },
        "anti_shortcut": {
            "category": anti_shortcut,
            "delta": anti_shortcut_delta,
            "maximum_mean_regression": gate["maximum_mean_anti_shortcut_regression"],
            "passed": anti_shortcut_passed,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--full-comparison", type=Path, required=True)
    parser.add_argument("--primary-comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = evaluate_gate(load(args.arm), load(args.full_comparison), load(args.primary_comparison))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
