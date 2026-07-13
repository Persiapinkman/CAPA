#!/usr/bin/env python3
"""Apply a preregistered category/action development gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arm = load(args.arm)
    comparison = load(args.comparison)
    gate = arm["development_gate"]
    baseline = comparison["runs"][args.baseline_label]
    candidate = comparison["runs"][args.candidate_label]

    primary_category = gate["primary_category"]
    primary_category_delta = (
        candidate["categories"][primary_category]
        - baseline["categories"][primary_category]
    )
    primary_category_action_delta = (
        candidate["category_action_match"][primary_category]
        - baseline["category_action_match"][primary_category]
    )
    primary_step = gate.get("primary_step")
    if primary_step is None:
        primary_action_delta = (
            candidate["category_action_match"][primary_category]
            - baseline["category_action_match"][primary_category]
        )
        primary_action_scope = "category_all_steps"
    else:
        primary_step_key = f"{primary_category}#step{primary_step}"
        primary_action_delta = (
            candidate["category_steps"][primary_step_key]["action_match_rate"]
            - baseline["category_steps"][primary_step_key]["action_match_rate"]
        )
        primary_action_scope = f"step_{primary_step}"
    primary_category_passed = (
        primary_category_delta >= gate["minimum_primary_category_delta"]
    )
    primary_action_passed = (
        primary_action_delta >= gate["minimum_primary_action_match_delta"]
    )
    minimum_category_action_delta = gate.get("minimum_primary_category_action_match_delta")
    primary_category_action_passed = (
        minimum_category_action_delta is None
        or primary_category_action_delta >= minimum_category_action_delta
    )

    guardrail_deltas = {
        category: candidate["categories"][category] - baseline["categories"][category]
        for category in gate["guardrail_categories"]
    }
    guardrails_passed = all(
        delta >= -gate["maximum_guardrail_regression"]
        for delta in guardrail_deltas.values()
    )
    anti_shortcut = gate["anti_shortcut_category"]
    anti_shortcut_delta = (
        candidate["categories"][anti_shortcut] - baseline["categories"][anti_shortcut]
    )
    anti_shortcut_passed = (
        anti_shortcut_delta >= -gate["maximum_anti_shortcut_regression"]
    )

    payload = {
        "schema_version": "1.0",
        "study_id": arm["study_id"],
        "arm_id": arm["arm_id"],
        "comparison": str(args.comparison),
        "baseline": args.baseline_label,
        "candidate": args.candidate_label,
        "passed": (
            primary_category_passed
            and primary_action_passed
            and primary_category_action_passed
            and guardrails_passed
            and anti_shortcut_passed
        ),
        "primary": {
            "category": primary_category,
            "category_delta": primary_category_delta,
            "minimum_category_delta": gate["minimum_primary_category_delta"],
            "category_passed": primary_category_passed,
            "category_action_match_delta": primary_category_action_delta,
            "minimum_category_action_match_delta": minimum_category_action_delta,
            "category_action_match_passed": primary_category_action_passed,
            "action_scope": primary_action_scope,
            "step": primary_step,
            "action_match_delta": primary_action_delta,
            "minimum_action_match_delta": gate["minimum_primary_action_match_delta"],
            "action_passed": primary_action_passed,
        },
        "guardrails": {
            "category_deltas": guardrail_deltas,
            "maximum_regression": gate["maximum_guardrail_regression"],
            "passed": guardrails_passed,
        },
        "anti_shortcut": {
            "category": anti_shortcut,
            "delta": anti_shortcut_delta,
            "maximum_regression": gate["maximum_anti_shortcut_regression"],
            "passed": anti_shortcut_passed,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
