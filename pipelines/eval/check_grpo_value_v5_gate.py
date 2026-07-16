#!/usr/bin/env python3
"""Apply the preregistered V5 calibration or confirmation gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.eval.check_grpo_value_challenge_gate import (
    case_outcomes,
    family_stats,
    load_json,
    load_jsonl,
    operational_valid,
    select_balanced_families,
    write_json,
)
from pipelines.eval.check_grpo_value_v4_gate import action_mix, overall_stats


def calibration_gate(
    *, base: dict[str, Any], reference: dict[str, Any], cases: list[dict[str, Any]]
) -> dict[str, Any]:
    thresholds = {
        "reference_min": 0.95,
        "base_min": 0.0,
        "base_max": 0.80,
        "gap_min": 0.20,
        "base_route_failure_min": 0.15,
        "argument_only_failure_share_max": 0.25,
    }
    base_stats = family_stats(aggregate=base, cases=cases)
    ref_stats = family_stats(aggregate=reference, cases=cases)
    scenario_order = list(dict.fromkeys(str(case["scenario_id"]) for case in cases))
    run_valid = operational_valid(base, 1) and operational_valid(reference, 1)
    families: dict[str, Any] = {}
    admitted: list[str] = []
    for scenario in scenario_order:
        b = base_stats[scenario]
        r = ref_stats[scenario]
        gap = float(r["strict_pass_all"]) - float(b["strict_pass_all"])
        scenario_cases = [row for row in cases if str(row["scenario_id"]) == scenario]
        mix = action_mix(scenario_cases)
        checks = {
            "reference_at_least_min": float(r["strict_pass_all"]) >= thresholds["reference_min"],
            "base_at_least_min": float(b["strict_pass_all"]) >= thresholds["base_min"],
            "base_at_most_max": float(b["strict_pass_all"]) <= thresholds["base_max"],
            "gap_at_least_min": gap + 1e-12 >= thresholds["gap_min"],
            "base_route_failure_at_least_min": float(b["route_failure_rate"])
            >= thresholds["base_route_failure_min"],
            "argument_only_failure_share_at_most_max": float(b["argument_only_failure_share"])
            <= thresholds["argument_only_failure_share_max"],
            "preregistered_action_mix": mix == {"migrate": 12, "retry": 8},
            "run_valid": run_valid,
        }
        admitted_now = all(checks.values())
        if admitted_now:
            admitted.append(scenario)
        families[scenario] = {
            "base": b,
            "reference": r,
            "strict_gap": gap,
            "action_mix": mix,
            "checks": checks,
            "admitted": admitted_now,
        }
    selected = select_balanced_families(admitted, scenario_order)
    global_checks = {
        "exactly_240_calibration_cases": len(cases) == 240,
        "run_valid": run_valid,
        "four_qwen_families_selected": sum(value.startswith("qwen_") for value in selected) == 4,
        "four_rex_families_selected": sum(value.startswith("rex_") for value in selected) == 4,
    }
    return {
        "schema_version": "1.0",
        "dataset_id": "planner_multistep_grpo_value_v5",
        "mode": "calibration",
        "status": "pass" if all(global_checks.values()) else "fail",
        "thresholds": thresholds,
        "global_checks": global_checks,
        "admitted_scenarios": admitted,
        "selected_scenarios": selected,
        "selection_rule": "first four admitted per detector in preregistered scenario order",
        "families": families,
    }


def confirmation_gate(
    *, base: dict[str, Any], reference: dict[str, Any], cases: list[dict[str, Any]]
) -> dict[str, Any]:
    base_family = family_stats(aggregate=base, cases=cases)
    ref_family = family_stats(aggregate=reference, cases=cases)
    b = overall_stats(case_outcomes(aggregate=base, cases=cases))
    r = overall_stats(case_outcomes(aggregate=reference, cases=cases))
    gap = float(r["strict_pass_all"]) - float(b["strict_pass_all"])
    families = {
        scenario: {
            "base": base_family[scenario],
            "reference": ref_family[scenario],
            "reference_at_least_0_90": float(ref_family[scenario]["strict_pass_all"]) >= 0.90,
        }
        for scenario in base_family
    }
    selected = list(base_family)
    checks = {
        "exactly_240_cases": len(cases) == 240,
        "eight_families": len(selected) == 8,
        "four_qwen_families": sum(value.startswith("qwen_") for value in selected) == 4,
        "four_rex_families": sum(value.startswith("rex_") for value in selected) == 4,
        "preregistered_action_mix": action_mix(cases) == {"migrate": 144, "retry": 96},
        "shortcut_ceiling_at_most_0_60": max(action_mix(cases).values()) / len(cases) <= 0.60,
        "one_repeat_and_runtime_valid": operational_valid(base, 1)
        and operational_valid(reference, 1),
        "reference_at_least_0_95": float(r["strict_pass_all"]) >= 0.95,
        "base_at_most_0_80": float(b["strict_pass_all"]) <= 0.80,
        "gap_at_least_0_20": gap + 1e-12 >= 0.20,
        "base_route_failure_rate_at_least_0_15": float(b["route_failure_rate"]) >= 0.15,
        "base_route_failures_dominate": float(b["route_failure_share_of_strict_failures"]) >= 0.70,
        "base_argument_only_share_at_most_0_20": float(b["argument_only_failure_share"]) <= 0.20,
        "all_families_reference_supported": all(
            value["reference_at_least_0_90"] for value in families.values()
        ),
    }
    return {
        "schema_version": "1.0",
        "dataset_id": "planner_multistep_grpo_value_v5",
        "mode": "confirmation",
        "status": "pass" if all(checks.values()) else "fail",
        "overall": {"base": b, "reference": r, "strict_gap": gap},
        "action_mix": action_mix(cases),
        "checks": checks,
        "families": families,
        "note": "Confirmation is accepted or rejected as a whole; no case-level filtering.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("calibration", "confirmation"), required=True)
    parser.add_argument("--base-aggregate", type=Path, required=True)
    parser.add_argument("--reference-aggregate", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--selected-scenarios-out", type=Path)
    args = parser.parse_args()
    base = load_json(args.base_aggregate)
    reference = load_json(args.reference_aggregate)
    cases = load_jsonl(args.cases)
    result = (
        calibration_gate(base=base, reference=reference, cases=cases)
        if args.mode == "calibration"
        else confirmation_gate(base=base, reference=reference, cases=cases)
    )
    if args.selected_scenarios_out:
        if args.mode != "calibration":
            raise ValueError("--selected-scenarios-out is calibration-only")
        args.selected_scenarios_out.parent.mkdir(parents=True, exist_ok=True)
        args.selected_scenarios_out.write_text(
            "\n".join(result["selected_scenarios"]) + "\n", encoding="utf-8"
        )
    result["inputs"] = {
        "base_aggregate": str(args.base_aggregate.resolve()),
        "reference_aggregate": str(args.reference_aggregate.resolve()),
        "cases": str(args.cases.resolve()),
    }
    write_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
