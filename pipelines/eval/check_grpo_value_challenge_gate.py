#!/usr/bin/env python3
"""Gate V3 on strict accuracy and whether failures are genuinely route-centric."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.planner_grpo_seed_v1.scripts.reward_planner_grpo import normalize_action


# These errors are raised after a model response is received but cannot be
# normalized into a legal Planner decision. They are model failures and must
# remain in the accuracy denominator, rather than invalidating the run itself.
MODEL_OUTPUT_FALLBACK_ERRORS = {"JSONDecodeError", "ValueError"}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def action_of(value: dict[str, Any]) -> str:
    decision_type = str(value.get("decision_type") or "").strip()
    if decision_type in {"end", "clarify"}:
        return decision_type
    return normalize_action(str(value.get("action") or ""))


def expected_actions(case: dict[str, Any]) -> list[str]:
    return [
        action_of(step)
        for step in case.get("expected_decisions", [])
        if isinstance(step, dict)
    ]


def actual_actions(prediction: dict[str, Any]) -> list[str]:
    return [
        action_of(step)
        for step in prediction.get("decisions", [])
        if isinstance(step, dict)
    ]


def reports_and_predictions(aggregate: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, dict[str, Any]]]]:
    reports: list[dict[str, Any]] = []
    predictions: list[dict[str, dict[str, Any]]] = []
    for item in aggregate.get("reports", []):
        reward = load_json(Path(item["reward_report"]))
        pred_rows = load_jsonl(Path(item["prediction_file"]))
        reports.append(reward)
        predictions.append({str(row["case_id"]): row for row in pred_rows})
    if len(reports) != int(aggregate.get("runs") or 0) or not reports:
        raise ValueError("aggregate report coverage does not match run count")
    return reports, predictions


def case_outcomes(
    *, aggregate: dict[str, Any], cases: list[dict[str, Any]]
) -> dict[str, dict[str, bool]]:
    reports, predictions = reports_and_predictions(aggregate)
    reward_maps = [
        {str(row["case_id"]): bool(row.get("passed") is True) for row in report.get("results", [])}
        for report in reports
    ]
    out: dict[str, dict[str, bool]] = {}
    for case in cases:
        case_id = str(case["case_id"])
        expected = expected_actions(case)
        strict_values = [mapping.get(case_id, False) for mapping in reward_maps]
        route_values = [actual_actions(mapping.get(case_id, {})) == expected for mapping in predictions]
        if len(strict_values) != len(reports) or len(route_values) != len(reports):
            raise ValueError(f"incomplete repeat coverage for {case_id}")
        out[case_id] = {
            "strict_all": all(strict_values),
            "route_all": all(route_values),
        }
    return out


def family_stats(
    *, aggregate: dict[str, Any], cases: list[dict[str, Any]]
) -> dict[str, dict[str, float | int]]:
    outcomes = case_outcomes(aggregate=aggregate, cases=cases)
    by_family: dict[str, list[dict[str, bool]]] = defaultdict(list)
    for case in cases:
        by_family[str(case["scenario_id"])].append(outcomes[str(case["case_id"])])
    result: dict[str, dict[str, float | int]] = {}
    for family, values in by_family.items():
        count = len(values)
        strict_pass = sum(item["strict_all"] for item in values)
        route_pass = sum(item["route_all"] for item in values)
        strict_fail = count - strict_pass
        route_fail = count - route_pass
        argument_only = sum(
            (not item["strict_all"]) and item["route_all"] for item in values
        )
        result[family] = {
            "cases": count,
            "strict_pass_all": strict_pass / count,
            "route_pass_all": route_pass / count,
            "route_failure_rate": route_fail / count,
            "strict_failed_cases": strict_fail,
            "route_failed_cases": route_fail,
            "argument_only_failed_cases": argument_only,
            "argument_only_failure_share": argument_only / strict_fail if strict_fail else 0.0,
        }
    return result


def operational_valid(aggregate: dict[str, Any], required_runs: int) -> bool:
    stats = aggregate.get("prediction_stats") if isinstance(aggregate.get("prediction_stats"), dict) else {}
    errors = stats.get("errors_total") if isinstance(stats.get("errors_total"), dict) else {}
    fallback_errors = (
        stats.get("fallback_errors_total")
        if isinstance(stats.get("fallback_errors_total"), dict)
        else {}
    )
    infrastructure_fallbacks = sum(
        int(value)
        for error_type, value in fallback_errors.items()
        if str(error_type) not in MODEL_OUTPUT_FALLBACK_ERRORS
    )
    return (
        int(aggregate.get("runs") or 0) >= required_runs
        and sum(int(value) for value in errors.values()) == 0
        and infrastructure_fallbacks == 0
        and float(stats.get("empty_decisions_mean") or 0.0) == 0.0
        and int(stats.get("retry_length_truncations_total") or 0) == 0
    )


def select_balanced_families(admitted: list[str], scenario_order: list[str]) -> list[str]:
    selected_by_detector: dict[str, list[str]] = {"qwen": [], "rex": []}
    admitted_set = set(admitted)
    for scenario in scenario_order:
        detector = "qwen" if scenario.startswith("qwen_") else "rex"
        if scenario in admitted_set and len(selected_by_detector[detector]) < 4:
            selected_by_detector[detector].append(scenario)
    selected_set = set(selected_by_detector["qwen"] + selected_by_detector["rex"])
    return [scenario for scenario in scenario_order if scenario in selected_set]


def calibration_gate(
    *, base: dict[str, Any], reference: dict[str, Any], cases: list[dict[str, Any]]
) -> dict[str, Any]:
    thresholds = {
        "reference_min": 0.95,
        "base_min": 0.08,
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
        checks = {
            "reference_at_least_min": float(r["strict_pass_all"]) >= thresholds["reference_min"],
            "base_at_least_min": float(b["strict_pass_all"]) >= thresholds["base_min"],
            "base_at_most_max": float(b["strict_pass_all"]) <= thresholds["base_max"],
            "gap_at_least_min": gap >= thresholds["gap_min"],
            "base_route_failure_at_least_min": float(b["route_failure_rate"])
            >= thresholds["base_route_failure_min"],
            "argument_only_failure_share_at_most_max": float(b["argument_only_failure_share"])
            <= thresholds["argument_only_failure_share_max"],
            "run_valid": run_valid,
        }
        if all(checks.values()):
            admitted.append(scenario)
        families[scenario] = {
            "base": b,
            "reference": r,
            "strict_gap": gap,
            "checks": checks,
            "admitted": all(checks.values()),
        }
    selected = select_balanced_families(admitted, scenario_order)
    selected_qwen = [value for value in selected if value.startswith("qwen_")]
    selected_rex = [value for value in selected if value.startswith("rex_")]
    selected_three_step = [
        value
        for value in selected
        if any(
            str(case["scenario_id"]) == value and len(case["expected_decisions"]) == 3
            for case in cases
        )
    ]
    global_checks = {
        "run_valid": run_valid,
        "four_qwen_families_selected": len(selected_qwen) == 4,
        "four_rex_families_selected": len(selected_rex) == 4,
        "at_least_two_three_step_families": len(selected_three_step) >= 2,
    }
    return {
        "schema_version": "1.0",
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
    base_stats = family_stats(aggregate=base, cases=cases)
    ref_stats = family_stats(aggregate=reference, cases=cases)
    base_outcomes = case_outcomes(aggregate=base, cases=cases)
    ref_outcomes = case_outcomes(aggregate=reference, cases=cases)

    def overall(outcomes: dict[str, dict[str, bool]]) -> dict[str, float | int]:
        values = list(outcomes.values())
        count = len(values)
        strict_pass = sum(item["strict_all"] for item in values)
        route_pass = sum(item["route_all"] for item in values)
        strict_fail = count - strict_pass
        route_fail = count - route_pass
        argument_only = sum((not item["strict_all"]) and item["route_all"] for item in values)
        return {
            "cases": count,
            "strict_pass_all": strict_pass / count,
            "route_pass_all": route_pass / count,
            "route_failure_rate": route_fail / count,
            "route_failure_share_of_strict_failures": route_fail / strict_fail if strict_fail else 0.0,
            "argument_only_failure_share": argument_only / strict_fail if strict_fail else 0.0,
        }

    b = overall(base_outcomes)
    r = overall(ref_outcomes)
    gap = float(r["strict_pass_all"]) - float(b["strict_pass_all"])
    family_checks = {
        scenario: {
            "base": base_stats[scenario],
            "reference": ref_stats[scenario],
            "reference_at_least_0_90": float(ref_stats[scenario]["strict_pass_all"]) >= 0.90,
            "base_at_most_0_90": float(base_stats[scenario]["strict_pass_all"]) <= 0.90,
        }
        for scenario in base_stats
    }
    checks = {
        "exactly_600_cases": len(cases) == 600,
        "three_repeats_and_runtime_valid": operational_valid(base, 3)
        and operational_valid(reference, 3),
        "reference_at_least_0_95": float(r["strict_pass_all"]) >= 0.95,
        "base_at_least_0_08": float(b["strict_pass_all"]) >= 0.08,
        "base_at_most_0_80": float(b["strict_pass_all"]) <= 0.80,
        "gap_at_least_0_20": gap >= 0.20,
        "base_route_failure_rate_at_least_0_15": float(b["route_failure_rate"]) >= 0.15,
        "base_route_failures_dominate": float(b["route_failure_share_of_strict_failures"]) >= 0.70,
        "base_argument_only_share_at_most_0_20": float(b["argument_only_failure_share"]) <= 0.20,
        "all_families_reference_supported": all(
            value["reference_at_least_0_90"] for value in family_checks.values()
        ),
        "all_families_base_non_saturated": all(
            value["base_at_most_0_90"] for value in family_checks.values()
        ),
    }
    return {
        "schema_version": "1.0",
        "mode": "confirmation",
        "status": "pass" if all(checks.values()) else "fail",
        "overall": {"base": b, "reference": r, "strict_gap": gap},
        "checks": checks,
        "families": family_checks,
        "note": "Confirmation is accepted or rejected as a whole; no item-level filtering.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["calibration", "confirmation"], required=True)
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
