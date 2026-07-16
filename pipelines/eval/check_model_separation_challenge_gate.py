#!/usr/bin/env python3
"""Gate a Planner challenge on 35B competence and train-base difficulty.

Calibration admits or rejects complete scenario families.  Confirmation evaluates
the frozen dataset as a whole and never emits an item-level selection list.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected object")
        rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def metric(report: dict[str, Any], key: str) -> float:
    value = report.get("aggregate", {}).get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"aggregate.{key} missing")
    return float(value)


def family_metric(report: dict[str, Any], family: str, key: str) -> float:
    value = report.get("by_category", {}).get(family, {}).get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"by_category.{family}.{key} missing")
    return float(value)


def error_count(report: dict[str, Any]) -> int:
    values = report.get("prediction_stats", {}).get("errors_total", {})
    if not isinstance(values, dict):
        return 0
    return sum(int(value) for value in values.values() if isinstance(value, (int, float)))


def empty_count(report: dict[str, Any]) -> float:
    value = report.get("prediction_stats", {}).get("empty_decisions_mean", 0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def first_truncation_count(report: dict[str, Any]) -> int:
    stats = report.get("prediction_stats", {})
    if not isinstance(stats, dict):
        return 0
    return int(stats.get("first_length_truncations_total") or 0)


def unrecovered_truncation_count(report: dict[str, Any]) -> int:
    stats = report.get("prediction_stats", {})
    if not isinstance(stats, dict):
        return 0
    return int(stats.get("retry_length_truncations_total") or 0)


def family_order_and_steps(cases: list[dict[str, Any]]) -> tuple[list[str], dict[str, int]]:
    order: list[str] = []
    max_steps: dict[str, int] = {}
    for row in cases:
        family = str(row.get("scenario_id") or row.get("category") or "").strip()
        if family and family not in order:
            order.append(family)
        expected = row.get("expected_decisions") if isinstance(row.get("expected_decisions"), list) else []
        max_steps[family] = max(max_steps.get(family, 0), len(expected))
    return order, max_steps


def calibration_gate(
    *,
    base: dict[str, Any],
    reference: dict[str, Any],
    cases: list[dict[str, Any]],
    reference_min: float,
    base_max: float,
    gap_min: float,
    min_families: int,
    min_multistep_families: int,
) -> dict[str, Any]:
    order, max_steps = family_order_and_steps(cases)
    invalid_run = (
        error_count(base) > 0
        or error_count(reference) > 0
        or empty_count(base) > 0
        or empty_count(reference) > 0
        or unrecovered_truncation_count(base) > 0
        or unrecovered_truncation_count(reference) > 0
    )
    families: dict[str, Any] = {}
    accepted: list[str] = []
    for family in order:
        base_rate = family_metric(base, family, "pass_all_runs_rate")
        reference_rate = family_metric(reference, family, "pass_all_runs_rate")
        gap = reference_rate - base_rate
        checks = {
            "reference_at_least_min": reference_rate >= reference_min,
            "base_at_most_max": base_rate <= base_max,
            "gap_at_least_min": gap >= gap_min,
            "run_valid": not invalid_run,
        }
        admitted = all(checks.values())
        if admitted:
            accepted.append(family)
        families[family] = {
            "reference_pass_all": reference_rate,
            "base_pass_all": base_rate,
            "gap": gap,
            "max_steps": max_steps.get(family, 0),
            "checks": checks,
            "admitted": admitted,
        }
    multi = [family for family in accepted if max_steps.get(family, 0) > 1]
    global_checks = {
        "run_valid": not invalid_run,
        "accepted_family_count": len(accepted) >= min_families,
        "accepted_multistep_family_count": len(multi) >= min_multistep_families,
    }
    return {
        "mode": "calibration",
        "status": "pass" if all(global_checks.values()) else "fail",
        "thresholds": {
            "reference_min": reference_min,
            "base_max": base_max,
            "gap_min": gap_min,
            "min_families": min_families,
            "min_multistep_families": min_multistep_families,
        },
        "global_checks": global_checks,
        "accepted_scenarios": accepted,
        "accepted_multistep_scenarios": multi,
        "families": families,
    }


def confirmation_gate(
    *,
    base: dict[str, Any],
    reference: dict[str, Any],
    cases: list[dict[str, Any]],
    reference_min: float,
    base_max: float,
    gap_min: float,
    family_reference_min: float,
    family_base_max: float,
    expected_cases: int,
    required_runs: int,
) -> dict[str, Any]:
    order, _ = family_order_and_steps(cases)
    base_rate = metric(base, "pass_all_runs_rate")
    reference_rate = metric(reference, "pass_all_runs_rate")
    gap = reference_rate - base_rate
    family_results: dict[str, Any] = {}
    for family in order:
        base_family = family_metric(base, family, "pass_all_runs_rate")
        reference_family = family_metric(reference, family, "pass_all_runs_rate")
        family_results[family] = {
            "reference_pass_all": reference_family,
            "base_pass_all": base_family,
            "gap": reference_family - base_family,
            "reference_at_least_min": reference_family >= family_reference_min,
            "base_at_most_max": base_family <= family_base_max,
        }
    checks = {
        "expected_case_count": len(cases) == expected_cases,
        "required_repeats": int(base.get("runs") or 0) >= required_runs and int(reference.get("runs") or 0) >= required_runs,
        "no_errors": error_count(base) == 0 and error_count(reference) == 0,
        "no_empty_decisions": empty_count(base) == 0 and empty_count(reference) == 0,
        "no_unrecovered_length_truncations": (
            unrecovered_truncation_count(base) == 0
            and unrecovered_truncation_count(reference) == 0
        ),
        "reference_at_least_min": reference_rate >= reference_min,
        "base_at_most_max": base_rate <= base_max,
        "gap_at_least_min": gap >= gap_min,
        "all_families_reference_supported": all(item["reference_at_least_min"] for item in family_results.values()),
        "all_families_base_challenging": all(item["base_at_most_max"] for item in family_results.values()),
    }
    return {
        "mode": "confirmation",
        "status": "pass" if all(checks.values()) else "fail",
        "thresholds": {
            "reference_min": reference_min,
            "base_max": base_max,
            "gap_min": gap_min,
            "family_reference_min": family_reference_min,
            "family_base_max": family_base_max,
            "expected_cases": expected_cases,
            "required_runs": required_runs,
        },
        "overall": {
            "reference_pass_all": reference_rate,
            "base_pass_all": base_rate,
            "gap": gap,
        },
        "runtime_diagnostics": {
            "base_recovered_first_length_truncations": first_truncation_count(base),
            "reference_recovered_first_length_truncations": first_truncation_count(reference),
            "base_retry_length_truncations": unrecovered_truncation_count(base),
            "reference_retry_length_truncations": unrecovered_truncation_count(reference),
        },
        "checks": checks,
        "families": family_results,
        "note": "Confirmation is accepted or rejected as a whole; no case-level filtering is permitted.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["calibration", "confirmation"], required=True)
    parser.add_argument("--base-aggregate", type=Path, required=True)
    parser.add_argument("--reference-aggregate", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--accepted-scenarios-out", type=Path)
    parser.add_argument("--reference-min", type=float, default=0.95)
    parser.add_argument("--base-max", type=float, default=0.70)
    parser.add_argument("--gap-min", type=float, default=0.25)
    parser.add_argument("--min-families", type=int, default=6)
    parser.add_argument("--min-multistep-families", type=int, default=3)
    parser.add_argument("--family-reference-min", type=float, default=0.90)
    parser.add_argument("--family-base-max", type=float, default=0.75)
    parser.add_argument("--expected-cases", type=int, default=600)
    parser.add_argument("--required-runs", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = load_json(args.base_aggregate)
    reference = load_json(args.reference_aggregate)
    cases = load_jsonl(args.cases)
    if args.mode == "calibration":
        result = calibration_gate(
            base=base,
            reference=reference,
            cases=cases,
            reference_min=args.reference_min,
            base_max=args.base_max,
            gap_min=args.gap_min,
            min_families=args.min_families,
            min_multistep_families=args.min_multistep_families,
        )
        if args.accepted_scenarios_out:
            args.accepted_scenarios_out.parent.mkdir(parents=True, exist_ok=True)
            args.accepted_scenarios_out.write_text(
                "\n".join(result["accepted_scenarios"]) + "\n",
                encoding="utf-8",
            )
    else:
        if args.accepted_scenarios_out:
            raise ValueError("--accepted-scenarios-out is calibration-only")
        result = confirmation_gate(
            base=base,
            reference=reference,
            cases=cases,
            reference_min=args.reference_min,
            base_max=args.base_max,
            gap_min=args.gap_min,
            family_reference_min=args.family_reference_min,
            family_base_max=args.family_base_max,
            expected_cases=args.expected_cases,
            required_runs=args.required_runs,
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
