#!/usr/bin/env python3
"""Check the preregistered multi-seed runtime-routing development gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.eval.check_runtime_routing_gate import _load_run, _read_json, _summary
from pipelines.eval.compare_generation_runs import paired_comparison


def _mean_metric(items: list[dict[str, Any]], *keys: str) -> float:
    values: list[float] = []
    for item in items:
        value: Any = item
        for key in keys:
            value = value[key]
        values.append(float(value))
    return mean(values)


def evaluate_gate(
    study: dict[str, Any],
    baseline: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
    mean_policy_comparison: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate every condition without changing the preregistered thresholds."""

    gate = study["development_replication_gate"]
    candidate_items = list(candidates.values())
    primary = str(gate["primary_category"])
    contrast = str(gate["contrast_category"])
    expected_seed_count = len(study["test_confirmation"]["models"]) - 1

    seed_primary_strict_deltas = {
        label: float(summary["categories"][primary]["strict_case_action_rate"])
        - float(baseline["categories"][primary]["strict_case_action_rate"])
        for label, summary in candidates.items()
    }
    mean_primary_strict_delta = mean(seed_primary_strict_deltas.values())
    positive_seed_count = sum(delta > 0.0 for delta in seed_primary_strict_deltas.values())

    mean_step_action_delta = _mean_metric(candidate_items, "action_match_rate") - float(
        baseline["action_match_rate"]
    )
    action_ci95 = [
        float(value) for value in mean_policy_comparison["action_match_ci95"]
    ]
    mean_primary_verifier_delta = _mean_metric(
        candidate_items, "categories", primary, "mean_score"
    ) - float(baseline["categories"][primary]["mean_score"])

    primary_step2_key = f"{primary}#step2"
    mean_step2_delta = _mean_metric(
        candidate_items, "category_steps", primary_step2_key, "action_match_rate"
    ) - float(baseline["category_steps"][primary_step2_key]["action_match_rate"])
    mean_contrast_delta = _mean_metric(
        candidate_items, "categories", contrast, "action_match_rate"
    ) - float(baseline["categories"][contrast]["action_match_rate"])

    category_action_deltas = {
        category: _mean_metric(
            candidate_items, "categories", category, "action_match_rate"
        )
        - float(baseline["categories"][category]["action_match_rate"])
        for category in baseline["categories"]
    }
    other_category_deltas = {
        category: delta
        for category, delta in category_action_deltas.items()
        if category not in {primary, contrast}
    }
    mean_wrong_side_effect_delta = _mean_metric(
        candidate_items, "wrong_side_effect_count"
    ) - float(baseline["wrong_side_effect_count"])

    checks = {
        "candidate_seed_count": len(candidates) == expected_seed_count,
        "primary_strict_case_action": mean_primary_strict_delta
        >= float(gate["minimum_three_seed_mean_strict_case_action_delta"]),
        "overall_step_action": mean_step_action_delta
        >= float(gate["minimum_three_seed_mean_step_action_delta"]),
        "positive_seeds": positive_seed_count
        >= int(gate["minimum_positive_seed_count"]),
        "entity_clustered_action_ci": (
            action_ci95[0] > 0.0
            if gate["require_entity_clustered_action_ci_lower_above_zero"]
            else True
        ),
        "primary_verifier": mean_primary_verifier_delta
        >= float(gate["minimum_three_seed_mean_verifier_delta"]),
        "primary_step2": (
            mean_step2_delta >= 0.0
            if gate["require_mean_step2_action_no_regression"]
            else True
        ),
        "contrast": mean_contrast_delta
        >= -float(gate["maximum_mean_contrast_action_regression"]),
        "other_categories": all(
            delta >= -float(gate["maximum_mean_other_category_action_regression"])
            for delta in other_category_deltas.values()
        ),
        "wrong_side_effects": (
            mean_wrong_side_effect_delta <= 0.0
            if gate["require_no_mean_increase_in_wrong_side_effecting_actions"]
            else True
        ),
    }
    passed = all(checks.values())
    return {
        "schema_version": "1.0",
        "study_id": study["study_id"],
        "arm_id": study["arm_id"],
        "candidate_seeds": list(candidates),
        "expected_seed_count": expected_seed_count,
        "primary": {
            "category": primary,
            "mean_strict_case_action_delta": mean_primary_strict_delta,
            "minimum_mean_strict_case_action_delta": gate[
                "minimum_three_seed_mean_strict_case_action_delta"
            ],
            "seed_strict_case_action_deltas": seed_primary_strict_deltas,
            "positive_seed_count": positive_seed_count,
            "minimum_positive_seed_count": gate["minimum_positive_seed_count"],
            "mean_verifier_delta": mean_primary_verifier_delta,
            "minimum_mean_verifier_delta": gate[
                "minimum_three_seed_mean_verifier_delta"
            ],
            "mean_step2_action_delta": mean_step2_delta,
        },
        "overall": {
            "mean_step_action_delta": mean_step_action_delta,
            "minimum_mean_step_action_delta": gate[
                "minimum_three_seed_mean_step_action_delta"
            ],
            "entity_clustered_action_ci95": action_ci95,
        },
        "contrast": {
            "category": contrast,
            "mean_action_delta": mean_contrast_delta,
        },
        "category_action_deltas": category_action_deltas,
        "other_category_action_deltas": other_category_deltas,
        "mean_wrong_side_effect_delta": mean_wrong_side_effect_delta,
        "checks": checks,
        "passed": passed,
        "test_may_open": passed,
    }


def _mean_policy_rows(
    candidate_rows: list[dict[tuple[str, int], dict[str, Any]]],
) -> dict[tuple[str, int], dict[str, Any]]:
    keys = set(candidate_rows[0])
    if any(set(rows) != keys for rows in candidate_rows[1:]):
        raise ValueError("candidate prediction keys do not match")
    return {
        key: {
            **candidate_rows[0][key],
            "score": mean(float(rows[key]["score"]) for rows in candidate_rows),
            "action_match": mean(
                float(rows[key]["action_match"]) for rows in candidate_rows
            ),
        }
        for key in sorted(keys)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="LABEL=run_record.json; pass all preregistered seeds",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _parse_candidate(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise ValueError(f"expected LABEL=PATH, got {value!r}")
    return label, Path(path)


def main() -> None:
    args = parse_args()
    study = _read_json(args.study)
    baseline_run = _load_run(args.baseline)
    candidate_runs = {
        label: _load_run(path) for label, path in map(_parse_candidate, args.candidate)
    }
    baseline_keys = set(baseline_run["rows"])
    if any(set(run["rows"]) != baseline_keys for run in candidate_runs.values()):
        raise ValueError("baseline and candidate prediction keys do not match")

    averaged_rows = _mean_policy_rows(
        [run["rows"] for run in candidate_runs.values()]
    )
    comparison = paired_comparison(
        baseline_run["rows"],
        averaged_rows,
        seed=args.seed,
        samples=args.bootstrap_samples,
        cluster_key="entity_id",
    )
    payload = evaluate_gate(
        study,
        _summary(baseline_run["rows"]),
        {
            label: _summary(run["rows"])
            for label, run in candidate_runs.items()
        },
        comparison,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": payload["passed"],
                "test_may_open": payload["test_may_open"],
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
