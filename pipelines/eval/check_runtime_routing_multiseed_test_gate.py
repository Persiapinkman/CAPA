#!/usr/bin/env python3
"""Check the preregistered runtime-routing multi-seed sealed-test gate."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.eval.check_runtime_routing_gate import _load_run, _read_json, _summary
from pipelines.eval.check_runtime_routing_multiseed_gate import (
    _mean_policy_rows,
    evaluate_gate as evaluate_development_gate,
)
from pipelines.eval.compare_generation_runs import paired_comparison


def evaluate_test_gate(
    study: dict[str, Any],
    baseline: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
    mean_policy_comparison: dict[str, Any],
) -> dict[str, Any]:
    gate = study["test_confirmation_gate"]
    proxy = copy.deepcopy(study)
    proxy["development_replication_gate"] = {
        "primary_category": gate["primary_category"],
        "minimum_three_seed_mean_strict_case_action_delta": 0.0,
        "minimum_three_seed_mean_step_action_delta": 0.0,
        "minimum_positive_seed_count": gate["minimum_positive_seed_count"],
        "require_entity_clustered_action_ci_lower_above_zero": gate[
            "require_entity_clustered_action_ci_lower_above_zero"
        ],
        "minimum_three_seed_mean_verifier_delta": 0.0,
        "require_mean_step2_action_no_regression": gate[
            "require_mean_step2_action_no_regression"
        ],
        "contrast_category": gate["contrast_category"],
        "maximum_mean_contrast_action_regression": gate[
            "maximum_mean_contrast_action_regression"
        ],
        "maximum_mean_other_category_action_regression": gate[
            "maximum_mean_other_category_action_regression"
        ],
        "require_no_mean_increase_in_wrong_side_effecting_actions": gate[
            "require_no_mean_increase_in_wrong_side_effecting_actions"
        ],
    }
    payload = evaluate_development_gate(
        proxy, baseline, candidates, mean_policy_comparison
    )
    checks = payload["checks"]
    checks["primary_strict_case_action"] = (
        payload["primary"]["mean_strict_case_action_delta"] > 0.0
        if gate["require_three_seed_mean_strict_case_action_delta_positive"]
        else True
    )
    checks["overall_step_action"] = (
        payload["overall"]["mean_step_action_delta"] > 0.0
        if gate["require_three_seed_mean_step_action_delta_positive"]
        else True
    )
    checks["primary_verifier"] = (
        payload["primary"]["mean_verifier_delta"] > 0.0
        if gate["require_three_seed_mean_verifier_delta_positive"]
        else True
    )
    checks["test_gate_preregistered"] = bool(
        gate.get("preregistered_while_test_sealed", False)
    )
    passed = all(checks.values())
    payload.update(
        {
            "split": "test",
            "requirements": {
                "mean_primary_strict_case_action_delta": "positive",
                "mean_step_action_delta": "positive",
                "mean_primary_verifier_delta": "positive",
            },
            "passed": passed,
            "test_confirmed": passed,
        }
    )
    payload.pop("test_may_open", None)
    return payload


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
    payload = evaluate_test_gate(
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
                "test_confirmed": payload["test_confirmed"],
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
