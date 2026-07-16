#!/usr/bin/env python3
"""Evaluate the preregistered V9 sealed SFT/GRPO/larger-model objective."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.planner_grpo_seed_v1.scripts.compare_planner_rollout_models import (  # noqa: E402
    compare_rollouts,
    index_predictions,
    load_jsonl,
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def finalize_objective(
    *,
    contract: dict[str, Any],
    cases: list[dict[str, Any]],
    sft: dict[str, dict[str, Any]],
    grpo: dict[str, dict[str, Any]],
    larger: dict[str, dict[str, Any]],
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    primary = set(contract["units_and_metrics"]["primary_scope"])
    grpo_vs_sft = compare_rollouts(
        cases=cases,
        candidate_predictions=grpo,
        reference_predictions=sft,
        primary_scenarios=primary,
        candidate_label="grpo",
        reference_label="sft",
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    grpo_vs_larger = compare_rollouts(
        cases=cases,
        candidate_predictions=grpo,
        reference_predictions=larger,
        primary_scenarios=primary,
        candidate_label="grpo",
        reference_label="larger",
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    checks = {
        "grpo_above_sft_primary": grpo_vs_sft["primary"]["pass_rate_delta"] > 0,
        "grpo_above_larger_primary": grpo_vs_larger["primary"]["pass_rate_delta"] > 0,
        "control_guardrail": grpo_vs_sft["controls"]["pass_rate_delta"] >= -0.05,
        "wrong_side_effecting_action_guardrail": grpo_vs_sft["overall"]["grpo"][
            "wrong_side_effecting_actions"
        ]
        <= grpo_vs_sft["overall"]["sft"]["wrong_side_effecting_actions"],
    }
    ci = grpo_vs_larger["entity_clustered_primary_delta_bootstrap"]["ci95"]
    objective_met = all(checks.values())
    return {
        "schema_version": "1.0",
        "study_id": contract["study_id"],
        "status": "objective_met" if objective_met else "objective_not_met",
        "checks": checks,
        "objective_met": objective_met,
        "strong_superiority": objective_met and float(ci[0]) > 0,
        "point_estimate_only": objective_met and float(ci[0]) <= 0,
        "grpo_vs_sft": grpo_vs_sft,
        "grpo_vs_larger": grpo_vs_larger,
        "single_use_sealed_test": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--sft-predictions", type=Path, required=True)
    parser.add_argument("--grpo-predictions", type=Path, required=True)
    parser.add_argument("--larger-predictions", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = finalize_objective(
        contract=load_json(args.contract),
        cases=load_jsonl(args.cases),
        sft=index_predictions(args.sft_predictions),
        grpo=index_predictions(args.grpo_predictions),
        larger=index_predictions(args.larger_predictions),
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
