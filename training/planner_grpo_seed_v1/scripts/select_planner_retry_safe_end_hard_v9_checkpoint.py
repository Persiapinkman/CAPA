#!/usr/bin/env python3
"""Select one V9 GRPO checkpoint without exposing the larger reference."""

from __future__ import annotations

import argparse
import json
import re
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


def parse_candidate(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("candidate must be LABEL=PREDICTIONS.jsonl")
    return label.strip(), Path(raw_path.strip())


def operational_metrics(predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    attempts = 0
    valid = 0
    clipped = 0
    empty_cases = 0
    for prediction in predictions.values():
        decisions = prediction.get("decisions")
        if not isinstance(decisions, list) or not decisions:
            attempts += 1
            empty_cases += 1
            continue
        for decision in decisions:
            attempts += 1
            if not isinstance(decision, dict):
                continue
            metrics = decision.get("_planner_metrics")
            metrics = metrics if isinstance(metrics, dict) else {}
            error_type = str(metrics.get("error_type") or "").strip()
            has_decision = str(decision.get("decision_type") or "").strip() in {
                "tool", "end", "clarify"
            }
            valid += int(has_decision and not error_type)
            clipped += int(
                str(metrics.get("first_finish_reason") or "") == "length"
                or str(metrics.get("retry_finish_reason") or "") == "length"
            )
    return {
        "planner_decisions": attempts,
        "empty_cases": empty_cases,
        "json_valid_rate": valid / attempts if attempts else 0.0,
        "clipped_rate": clipped / attempts if attempts else 1.0,
    }


def checkpoint_number(label: str) -> int:
    match = re.search(r"(?:checkpoint[-_])?(\d+)$", label)
    return int(match.group(1)) if match else 10**9


def select_checkpoint(
    *,
    preregistration: dict[str, Any],
    cases: list[dict[str, Any]],
    sft_predictions: dict[str, dict[str, Any]],
    candidates: list[tuple[str, Path, dict[str, dict[str, Any]]]],
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    primary_scenarios = set(preregistration["design"]["primary_scenarios"])
    gates = preregistration["selection_dev"]["promotion_gates"]
    expected_checkpoints = {
        checkpoint_number(str(model))
        for model in preregistration["selection_dev"]["models"]
        if "checkpoint" in str(model).lower()
    }
    observed_checkpoints = {checkpoint_number(label) for label, _, _ in candidates}
    if observed_checkpoints != expected_checkpoints:
        raise ValueError(
            f"candidate checkpoints mismatch: expected={sorted(expected_checkpoints)}, "
            f"observed={sorted(observed_checkpoints)}"
        )
    if len({label for label, _, _ in candidates}) != len(candidates):
        raise ValueError("candidate labels must be unique")

    sft_operations = operational_metrics(sft_predictions)
    results: list[dict[str, Any]] = []
    for label, path, predictions in candidates:
        comparison = compare_rollouts(
            cases=cases,
            candidate_predictions=predictions,
            reference_predictions=sft_predictions,
            primary_scenarios=primary_scenarios,
            candidate_label=label,
            reference_label="sft",
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
        )
        operations = operational_metrics(predictions)
        checks = {
            "primary_gain": comparison["primary"]["pass_rate_delta"]
            >= float(gates["minimum_primary_complete_trajectory_gain_over_sft"]),
            "control_regression": comparison["controls"]["pass_rate_delta"]
            >= -float(gates["maximum_control_complete_trajectory_regression"]),
            "wrong_side_effecting_actions": comparison["overall"][label][
                "wrong_side_effecting_actions"
            ]
            <= comparison["overall"]["sft"]["wrong_side_effecting_actions"],
            "json_valid_rate": operations["json_valid_rate"]
            >= float(gates["minimum_json_valid_rate"]),
            "clipped_rate": operations["clipped_rate"]
            <= float(gates["maximum_clipped_rate"]),
        }
        results.append(
            {
                "label": label,
                "checkpoint": checkpoint_number(label),
                "predictions": str(path),
                "promoted": all(checks.values()),
                "checks": checks,
                "operations": operations,
                "comparison_to_sft": comparison,
            }
        )

    promoted = [item for item in results if item["promoted"]]
    promoted.sort(
        key=lambda item: (
            -float(item["comparison_to_sft"]["primary"][item["label"]]["pass_rate"]),
            -float(item["comparison_to_sft"]["controls"][item["label"]]["pass_rate"]),
            int(item["comparison_to_sft"]["overall"][item["label"]][
                "wrong_side_effecting_actions"
            ]),
            int(item["checkpoint"]),
        )
    )
    selected = promoted[0] if promoted else None
    return {
        "schema_version": "1.0",
        "study_id": preregistration["study_id"],
        "status": "promote" if selected else "no_promotion",
        "larger_reference_used_for_selection": False,
        "sft_operations": sft_operations,
        "candidates": results,
        "selected": (
            {
                "label": selected["label"],
                "checkpoint": selected["checkpoint"],
                "predictions": selected["predictions"],
            }
            if selected
            else None
        ),
        "sealed_test_authorized": bool(selected),
        "on_fail": preregistration["selection_dev"]["on_fail"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--sft-predictions", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = select_checkpoint(
        preregistration=load_json(args.preregistration),
        cases=load_jsonl(args.cases),
        sft_predictions=index_predictions(args.sft_predictions),
        candidates=[
            (label, path, index_predictions(path)) for label, path in args.candidate
        ],
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
