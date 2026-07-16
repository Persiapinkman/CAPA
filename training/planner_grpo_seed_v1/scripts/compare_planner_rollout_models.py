#!/usr/bin/env python3
"""Compare two Planner rollout files with entity-clustered paired metrics."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.planner_grpo_seed_v1.scripts import reward_planner_grpo as rewardlib  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: all rows must be objects")
    return rows


def index_predictions(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    indexed = {str(row.get("case_id") or ""): row for row in rows}
    if "" in indexed or len(indexed) != len(rows):
        raise ValueError(f"{path}: prediction case IDs must be non-empty and unique")
    return indexed


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def aggregate(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    if not rows:
        return {
            "cases": 0,
            "pass_rate": 0.0,
            "mean_reward": 0.0,
            "wrong_side_effecting_actions": 0,
        }
    return {
        "cases": len(rows),
        "pass_rate": statistics.fmean(float(row[f"{label}_passed"]) for row in rows),
        "mean_reward": statistics.fmean(float(row[f"{label}_score"]) for row in rows),
        "wrong_side_effecting_actions": sum(
            len(row[f"{label}_forbidden_hit"]) for row in rows
        ),
    }


def compare_rollouts(
    *,
    cases: list[dict[str, Any]],
    candidate_predictions: dict[str, dict[str, Any]],
    reference_predictions: dict[str, dict[str, Any]],
    primary_scenarios: set[str],
    candidate_label: str,
    reference_label: str,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    case_ids = [str(case.get("case_id") or "") for case in cases]
    if not all(case_ids) or len(case_ids) != len(set(case_ids)):
        raise ValueError("case IDs must be non-empty and unique")
    expected = set(case_ids)
    for label, predictions in (
        (candidate_label, candidate_predictions),
        (reference_label, reference_predictions),
    ):
        if set(predictions) != expected:
            missing = sorted(expected - set(predictions))
            extra = sorted(set(predictions) - expected)
            raise ValueError(f"{label}: prediction coverage mismatch; missing={missing[:3]}, extra={extra[:3]}")

    paired: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["case_id"])
        candidate = rewardlib.score_case(
            case, candidate_predictions[case_id], use_expected_when_missing=False
        )
        reference = rewardlib.score_case(
            case, reference_predictions[case_id], use_expected_when_missing=False
        )
        paired.append(
            {
                "case_id": case_id,
                "entity_id": str(case.get("entity_id") or ""),
                "scenario_id": str(case.get("scenario_id") or case.get("category") or ""),
                "detector_family": str(case.get("detector_family") or ""),
                f"{candidate_label}_passed": bool(candidate["passed"]),
                f"{candidate_label}_score": float(candidate["score"]),
                f"{candidate_label}_forbidden_hit": list(candidate["forbidden_hit"]),
                f"{reference_label}_passed": bool(reference["passed"]),
                f"{reference_label}_score": float(reference["score"]),
                f"{reference_label}_forbidden_hit": list(reference["forbidden_hit"]),
            }
        )
    if any(not row["entity_id"] for row in paired):
        raise ValueError("every case must have an entity_id")

    primary = [row for row in paired if row["scenario_id"] in primary_scenarios]
    controls = [row for row in paired if row["scenario_id"] not in primary_scenarios]
    scenarios = sorted({row["scenario_id"] for row in paired})
    detectors = sorted({row["detector_family"] for row in paired})

    def scope_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
        candidate_metrics = aggregate(rows, candidate_label)
        reference_metrics = aggregate(rows, reference_label)
        return {
            candidate_label: candidate_metrics,
            reference_label: reference_metrics,
            "pass_rate_delta": candidate_metrics["pass_rate"] - reference_metrics["pass_rate"],
            "mean_reward_delta": candidate_metrics["mean_reward"] - reference_metrics["mean_reward"],
            "paired_case_counts": {
                "candidate_only": sum(
                    row[f"{candidate_label}_passed"] and not row[f"{reference_label}_passed"]
                    for row in rows
                ),
                "reference_only": sum(
                    row[f"{reference_label}_passed"] and not row[f"{candidate_label}_passed"]
                    for row in rows
                ),
                "both": sum(
                    row[f"{candidate_label}_passed"] and row[f"{reference_label}_passed"]
                    for row in rows
                ),
                "neither": sum(
                    not row[f"{candidate_label}_passed"] and not row[f"{reference_label}_passed"]
                    for row in rows
                ),
            },
        }

    entity_primary: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in primary:
        entity_primary[row["entity_id"]].append(row)
    entity_deltas = {
        entity: statistics.fmean(
            float(row[f"{candidate_label}_passed"]) - float(row[f"{reference_label}_passed"])
            for row in rows
        )
        for entity, rows in entity_primary.items()
    }
    rng = random.Random(seed)
    entities = sorted(entity_deltas)
    bootstrap: list[float] = []
    for _ in range(max(0, bootstrap_replicates)):
        sampled = [rng.choice(entities) for _ in entities]
        bootstrap.append(statistics.fmean(entity_deltas[entity] for entity in sampled))

    return {
        "schema_version": "1.0",
        "candidate_label": candidate_label,
        "reference_label": reference_label,
        "independent_unit": "entity_id",
        "primary_scenarios": sorted(primary_scenarios),
        "overall": scope_metrics(paired),
        "primary": scope_metrics(primary),
        "controls": scope_metrics(controls),
        "by_scenario": {
            scenario: scope_metrics([row for row in paired if row["scenario_id"] == scenario])
            for scenario in scenarios
        },
        "by_detector": {
            detector: scope_metrics(
                [row for row in paired if row["detector_family"] == detector]
            )
            for detector in detectors
        },
        "entity_primary_deltas": entity_deltas,
        "entity_clustered_primary_delta_bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": seed,
            "point_estimate": statistics.fmean(entity_deltas.values()) if entity_deltas else 0.0,
            "ci95": [percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--primary-scenarios", nargs="+", required=True)
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--reference-label", default="reference")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = compare_rollouts(
        cases=load_jsonl(args.cases),
        candidate_predictions=index_predictions(args.candidate_predictions),
        reference_predictions=index_predictions(args.reference_predictions),
        primary_scenarios=set(args.primary_scenarios),
        candidate_label=args.candidate_label,
        reference_label=args.reference_label,
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
