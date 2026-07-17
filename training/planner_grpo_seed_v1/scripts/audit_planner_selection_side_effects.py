#!/usr/bin/env python3
"""Audit forbidden-action changes between two complete Planner rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.planner_grpo_seed_v1.scripts.compare_planner_rollout_models import (  # noqa: E402
    index_predictions,
    load_jsonl,
)
from training.planner_grpo_seed_v1.scripts.reward_planner_grpo import (  # noqa: E402
    normalize_action,
    score_case,
)


def decision_actions(prediction: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    for decision in prediction.get("decisions") or []:
        if not isinstance(decision, dict):
            actions.append("")
            continue
        decision_type = str(decision.get("decision_type") or "")
        if decision_type in {"end", "clarify"}:
            actions.append(decision_type)
        else:
            actions.append(normalize_action(str(decision.get("action") or "")))
    return actions


def audit_side_effects(
    *,
    cases: list[dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    reference: dict[str, dict[str, Any]],
    candidate_label: str,
    reference_label: str,
) -> dict[str, Any]:
    case_ids = [str(case.get("case_id") or "") for case in cases]
    if not all(case_ids) or len(case_ids) != len(set(case_ids)):
        raise ValueError("case IDs must be non-empty and unique")
    expected = set(case_ids)
    for label, predictions in (
        (candidate_label, candidate),
        (reference_label, reference),
    ):
        if set(predictions) != expected:
            raise ValueError(f"{label} prediction coverage mismatch")

    rows: list[dict[str, Any]] = []
    totals = {
        candidate_label: 0,
        reference_label: 0,
        "introduced": 0,
        "removed": 0,
    }
    strata: dict[str, dict[str, dict[str, int]]] = {
        "by_scenario": defaultdict(lambda: dict.fromkeys(totals, 0)),
        "by_detector": defaultdict(lambda: dict.fromkeys(totals, 0)),
    }
    for case in cases:
        case_id = str(case["case_id"])
        candidate_score = score_case(
            case, candidate[case_id], use_expected_when_missing=False
        )
        reference_score = score_case(
            case, reference[case_id], use_expected_when_missing=False
        )
        candidate_hits = set(candidate_score["forbidden_hit"])
        reference_hits = set(reference_score["forbidden_hit"])
        introduced = sorted(candidate_hits - reference_hits)
        removed = sorted(reference_hits - candidate_hits)
        values = {
            candidate_label: len(candidate_hits),
            reference_label: len(reference_hits),
            "introduced": len(introduced),
            "removed": len(removed),
        }
        for key, value in values.items():
            totals[key] += value
        scenario = str(case.get("scenario_id") or case.get("category") or "")
        detector = str(case.get("detector_family") or "")
        for stratum, label in (("by_scenario", scenario), ("by_detector", detector)):
            for key, value in values.items():
                strata[stratum][label][key] += value
        if introduced or removed:
            rows.append(
                {
                    "case_id": case_id,
                    "entity_id": str(case.get("entity_id") or ""),
                    "scenario_id": scenario,
                    "detector_family": detector,
                    "forbidden_actions": list(case.get("forbidden_actions") or []),
                    "introduced_forbidden_actions": introduced,
                    "removed_forbidden_actions": removed,
                    f"{candidate_label}_forbidden_hit": sorted(candidate_hits),
                    f"{reference_label}_forbidden_hit": sorted(reference_hits),
                    f"{candidate_label}_actions": decision_actions(candidate[case_id]),
                    f"{reference_label}_actions": decision_actions(reference[case_id]),
                    "expected_actions": decision_actions(
                        {"decisions": case.get("expected_decisions") or []}
                    ),
                    f"{candidate_label}_passed": bool(candidate_score["passed"]),
                    f"{reference_label}_passed": bool(reference_score["passed"]),
                }
            )

    totals["net_added"] = totals[candidate_label] - totals[reference_label]
    return {
        "schema_version": "1.0",
        "candidate_label": candidate_label,
        "reference_label": reference_label,
        "cases": len(cases),
        "changed_cases": len(rows),
        "totals": totals,
        "by_scenario": dict(sorted(strata["by_scenario"].items())),
        "by_detector": dict(sorted(strata["by_detector"].items())),
        "changed_case_details": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--reference-label", default="reference")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = audit_side_effects(
        cases=load_jsonl(args.cases),
        candidate=index_predictions(args.candidate_predictions),
        reference=index_predictions(args.reference_predictions),
        candidate_label=args.candidate_label,
        reference_label=args.reference_label,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
