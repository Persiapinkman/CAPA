#!/usr/bin/env python3
"""Expand deduplicated V6 decisions back to cases and audit final trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN = ROOT / "experiments/runs/20260716_qwen35_4b_planner_v6_sft_ckpt100_sealedtest_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "training/planner_grpo_seed_v1/cases/planner_retry_migrate_v6_test_cases.jsonl",
    )
    parser.add_argument(
        "--formatted",
        type=Path,
        default=ROOT / "experiments/studies/planner_retry_migrate_v6_qwen35_4b_v1/sealed_test_data/test.jsonl",
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_RUN / "predictions.jsonl")
    parser.add_argument("--out", type=Path, default=DEFAULT_RUN / "trajectory_analysis.json")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: row must be an object")
        rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    count = len(values)
    return {
        "count": count,
        "action_match_rate": sum(bool(row["action_match"]) for row in values) / count if count else None,
        "argument_match_rate": sum(bool(row["argument_match"]) for row in values) / count if count else None,
        "finish_match_rate": sum(bool(row["finish_match"]) for row in values) / count if count else None,
        "mean_reward": sum(float(row["score"]) for row in values) / count if count else None,
    }


def grouped(rows: Iterable[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field) if row.get(field) is not None else "")].append(row)
    return {key: aggregate(groups[key]) for key in sorted(groups)}


def parse_action(completion: str) -> str:
    try:
        value = json.loads(completion)
    except json.JSONDecodeError:
        return "<invalid_json>"
    if not isinstance(value, dict):
        return "<non_object>"
    if value.get("decision_type") == "end":
        return "end"
    return str(value.get("action") or "<missing>")


def main() -> None:
    args = parse_args()
    cases = load_jsonl(args.cases)
    formatted = load_jsonl(args.formatted)
    predictions = load_jsonl(args.predictions)
    if len(cases) != 450 or len(formatted) != 780 or len(predictions) != 780:
        raise ValueError(
            f"unexpected cardinality: cases={len(cases)}, formatted={len(formatted)}, predictions={len(predictions)}"
        )

    def key(row: dict[str, Any]) -> tuple[str, int]:
        return str(row["case_id"]), int(row["step_index"])

    formatted_by_key = {key(row): row for row in formatted}
    predictions_by_key = {key(row): row for row in predictions}
    if len(formatted_by_key) != len(formatted) or len(predictions_by_key) != len(predictions):
        raise ValueError("duplicate formatted or prediction key")
    if formatted_by_key.keys() != predictions_by_key.keys():
        missing = sorted(formatted_by_key.keys() - predictions_by_key.keys())
        extra = sorted(predictions_by_key.keys() - formatted_by_key.keys())
        raise ValueError(f"prediction coverage mismatch: missing={missing[:5]}, extra={extra[:5]}")

    case_by_id = {str(case["case_id"]): case for case in cases}
    expanded: list[dict[str, Any]] = []
    case_steps: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unique_errors: list[dict[str, Any]] = []
    unique_confusion: Counter[str] = Counter()
    expanded_confusion: Counter[str] = Counter()

    for row_key, stage_row in formatted_by_key.items():
        prediction = predictions_by_key[row_key]
        expected = json.loads(str(stage_row["expected_step"]))
        expected_action = "end" if expected.get("decision_type") == "end" else str(expected.get("action") or "")
        actual_action = parse_action(str(prediction["scored_completion"]))
        if not bool(prediction["action_match"]):
            unique_confusion[f"{expected_action}->{actual_action}"] += 1
            unique_errors.append(
                {
                    "representative_case_id": row_key[0],
                    "source_case_ids": stage_row["source_case_ids"],
                    "source_case_count": int(stage_row["source_case_count"]),
                    "category": stage_row["category"],
                    "step_index": row_key[1],
                    "prompt_token_count": int(stage_row["prompt_token_count"]),
                    "expected_action": expected_action,
                    "actual_action": actual_action,
                    "score": float(prediction["score"]),
                    "completion": prediction["completion"],
                }
            )

        for source_case_id in stage_row["source_case_ids"]:
            source_case_id = str(source_case_id)
            case = case_by_id.get(source_case_id)
            if case is None:
                raise ValueError(f"unknown source case {source_case_id}")
            item = {
                "case_id": source_case_id,
                "category": str(case["category"]),
                "detector_family": str(case["detector_family"]),
                "error_alias": str(case.get("error_alias") or "none"),
                "badge_condition": str(case.get("badge_condition") or "none"),
                "fixture_family": str(case.get("image_fixture_family") or "none"),
                "template_id": str(case.get("template_id") or "none"),
                "guardrail": bool(case["guardrail"]),
                "post_retry_outcome": str(case.get("post_retry_outcome") or "none"),
                "target_action_class": str(case["target_action_class"]),
                "step_index": row_key[1],
                "prompt_token_count": int(stage_row["prompt_token_count"]),
                "expected_action": expected_action,
                "actual_action": actual_action,
                "action_match": bool(prediction["action_match"]),
                "argument_match": bool(prediction["argument_match"]),
                "finish_match": bool(prediction["finish_match"]),
                "score": float(prediction["score"]),
            }
            expanded.append(item)
            case_steps[source_case_id].append(item)
            if not item["action_match"]:
                expanded_confusion[f"{expected_action}->{actual_action}"] += 1

    if len(expanded) != 1020:
        raise ValueError(f"expected 1020 expanded decisions, got {len(expanded)}")

    trajectories: list[dict[str, Any]] = []
    for case_id, case in case_by_id.items():
        steps = sorted(case_steps[case_id], key=lambda row: int(row["step_index"]))
        expected_count = len(case["expected_decisions"])
        if len(steps) != expected_count or [row["step_index"] for row in steps] != list(range(1, expected_count + 1)):
            raise ValueError(f"incomplete trajectory for {case_id}: {len(steps)} != {expected_count}")
        trajectories.append(
            {
                "case_id": case_id,
                "category": str(case["category"]),
                "detector_family": str(case["detector_family"]),
                "error_alias": str(case.get("error_alias") or "none"),
                "badge_condition": str(case.get("badge_condition") or "none"),
                "fixture_family": str(case.get("image_fixture_family") or "none"),
                "template_id": str(case.get("template_id") or "none"),
                "guardrail": bool(case["guardrail"]),
                "post_retry_outcome": str(case.get("post_retry_outcome") or "none"),
                "target_action_class": str(case["target_action_class"]),
                "steps": expected_count,
                "full_action_match": all(row["action_match"] for row in steps),
                "full_argument_match": all(row["argument_match"] for row in steps),
                "full_finish_match": all(row["finish_match"] for row in steps),
                "full_rule_pass": all(abs(float(row["score"]) - 1.0) <= 1e-12 for row in steps),
            }
        )

    def trajectory_aggregate(items: Iterable[dict[str, Any]]) -> dict[str, Any]:
        values = list(items)
        count = len(values)
        return {
            "count": count,
            "full_action_match_rate": sum(bool(row["full_action_match"]) for row in values) / count if count else None,
            "full_argument_match_rate": sum(bool(row["full_argument_match"]) for row in values) / count if count else None,
            "full_finish_match_rate": sum(bool(row["full_finish_match"]) for row in values) / count if count else None,
            "full_rule_pass_rate": sum(bool(row["full_rule_pass"]) for row in values) / count if count else None,
        }

    def trajectory_grouped(field: str) -> dict[str, dict[str, Any]]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in trajectories:
            groups[str(row[field])].append(row)
        return {group_key: trajectory_aggregate(groups[group_key]) for group_key in sorted(groups)}

    result = {
        "schema_version": "1.0",
        "inputs": {
            "cases": {"path": str(args.cases), "sha256": sha256_file(args.cases)},
            "formatted": {"path": str(args.formatted), "sha256": sha256_file(args.formatted)},
            "predictions": {"path": str(args.predictions), "sha256": sha256_file(args.predictions)},
        },
        "cardinality": {
            "source_cases": len(cases),
            "unique_formatted_decisions": len(formatted),
            "expanded_case_decisions": len(expanded),
            "trajectories": len(trajectories),
        },
        "unique_decisions": {
            "overall": aggregate(predictions),
            "action_confusion_on_errors": dict(sorted(unique_confusion.items())),
            "errors": unique_errors,
        },
        "expanded_decisions": {
            "overall": aggregate(expanded),
            "by_category": grouped(expanded, "category"),
            "by_detector_family": grouped(expanded, "detector_family"),
            "by_error_alias": grouped(expanded, "error_alias"),
            "by_badge_condition": grouped(expanded, "badge_condition"),
            "by_fixture_family": grouped(expanded, "fixture_family"),
            "by_template_id": grouped(expanded, "template_id"),
            "by_step_index": grouped(expanded, "step_index"),
            "by_post_retry_outcome": grouped(expanded, "post_retry_outcome"),
            "action_confusion_on_errors": dict(sorted(expanded_confusion.items())),
        },
        "trajectories": {
            "overall": trajectory_aggregate(trajectories),
            "by_category": trajectory_grouped("category"),
            "by_detector_family": trajectory_grouped("detector_family"),
            "by_error_alias": trajectory_grouped("error_alias"),
            "by_badge_condition": trajectory_grouped("badge_condition"),
            "by_fixture_family": trajectory_grouped("fixture_family"),
            "by_post_retry_outcome": trajectory_grouped("post_retry_outcome"),
            "by_target_action_class": trajectory_grouped("target_action_class"),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
