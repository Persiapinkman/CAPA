#!/usr/bin/env python3
"""Check the preregistered runtime-routing development gate."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from capa.tools.registry import normalize_tool_action  # noqa: E402


SIDE_EFFECTING_ACTIONS = {
    "flux-image-generation",
    "pipeline_eval",
    "adela_cli_eval",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _action(value: str | dict[str, Any]) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return ""
    if not isinstance(value, dict):
        return ""
    decision_type = str(value.get("decision_type") or "")
    if decision_type in {"clarify", "end"}:
        return decision_type
    return normalize_tool_action(str(value.get("action") or ""))


def _load_run(record_path: Path) -> dict[str, Any]:
    record = _read_json(record_path)
    eval_rows = _read_jsonl(Path(record["data"]["files"]["eval"]))
    expected = {
        (str(row["case_id"]), int(row["step_index"])): {
            "action": _action(row["expected_step"]),
            "category": str(row["category"]),
            "entity_id": str(row.get("entity_id") or ""),
        }
        for row in eval_rows
    }
    prediction_paths = [
        Path(path) for path in record.get("artifacts", {}).get("predictions", [])
    ]
    if not prediction_paths:
        raise ValueError(f"{record_path}: no prediction artifacts")
    repeats = [
        {
            (str(row["case_id"]), int(row["step_index"])): row
            for row in _read_jsonl(path)
        }
        for path in prediction_paths
    ]
    expected_keys = set(expected)
    if any(set(rows) != expected_keys for rows in repeats):
        raise ValueError(f"{record_path}: prediction keys do not match eval rows")
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for key in sorted(expected):
        target = expected[key]
        repeat_rows = [items[key] for items in repeats]
        actual_actions = [
            _action(row.get("scored_completion", "")) for row in repeat_rows
        ]
        action_matches = [actual == target["action"] for actual in actual_actions]
        rows[key] = {
            **target,
            "case_id": key[0],
            "step_index": key[1],
            "actual_action": (
                actual_actions[0]
                if len(set(actual_actions)) == 1
                else "repeat_disagreement"
            ),
            "action_match": all(action_matches),
            "repeat_action_match_rate": mean(float(value) for value in action_matches),
            "score": mean(float(row["score"]) for row in repeat_rows),
            "wrong_side_effect": any(
                actual in SIDE_EFFECTING_ACTIONS and actual != target["action"]
                for actual in actual_actions
            ),
        }
    return {"record": record, "rows": rows, "repeat_count": len(repeats)}


def _summary(rows: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_category_step: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows.values():
        by_category[row["category"]].append(row)
        by_category_step[(row["category"], row["step_index"])].append(row)
        by_case[row["case_id"]].append(row)
    strict_cases = {
        case_id: all(bool(row["action_match"]) for row in items)
        for case_id, items in by_case.items()
    }
    category_cases: dict[str, list[bool]] = defaultdict(list)
    for case_id, passed in strict_cases.items():
        category_cases[by_case[case_id][0]["category"]].append(passed)
    return {
        "steps": len(rows),
        "cases": len(by_case),
        "action_match_rate": mean(
            float(row["action_match"]) for row in rows.values()
        ),
        "strict_case_action_rate": mean(float(value) for value in strict_cases.values()),
        "wrong_side_effect_count": sum(
            int(row["wrong_side_effect"]) for row in rows.values()
        ),
        "categories": {
            category: {
                "steps": len(items),
                "action_match_rate": mean(
                    float(row["action_match"]) for row in items
                ),
                "mean_score": mean(float(row["score"]) for row in items),
                "strict_case_action_rate": mean(
                    float(value) for value in category_cases[category]
                ),
                "wrong_side_effect_count": sum(
                    int(row["wrong_side_effect"]) for row in items
                ),
            }
            for category, items in sorted(by_category.items())
        },
        "category_steps": {
            f"{category}#step{step_index}": {
                "steps": len(items),
                "action_match_rate": mean(
                    float(row["action_match"]) for row in items
                ),
                "mean_score": mean(float(row["score"]) for row in items),
            }
            for (category, step_index), items in sorted(by_category_step.items())
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    study = _read_json(args.study)
    gate = study["development_gate"]
    baseline = _summary(_load_run(args.baseline)["rows"])
    candidate = _summary(_load_run(args.candidate)["rows"])
    primary = str(gate["primary_category"])
    contrast = str(gate["contrast_category"])
    minimum_primary_action_delta = float(
        gate.get(
            "minimum_primary_action_delta",
            gate.get("minimum_strict_case_action_delta", 0.0),
        )
    )
    maximum_guardrail_regression = float(
        gate.get(
            "maximum_any_guardrail_action_regression",
            gate.get("maximum_any_other_category_action_regression", 0.0),
        )
    )

    primary_action_delta = (
        candidate["categories"][primary]["strict_case_action_rate"]
        - baseline["categories"][primary]["strict_case_action_rate"]
    )
    primary_score_delta = (
        candidate["categories"][primary]["mean_score"]
        - baseline["categories"][primary]["mean_score"]
    )
    contrast_delta = (
        candidate["categories"][contrast]["action_match_rate"]
        - baseline["categories"][contrast]["action_match_rate"]
    )
    category_action_deltas = {
        category: (
            candidate["categories"][category]["action_match_rate"]
            - baseline["categories"][category]["action_match_rate"]
        )
        for category in baseline["categories"]
    }
    guardrail_floor = -maximum_guardrail_regression
    guardrails_passed = all(
        delta >= guardrail_floor
        for category, delta in category_action_deltas.items()
        if category != primary
    )
    side_effect_delta = (
        candidate["wrong_side_effect_count"] - baseline["wrong_side_effect_count"]
    )
    primary_step2_key = f"{primary}#step2"
    step2_delta = (
        candidate["category_steps"].get(primary_step2_key, {}).get(
            "action_match_rate", 0.0
        )
        - baseline["category_steps"].get(primary_step2_key, {}).get(
            "action_match_rate", 0.0
        )
    )
    checks = {
        "primary_action": primary_action_delta
        >= minimum_primary_action_delta,
        "primary_verifier": primary_score_delta
        >= float(gate["minimum_primary_verifier_delta"]),
        "contrast": contrast_delta
        >= -float(gate["maximum_contrast_action_regression"]),
        "guardrails": guardrails_passed,
        "side_effects": side_effect_delta <= 0,
    }
    if bool(gate.get("require_step2_action_match_no_regression", False)):
        checks["primary_step2"] = step2_delta >= 0.0
    payload = {
        "schema_version": "1.0",
        "study_id": study["study_id"],
        "baseline": baseline,
        "candidate": candidate,
        "primary": {
            "category": primary,
            "strict_case_action_delta": primary_action_delta,
            "minimum_action_delta": minimum_primary_action_delta,
            "verifier_delta": primary_score_delta,
            "minimum_verifier_delta": gate["minimum_primary_verifier_delta"],
            "step2_action_delta": step2_delta,
        },
        "contrast": {"category": contrast, "action_delta": contrast_delta},
        "category_action_deltas": category_action_deltas,
        "wrong_side_effect_delta": side_effect_delta,
        "checks": checks,
        "passed": all(checks.values()),
        "test_may_open": all(checks.values()),
    }
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
