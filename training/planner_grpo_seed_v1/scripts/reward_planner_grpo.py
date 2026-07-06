#!/usr/bin/env python3
"""Rule verifier for Planner GRPO seed cases."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_REWARD_SPEC = {
    "json_valid": 0.10,
    "decision_type_valid": 0.10,
    "action_match": 0.35,
    "argument_match": 0.25,
    "finish_after_tool": 0.10,
    "no_forbidden_action": 0.10,
}


ACTION_ALIASES = {
    "qwen-vlm-open-set-delection": "qwen_detection",
    "rexomni-open-set-detection": "rexomni_detection",
    "target-detection-evaluation": "pipeline_eval",
    "final_answer": "answerer",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"{path}:{line_no}: row must be an object")
        rows.append(data)
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_action(action: str) -> str:
    raw = str(action or "").strip()
    return ACTION_ALIASES.get(raw, raw)


def as_decision_list(value: Any) -> tuple[list[dict[str, Any]], bool]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return [], False
    if isinstance(value, dict):
        if isinstance(value.get("decisions"), list):
            decisions = value.get("decisions")
        elif isinstance(value.get("decision"), dict):
            decisions = [value.get("decision")]
        elif value.get("decision_type") or value.get("action"):
            decisions = [value]
        else:
            decisions = []
    elif isinstance(value, list):
        decisions = value
    else:
        decisions = []
    clean = [item for item in decisions if isinstance(item, dict)]
    return clean, len(clean) == len(decisions)


def value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return bool(actual) is expected
    if isinstance(expected, int) and not isinstance(expected, bool):
        try:
            return int(actual) == expected
        except (TypeError, ValueError):
            return False
    return str(actual or "").strip() == str(expected or "").strip()


def text_contains_all(actual: Any, tokens: list[Any]) -> bool:
    haystack = str(actual or "").lower()
    for token in tokens:
        needle = str(token or "").strip().lower()
        if needle and needle not in haystack:
            return False
    return True


def get_arg(decision: dict[str, Any], key: str) -> Any:
    if key == "clarification_question":
        return decision.get("clarification_question")
    action_input = decision.get("action_input") if isinstance(decision.get("action_input"), dict) else {}
    return action_input.get(key)


def score_expected_step(
    *,
    expected: dict[str, Any],
    actual: dict[str, Any] | None,
    reward_spec: dict[str, float],
) -> tuple[float, dict[str, Any]]:
    detail = {
        "json_valid": 0.0,
        "decision_type_valid": 0.0,
        "action_match": 0.0,
        "argument_match": 0.0,
        "finish_after_tool": 0.0,
    }
    failures: list[str] = []
    if not isinstance(actual, dict):
        failures.append("missing_decision")
        return 0.0, {"detail": detail, "failures": failures}

    detail["json_valid"] = 1.0
    expected_decision_type = str(expected.get("decision_type") or "tool").strip()
    actual_decision_type = str(actual.get("decision_type") or "").strip()
    if actual_decision_type == expected_decision_type:
        detail["decision_type_valid"] = 1.0
    else:
        failures.append(f"decision_type expected {expected_decision_type!r}, got {actual_decision_type!r}")

    expected_action = normalize_action(str(expected.get("action") or ""))
    if expected_decision_type == "clarify":
        actual_action = "clarify" if actual_decision_type == "clarify" else normalize_action(str(actual.get("action") or ""))
    else:
        actual_action = normalize_action(str(actual.get("action") or ""))
    if actual_action == expected_action:
        detail["action_match"] = 1.0
    else:
        failures.append(f"action expected {expected_action!r}, got {actual_action!r}")

    required_args = expected.get("required_args") if isinstance(expected.get("required_args"), dict) else {}
    arg_contains = expected.get("arg_contains") if isinstance(expected.get("arg_contains"), dict) else {}
    arg_checks = 0
    arg_hits = 0
    finish_checked = False
    finish_hit = False
    for key, expected_value in required_args.items():
        if key == "finish_after_tool":
            finish_checked = True
            finish_hit = value_matches(get_arg(actual, key), expected_value)
            if not finish_hit:
                failures.append(f"finish_after_tool expected {expected_value!r}, got {get_arg(actual, key)!r}")
            continue
        arg_checks += 1
        if value_matches(get_arg(actual, key), expected_value):
            arg_hits += 1
        else:
            failures.append(f"arg {key!r} expected {expected_value!r}, got {get_arg(actual, key)!r}")
    for key, tokens in arg_contains.items():
        token_list = tokens if isinstance(tokens, list) else [tokens]
        arg_checks += 1
        if text_contains_all(get_arg(actual, key), token_list):
            arg_hits += 1
        else:
            failures.append(f"arg {key!r} expected to contain {token_list!r}, got {get_arg(actual, key)!r}")

    detail["argument_match"] = 1.0 if arg_checks == 0 else arg_hits / arg_checks
    detail["finish_after_tool"] = 1.0 if not finish_checked else float(finish_hit)

    score = 0.0
    for key in ("json_valid", "decision_type_valid", "action_match", "argument_match", "finish_after_tool"):
        score += float(reward_spec.get(key, DEFAULT_REWARD_SPEC.get(key, 0.0))) * detail[key]
    return score, {"detail": detail, "failures": failures}


def score_case(
    case: dict[str, Any],
    prediction: dict[str, Any] | None = None,
    *,
    use_expected_when_missing: bool = True,
) -> dict[str, Any]:
    reward_spec = dict(DEFAULT_REWARD_SPEC)
    if isinstance(case.get("reward_spec"), dict):
        reward_spec.update(case["reward_spec"])

    if prediction is None and use_expected_when_missing:
        decisions = []
        for exp in case.get("expected_decisions") if isinstance(case.get("expected_decisions"), list) else []:
            action = str(exp.get("action") or "")
            decision_type = str(exp.get("decision_type") or "tool")
            if decision_type == "clarify":
                decisions.append(
                    {
                        "decision_type": "clarify",
                        "clarification_question": "请明确你是要检测图片、生成图片，还是查询方案？",
                    }
                )
                continue
            action_input = {}
            action_input.update(exp.get("required_args") if isinstance(exp.get("required_args"), dict) else {})
            for key, tokens in (exp.get("arg_contains") if isinstance(exp.get("arg_contains"), dict) else {}).items():
                if isinstance(tokens, list):
                    action_input.setdefault(key, " ".join(str(t) for t in tokens))
            decisions.append({"decision_type": decision_type, "action": action, "action_input": action_input})
        parse_ok = True
    elif prediction is None:
        return {
            "case_id": case.get("case_id"),
            "category": case.get("category"),
            "score": 0.0,
            "raw_score": 0.0,
            "max_score": 1.0,
            "forbidden_hit": [],
            "used_actions": [],
            "step_scores": [],
            "parse_ok": False,
            "missing_prediction": True,
        }
    else:
        decisions, parse_ok = as_decision_list(prediction.get("decisions", prediction))

    expected_list = case.get("expected_decisions") if isinstance(case.get("expected_decisions"), list) else []
    step_scores: list[dict[str, Any]] = []
    raw_score = 0.0
    for index, expected in enumerate(expected_list):
        actual = decisions[index] if index < len(decisions) else None
        part, info = score_expected_step(expected=expected, actual=actual, reward_spec=reward_spec)
        raw_score += part
        step_scores.append({"step": index + 1, "score_without_forbidden": part, **info})

    expected_weight = sum(
        float(reward_spec.get(key, DEFAULT_REWARD_SPEC.get(key, 0.0)))
        for key in ("json_valid", "decision_type_valid", "action_match", "argument_match", "finish_after_tool")
    )
    max_step_score = expected_weight * max(1, len(expected_list))

    forbidden = {normalize_action(str(x)) for x in case.get("forbidden_actions", []) if str(x).strip()}
    used_actions = []
    for decision in decisions:
        if str(decision.get("decision_type") or "").strip() == "clarify":
            used_actions.append("clarify")
        else:
            used_actions.append(normalize_action(str(decision.get("action") or "")))
    forbidden_hit = sorted(action for action in used_actions if action in forbidden)
    forbidden_score = 0.0 if forbidden_hit else float(reward_spec.get("no_forbidden_action", 0.0))
    total_possible = max_step_score + float(reward_spec.get("no_forbidden_action", 0.0))
    total_score = (raw_score + forbidden_score) / total_possible if total_possible > 0 else 0.0
    if not parse_ok:
        total_score = 0.0

    return {
        "case_id": case.get("case_id"),
        "category": case.get("category"),
        "score": round(total_score, 6),
        "raw_score": round(raw_score + forbidden_score, 6),
        "max_score": round(total_possible, 6),
        "forbidden_hit": forbidden_hit,
        "used_actions": used_actions,
        "step_scores": step_scores,
        "parse_ok": parse_ok,
    }


def load_predictions(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    rows = load_jsonl(path)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        cid = str(row.get("case_id") or "").strip()
        if not cid:
            raise ValueError(f"{path}: prediction row missing case_id")
        out[cid] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Planner GRPO rollout predictions.")
    parser.add_argument("--cases", type=Path, required=True, help="GRPO case JSONL")
    parser.add_argument("--predictions", type=Path, default=None, help="Optional rollout prediction JSONL")
    parser.add_argument("--out", type=Path, default=None, help="Optional report JSON path")
    args = parser.parse_args()

    cases = load_jsonl(args.cases)
    predictions = load_predictions(args.predictions)
    results = []
    by_category: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        cid = str(case.get("case_id") or "")
        result = score_case(
            case,
            predictions.get(cid),
            use_expected_when_missing=args.predictions is None,
        )
        results.append(result)
        by_category[str(case.get("category") or "unknown")].append(float(result["score"]))

    summary = {
        "cases": len(cases),
        "prediction_mode": "provided" if args.predictions else "expected_decisions_smoke",
        "mean_score": round(sum(float(r["score"]) for r in results) / max(1, len(results)), 6),
        "by_category": {
            key: {
                "count": len(values),
                "mean_score": round(sum(values) / max(1, len(values)), 6),
            }
            for key, values in sorted(by_category.items())
        },
        "used_action_counts": dict(
            Counter(action for result in results for action in result.get("used_actions", []))
        ),
    }
    report = {"summary": summary, "results": results}
    if args.out:
        write_json(args.out, report)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
