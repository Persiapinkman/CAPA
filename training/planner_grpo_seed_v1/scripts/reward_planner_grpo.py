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

PROCESS_REWARD_KEYS = (
    "no_premature_stop",
    "no_repeated_tool",
    "no_skip_required_probe",
    "final_tool_finish",
)


ACTION_ALIASES = {
    "qwen-vlm-open-set-delection": "qwen_detection",
    "rexomni-open-set-detection": "rexomni_detection",
    "target-detection-evaluation": "pipeline_eval",
    "final_answer": "answerer",
}

ACTION_EQUIVALENTS = {
    "qwen_detection": {"qwen_detection", "rexomni_detection"},
    "rexomni_detection": {"qwen_detection", "rexomni_detection"},
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


def value_matches(actual: Any, expected: Any, *, strict_types: bool = False) -> bool:
    if isinstance(expected, bool):
        if strict_types:
            return isinstance(actual, bool) and actual is expected
        return bool(actual) is expected
    if isinstance(expected, int) and not isinstance(expected, bool):
        if strict_types:
            return isinstance(actual, int) and not isinstance(actual, bool) and actual == expected
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


def text_contains_any(actual: Any, tokens: list[Any]) -> bool:
    haystack = str(actual or "").lower()
    return any(str(token or "").strip().lower() in haystack for token in tokens if str(token or "").strip())


def get_arg(decision: dict[str, Any], key: str) -> Any:
    if key in {"clarification_question", "end_reason", "final_answer"}:
        return decision.get(key)
    action_input = decision.get("action_input") if isinstance(decision.get("action_input"), dict) else {}
    return action_input.get(key)


def score_expected_step(
    *,
    expected: dict[str, Any],
    actual: dict[str, Any] | None,
    reward_spec: dict[str, float],
) -> tuple[float, dict[str, Any]]:
    strict_argument_types = bool(reward_spec.get("strict_argument_types", False))
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

    if expected_decision_type in {"clarify", "end"}:
        expected_action = expected_decision_type
    else:
        expected_action = normalize_action(str(expected.get("action") or ""))
    if actual_decision_type in {"clarify", "end"}:
        actual_action = actual_decision_type
    else:
        actual_action = normalize_action(str(actual.get("action") or ""))
    accepted_actions = (
        {expected_action}
        if bool(reward_spec.get("strict_action_match", False))
        else ACTION_EQUIVALENTS.get(expected_action, {expected_action})
    )
    if actual_action in accepted_actions:
        detail["action_match"] = 1.0
    else:
        if len(accepted_actions) > 1:
            failures.append(f"action expected one of {sorted(accepted_actions)!r}, got {actual_action!r}")
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
            finish_hit = value_matches(
                get_arg(actual, key),
                expected_value,
                strict_types=strict_argument_types,
            )
            if not finish_hit:
                failures.append(f"finish_after_tool expected {expected_value!r}, got {get_arg(actual, key)!r}")
            continue
        arg_checks += 1
        if value_matches(
            get_arg(actual, key),
            expected_value,
            strict_types=strict_argument_types,
        ):
            arg_hits += 1
        else:
            failures.append(f"arg {key!r} expected {expected_value!r}, got {get_arg(actual, key)!r}")
    for key, tokens in arg_contains.items():
        token_list = tokens if isinstance(tokens, list) else [tokens]
        arg_checks += 1
        actual_arg = get_arg(actual, key)
        # arg_contains uses "any" semantics: the actual argument counts as
        # matching if it contains ANY of the listed token synonyms. This
        # matches the natural reading of "contains" and prevents synonym
        # lists from becoming logical-AND traps (which would force the
        # planner to emit every synonym at once).
        if str(actual_arg or "").strip() and text_contains_any(actual_arg, token_list):
            arg_hits += 1
        else:
            failures.append(f"arg {key!r} expected to contain {token_list!r}, got {actual_arg!r}")

    detail["argument_match"] = 1.0 if arg_checks == 0 else arg_hits / arg_checks
    detail["finish_after_tool"] = 1.0 if not finish_checked else float(finish_hit)

    score = 0.0
    for key in ("json_valid", "decision_type_valid", "action_match", "argument_match", "finish_after_tool"):
        score += float(reward_spec.get(key, DEFAULT_REWARD_SPEC.get(key, 0.0))) * detail[key]
    return score, {"detail": detail, "failures": failures}


STOP_ACTIONS = {"answerer", "final_answer"}
DETECTION_ACTIONS = {"qwen_detection", "rexomni_detection"}


def detect_premature_stop(
    expected_list: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> bool:
    """判定是否过早收口。

    对 expected 中的每个"实质性工具步"（decision_type=tool 且非 answerer/final_answer），
    检查实际决策是否在到达该步之前就 end 或跳到 answerer/final_answer。
    这精确惩罚 probe -> migration 这类残余失败中"探针后立即收口"的偏置。
    """
    for index, expected in enumerate(expected_list[:-1]):
        expected_type = str(expected.get("decision_type") or "tool").strip()
        expected_action = normalize_action(str(expected.get("action") or ""))
        if expected_type != "tool" or expected_action in STOP_ACTIONS:
            continue
        actual = decisions[index] if index < len(decisions) else None
        if not isinstance(actual, dict):
            # 期望的实质工具步没有对应决策，说明前面已提前终止。
            return True
        actual_type = str(actual.get("decision_type") or "").strip()
        actual_action = normalize_action(str(actual.get("action") or ""))
        if actual_type == "end":
            return True
        if actual_action in STOP_ACTIONS:
            return True
        if value_matches(get_arg(actual, "finish_after_tool"), True):
            return True
    return False


def _decision_action(decision: dict[str, Any] | None) -> str:
    if not isinstance(decision, dict):
        return ""
    if str(decision.get("decision_type") or "").strip() == "clarify":
        return "clarify"
    if str(decision.get("decision_type") or "").strip() == "end":
        return "final_answer"
    return normalize_action(str(decision.get("action") or ""))


def _expected_action(expected: dict[str, Any] | None) -> str:
    if not isinstance(expected, dict):
        return ""
    decision_type = str(expected.get("decision_type") or "tool").strip()
    if decision_type in {"clarify", "end"}:
        return decision_type
    return normalize_action(str(expected.get("action") or ""))


def _accepted_actions(expected_action: str) -> set[str]:
    return set(ACTION_EQUIVALENTS.get(expected_action, {expected_action}))


def detect_repeated_tool(
    expected_list: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> bool:
    """Penalize repeated same-tool loops when the expected sequence changes tools."""
    if len(expected_list) < 2 or len(decisions) < 2:
        return False
    expected_actions = [_expected_action(item) for item in expected_list]
    repeated_allowed = any(
        a and b and _accepted_actions(a).intersection(_accepted_actions(b))
        for a, b in zip(expected_actions, expected_actions[1:])
    )
    if repeated_allowed:
        return False
    for prev, curr in zip(decisions, decisions[1:]):
        prev_action = _decision_action(prev)
        curr_action = _decision_action(curr)
        if not prev_action or not curr_action:
            continue
        if prev_action in STOP_ACTIONS or curr_action in STOP_ACTIONS:
            continue
        if _accepted_actions(prev_action).intersection(_accepted_actions(curr_action)):
            return True
    return False


def detect_skip_required_probe(
    expected_list: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> bool:
    """Penalize jumping straight to migration/pipeline when step 1 must be detection."""
    if len(expected_list) < 2 or not decisions:
        return False
    first_expected = _expected_action(expected_list[0])
    second_expected = _expected_action(expected_list[1])
    if first_expected not in DETECTION_ACTIONS or second_expected != "migration_advisor":
        return False
    first_actual = _decision_action(decisions[0])
    return first_actual not in DETECTION_ACTIONS


def detect_final_tool_not_finished(
    expected_list: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    *,
    strict_types: bool = False,
) -> bool:
    """Penalize a correct final substantive tool that still sets finish_after_tool=false."""
    if not expected_list or not decisions:
        return False
    final_expected = expected_list[-1]
    final_actual = decisions[len(expected_list) - 1] if len(decisions) >= len(expected_list) else None
    if not isinstance(final_actual, dict):
        return False
    if _decision_action(final_actual) not in _accepted_actions(_expected_action(final_expected)):
        return False
    required_args = final_expected.get("required_args") if isinstance(final_expected.get("required_args"), dict) else {}
    if required_args.get("finish_after_tool") is not True:
        return False
    return value_matches(
        get_arg(final_actual, "finish_after_tool"),
        True,
        strict_types=strict_types,
    ) is False


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
            if decision_type == "end":
                required_args = exp.get("required_args") if isinstance(exp.get("required_args"), dict) else {}
                decisions.append(
                    {
                        "decision_type": "end",
                        "end_reason": str(required_args.get("end_reason") or "memory_hit"),
                        "final_answer": "",
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
            "passed": False,
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
        decision_type = str(decision.get("decision_type") or "").strip()
        if decision_type in {"clarify", "end"}:
            used_actions.append(decision_type)
        else:
            used_actions.append(normalize_action(str(decision.get("action") or "")))
    forbidden_hit = sorted(action for action in used_actions if action in forbidden)
    forbidden_weight = float(reward_spec.get("no_forbidden_action", 0.0))
    forbidden_score = 0.0 if forbidden_hit else forbidden_weight

    # 过程奖励：默认权重 0，仅当 case 显式声明时生效。
    premature_weight = float(reward_spec.get("no_premature_stop", 0.0))
    premature_stop_hit = (
        detect_premature_stop(expected_list, decisions) if premature_weight > 0 else False
    )
    premature_score = 0.0 if premature_stop_hit else premature_weight

    repeated_weight = float(reward_spec.get("no_repeated_tool", 0.0))
    repeated_tool_hit = (
        detect_repeated_tool(expected_list, decisions) if repeated_weight > 0 else False
    )
    repeated_score = 0.0 if repeated_tool_hit else repeated_weight

    skip_probe_weight = float(reward_spec.get("no_skip_required_probe", 0.0))
    skip_required_probe_hit = (
        detect_skip_required_probe(expected_list, decisions) if skip_probe_weight > 0 else False
    )
    skip_probe_score = 0.0 if skip_required_probe_hit else skip_probe_weight

    final_finish_weight = float(reward_spec.get("final_tool_finish", 0.0))
    final_tool_not_finished_hit = (
        detect_final_tool_not_finished(
            expected_list,
            decisions,
            strict_types=bool(reward_spec.get("strict_argument_types", False)),
        )
        if final_finish_weight > 0
        else False
    )
    final_finish_score = 0.0 if final_tool_not_finished_hit else final_finish_weight

    process_weight = sum(float(reward_spec.get(key, 0.0)) for key in PROCESS_REWARD_KEYS)
    total_possible = max_step_score + forbidden_weight + process_weight
    numerator = (
        raw_score
        + forbidden_score
        + premature_score
        + repeated_score
        + skip_probe_score
        + final_finish_score
    )
    total_score = numerator / total_possible if total_possible > 0 else 0.0
    if not parse_ok:
        total_score = 0.0

    passed = bool(round(total_score, 6) >= 1.0)
    return {
        "case_id": case.get("case_id"),
        "category": case.get("category"),
        "score": round(total_score, 6),
        "passed": passed,
        "raw_score": round(numerator, 6),
        "max_score": round(total_possible, 6),
        "forbidden_hit": forbidden_hit,
        "premature_stop_hit": premature_stop_hit,
        "repeated_tool_hit": repeated_tool_hit,
        "skip_required_probe_hit": skip_required_probe_hit,
        "final_tool_not_finished_hit": final_tool_not_finished_hit,
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
        "passed": sum(1 for r in results if r.get("passed") is True),
        "failed": sum(1 for r in results if r.get("passed") is False),
        "pass_rate": round(
            sum(1 for r in results if r.get("passed") is True) / max(1, len(results)),
            6,
        ),
        "by_category": {
            key: {
                "count": len(values),
                "mean_score": round(sum(values) / max(1, len(values)), 6),
                "passed": sum(
                    1
                    for result in results
                    if str(result.get("category") or "unknown") == key and result.get("passed") is True
                ),
                "pass_rate": round(
                    sum(
                        1
                        for result in results
                        if str(result.get("category") or "unknown") == key and result.get("passed") is True
                    )
                    / max(1, len(values)),
                    6,
                ),
            }
            for key, values in sorted(by_category.items())
        },
        "used_action_counts": dict(
            Counter(action for result in results for action in result.get("used_actions", []))
        ),
        "premature_stop_cases": sum(1 for result in results if result.get("premature_stop_hit")),
    }
    report = {"summary": summary, "results": results}
    if args.out:
        write_json(args.out, report)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
