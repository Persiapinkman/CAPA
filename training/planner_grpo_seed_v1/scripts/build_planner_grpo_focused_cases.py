#!/usr/bin/env python3
"""Build a focused Planner GRPO training set for the 4B model.

The full case file is useful as a regression suite. For GRPO, keep only the
soft boundaries that are not already better solved by runtime rules or SFT:
probe -> migration transition, its probe-only contrastive twin, and a small
clarify/guardrail slice.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = ROOT / "training" / "planner_grpo_seed_v1" / "cases" / "planner_grpo_train_cases.jsonl"
DEFAULT_OUT = ROOT / "training" / "planner_grpo_seed_v1" / "cases" / "planner_grpo_focused_4b_cases.jsonl"
DEFAULT_REPORT = ROOT / "training" / "planner_grpo_seed_v1" / "reports" / "planner_grpo_focused_4b_data_report.json"

FOCUSED_CATEGORIES = {
    "probe_then_migration",
    "probe_then_migration_strict",
    "probe_only_contrastive",
    "clarify_intent_ambiguity",
}

GUARDRAIL_LIMITS = {
    "single_image_probe": 12,
    "full_detection_eval": 5,
    "general_answer": 4,
    "historical_asset_qa": 4,
}

PROBE_PROCESS_REWARD = {
    "no_premature_stop": 0.20,
    "no_repeated_tool": 0.25,
    "no_skip_required_probe": 0.25,
    "final_tool_finish": 0.15,
}

PROBE_ONLY_PROCESS_REWARD = {
    "no_repeated_tool": 0.20,
}

CLARIFY_REWARD = {
    "json_valid": 0.10,
    "decision_type_valid": 0.30,
    "action_match": 0.50,
    "argument_match": 0.00,
    "finish_after_tool": 0.00,
    "no_forbidden_action": 0.10,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: row must be object")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def expected_actions(row: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for step in row.get("expected_decisions") if isinstance(row.get("expected_decisions"), list) else []:
        if not isinstance(step, dict):
            continue
        if str(step.get("decision_type") or "tool") == "clarify":
            out.append("clarify")
        else:
            out.append(str(step.get("action") or ""))
    return out


def strengthen_reward(row: dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    category = str(item.get("category") or "")
    reward_spec = dict(item.get("reward_spec") if isinstance(item.get("reward_spec"), dict) else {})
    actions = expected_actions(item)
    if category in {"probe_then_migration", "probe_then_migration_strict"}:
        reward_spec.update(PROBE_PROCESS_REWARD)
        item["grpo_focus"] = "probe_to_migration_process"
    elif category == "probe_only_contrastive":
        reward_spec.update(PROBE_ONLY_PROCESS_REWARD)
        item["grpo_focus"] = "probe_only_contrastive_guardrail"
    elif category == "clarify_intent_ambiguity":
        reward_spec.update(CLARIFY_REWARD)
        item["grpo_focus"] = "clarify_ambiguity"
    else:
        item["grpo_focus"] = "boundary_guardrail"
    item["reward_spec"] = reward_spec
    item["expected_action_signature"] = " -> ".join(actions)
    return item


def build_focused_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    guard_counts: Counter[str] = Counter()
    seen_queries: set[str] = set()
    for row in rows:
        category = str(row.get("category") or "")
        keep = category in FOCUSED_CATEGORIES
        if not keep and category in GUARDRAIL_LIMITS:
            keep = guard_counts[category] < GUARDRAIL_LIMITS[category]
        if not keep:
            continue
        query = str(row.get("user_query") or "").strip()
        if query in seen_queries:
            continue
        seen_queries.add(query)
        if category in GUARDRAIL_LIMITS:
            guard_counts[category] += 1
        selected.append(strengthen_reward(row))
    return selected


def build_report(*, source: Path, out: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_category = Counter(str(row.get("category") or "") for row in rows)
    by_focus = Counter(str(row.get("grpo_focus") or "") for row in rows)
    by_signature = Counter(str(row.get("expected_action_signature") or "") for row in rows)
    process_weights = Counter()
    for row in rows:
        spec = row.get("reward_spec") if isinstance(row.get("reward_spec"), dict) else {}
        for key in ("no_premature_stop", "no_repeated_tool", "no_skip_required_probe", "final_tool_finish"):
            if float(spec.get(key, 0.0) or 0.0) > 0:
                process_weights[key] += 1
    return {
        "source": str(source),
        "out": str(out),
        "count": len(rows),
        "by_category": dict(by_category),
        "by_focus": dict(by_focus),
        "by_expected_action_signature": dict(by_signature),
        "process_reward_case_counts": dict(process_weights),
        "intended_use": (
            "Focused GRPO training set for Qwen2.5-7B Planner. "
            "Use full planner_grpo_train_cases.jsonl as regression eval, not as the first GRPO train mix."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build focused 4B Planner GRPO cases.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    source = args.source if args.source.is_absolute() else ROOT / args.source
    out = args.out if args.out.is_absolute() else ROOT / args.out
    report = args.report if args.report.is_absolute() else ROOT / args.report
    rows = build_focused_rows(load_jsonl(source))
    write_jsonl(out, rows)
    write_json(report, build_report(source=source, out=out, rows=rows))
    print(json.dumps({"out": str(out), "report": str(report), "count": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
