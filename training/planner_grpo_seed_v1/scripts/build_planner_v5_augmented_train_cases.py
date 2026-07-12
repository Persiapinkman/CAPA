#!/usr/bin/env python3
"""Create train-only augmented hard cases for the Planner v5 refresh."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "demo"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from training.planner_grpo_seed_v1.scripts.build_planner_sft_data import write_json  # noqa: E402
from training.planner_grpo_seed_v1.scripts.train_planner_grpo import load_jsonl  # noqa: E402


DEFAULT_INPUT = ROOT / "training" / "planner_grpo_seed_v1" / "cases" / "planner_grpo_focused_train_v3_cases.jsonl"
DEFAULT_OUTPUT = ROOT / "training" / "planner_grpo_seed_v1" / "cases" / "planner_grpo_focused_train_v5_augmented_cases.jsonl"

CLARIFY_REWARD_SPEC = {
    "json_valid": 0.1,
    "decision_type_valid": 0.3,
    "action_match": 0.5,
    "argument_match": 0.0,
    "finish_after_tool": 0.0,
    "no_forbidden_action": 0.1,
}

DEFAULT_REWARD_SPEC = {
    "json_valid": 0.1,
    "decision_type_valid": 0.1,
    "action_match": 0.35,
    "argument_match": 0.25,
    "finish_after_tool": 0.1,
    "no_forbidden_action": 0.1,
}

CLARIFY_FORBIDDEN = [
    "qwen_detection",
    "rexomni_detection",
    "flux-image-generation",
    "pipeline_eval",
    "migration_advisor",
    "rag_answer",
    "answerer",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v5 augmented train-case JSONL.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--report-out",
        type=Path,
        default=ROOT / "training" / "planner_grpo_seed_v1" / "reports" / "planner_v5_augmented_train_cases_report.json",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def clarify_case(case_id: str, query: str, image_fixture: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "category": "clarify_intent_ambiguity",
        "grpo_focus": "hard_v5_augmented",
        "user_query": query,
        "setup": {
            "has_image": bool(image_fixture),
            "image_fixture": image_fixture,
            "query_trajectories": [],
        },
        "expected_action_signature": "clarify",
        "expected_decisions": [
            {
                "step": 1,
                "decision_type": "clarify",
                "action": "clarify",
                "arg_contains": {},
            }
        ],
        "reward_spec": dict(CLARIFY_REWARD_SPEC),
        "forbidden_actions": list(CLARIFY_FORBIDDEN),
    }


def pipeline_case(case_id: str, query: str, image_fixture: str, required_token: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "category": "full_detection_eval",
        "grpo_focus": "hard_v5_augmented",
        "user_query": query,
        "setup": {
            "has_image": True,
            "image_fixture": image_fixture,
            "query_trajectories": [],
        },
        "expected_action_signature": "pipeline_eval",
        "expected_decisions": [
            {
                "step": 1,
                "action": "pipeline_eval",
                "required_args": {"finish_after_tool": True},
                "arg_contains": {"task_text": [required_token]},
            }
        ],
        "reward_spec": dict(DEFAULT_REWARD_SPEC),
        "forbidden_actions": ["qwen_detection", "migration_advisor", "answerer"],
    }


def build_augmented_cases() -> list[dict[str, Any]]:
    assets = [
        ("烟雾", "examples/images/smoke.jpg"),
        ("横幅", "examples/images/banner.jpg"),
        ("厨师帽", "examples/images/person_with_bag.png"),
        ("反光条", "examples/images/person_with_bag.png"),
        ("垃圾车", "examples/images/trash_truck.jpg"),
        ("钓鱼的人", "examples/images/fisherman.jpg"),
    ]
    templates = [
        "这张图里的{label}，你帮我处理下。",
        "{label}这个，你看着处理。",
        "帮我把{label}这块搞一下。",
        "这张图，{label}，帮我弄一下。",
    ]
    rows: list[dict[str, Any]] = []
    idx = 1
    for label, image in assets:
        for template in templates:
            rows.append(clarify_case(f"GRPO-EXP-CLARIFY-AUG-V5-{idx:03d}", template.format(label=label), image))
            idx += 1

    pipeline_specs = [
        ("厨师帽", "examples/images/person_with_bag.png"),
        ("烟雾", "examples/images/smoke.jpg"),
        ("横幅", "examples/images/banner.jpg"),
        ("反光条", "examples/images/person_with_bag.png"),
        ("检测", "examples/images/fisherman.jpg"),
        ("检测", "examples/images/person_with_bag.png"),
    ]
    for idx, (token, image) in enumerate(pipeline_specs, start=1):
        if token == "检测":
            query = "请根据这张参考图扩增样本，比较两个开放集检测模型并输出报告。"
        else:
            query = f"请用这张参考图扩增样本，并做{token}开放集检测效果对比，输出评估报告。"
        rows.append(pipeline_case(f"GRPO-EXP-PIPELINE-AUG-V5-{idx:03d}", query, image, token))
    return rows


def main() -> None:
    args = parse_args()
    input_path = resolve(args.input)
    output_path = resolve(args.output)
    report_path = resolve(args.report_out)
    original = load_jsonl(input_path)
    original_ids = {str(row.get("case_id") or "") for row in original}
    augmented = build_augmented_cases()
    duplicate_ids = sorted(original_ids & {str(row.get("case_id") or "") for row in augmented})
    if duplicate_ids:
        raise RuntimeError(f"duplicate augmented ids: {duplicate_ids[:5]}")
    rows = original + augmented
    write_jsonl(output_path, rows)
    report = {
        "source": str(input_path),
        "output": str(output_path),
        "original_cases": len(original),
        "augmented_cases": len(augmented),
        "total_cases": len(rows),
        "augmented_categories": {
            "clarify_intent_ambiguity": sum(row["category"] == "clarify_intent_ambiguity" for row in augmented),
            "full_detection_eval": sum(row["category"] == "full_detection_eval" for row in augmented),
        },
        "leakage_guard": "Augmented rows are train-only and do not copy held-out val case IDs or exact held-out user queries.",
    }
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
