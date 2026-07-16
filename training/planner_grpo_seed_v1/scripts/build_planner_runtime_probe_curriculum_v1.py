#!/usr/bin/env python3
"""Build a strict, action-dominant curriculum for runtime probe continuation."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capa.evaluation.dataset_audit import load_jsonl, step_stats  # noqa: E402
from training.planner_grpo_seed_v1.scripts.build_planner_sft_data import (  # noqa: E402
    build_rows,
)


SOURCE_ID = "planner_runtime_routing_v1"
DATASET_ID = "planner_runtime_probe_curriculum_v1"
CASE_DIR = ROOT / "training/planner_grpo_seed_v1/cases"
STEP_DIR = (
    ROOT
    / "training/planner_grpo_seed_v1/sft_data_runtime_probe_curriculum_v1_chatml"
)

STRICT_REWARD = {
    "json_valid": 0.02,
    "decision_type_valid": 0.03,
    "action_match": 0.55,
    "argument_match": 0.25,
    "finish_after_tool": 0.10,
    "no_forbidden_action": 0.05,
    "wrong_action_cap": 0.20,
    "strict_action_match": True,
    "no_premature_stop": 0.10,
    "no_skip_required_probe": 0.15,
    "final_tool_finish": 0.05,
}

PRIMARY_TEMPLATES = [
    "流程顺序不能颠倒：先让 Qwen 对附件中的{target}做一次探针；拿到结果后，再完成{project}的迁移可行性报告。",
    "即使最终要评估{project}的能力迁移，也必须先用 Qwen 检查样例里的{target}，探针不能被后续方案跳过。",
    "这是串行任务：第一阶段 Qwen 单图检测{target}且不要结束，第二阶段结合图片给{project}提出低成本迁移路径。",
]

CONTRAST_TEMPLATES = [
    "本轮只有视觉探针：用 Qwen 检查附件中的{target}后立即停止，{project}的迁移方案以后再说。",
    "请勿因为提到{project}就进入迁移顾问；现在只需 Qwen 返回{target}的单图检测结果。",
]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _tokens(case: dict[str, Any], project: str) -> tuple[str, str]:
    decisions = case.get("expected_decisions") or []
    first_contains = decisions[0].get("arg_contains") or {}
    label_tokens = first_contains.get("label") or ["目标"]
    target = str(label_tokens[0] if isinstance(label_tokens, list) else label_tokens)
    if len(decisions) > 1:
        second_contains = decisions[1].get("arg_contains") or {}
        query_tokens = second_contains.get("user_query") or []
        if isinstance(query_tokens, list) and query_tokens:
            project = str(query_tokens[0])
    return target, project


def _with_strict_reward(case: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(case)
    result["reward_spec"] = dict(STRICT_REWARD)
    result["training_view"] = DATASET_ID
    return result


def _augment_train(source: list[dict[str, Any]]) -> list[dict[str, Any]]:
    project_by_entity: dict[str, str] = {}
    for case in source:
        if case.get("category") != "migration_advisor":
            continue
        decisions = case.get("expected_decisions") or []
        contains = (decisions[0].get("arg_contains") or {}) if decisions else {}
        tokens = contains.get("user_query") or []
        if isinstance(tokens, list) and tokens:
            project_by_entity[str(case.get("entity_id") or "")] = str(tokens[0])

    rows: list[dict[str, Any]] = []
    project_categories = {
        "qwen_probe",
        "rex_probe",
        "qwen_probe_only_contrast",
        "pipeline_eval",
    }
    for case in source:
        item = _with_strict_reward(case)
        project = project_by_entity.get(str(case.get("entity_id") or ""), "目标项目")
        if case.get("category") in project_categories and project not in item["user_query"]:
            item["user_query"] = f"{item['user_query']} 项目背景：{project}。"
        rows.append(item)

    for case in source:
        category = str(case.get("category") or "")
        if category not in {
            "qwen_probe_then_migration",
            "qwen_probe_only_contrast",
        }:
            continue
        project = project_by_entity.get(str(case.get("entity_id") or ""), "目标项目")
        target, project = _tokens(case, project)
        templates = (
            PRIMARY_TEMPLATES
            if category == "qwen_probe_then_migration"
            else CONTRAST_TEMPLATES
        )
        for variant, template in enumerate(templates, start=1):
            augmented = _with_strict_reward(case)
            source_id = str(case["case_id"])
            augmented["case_id"] = f"{source_id}-CURR-{variant}"
            augmented["template_id"] = f"{category}_curriculum_train_{variant}"
            augmented["user_query"] = template.format(
                target=target,
                project=project,
            )
            augmented["provenance_class"] = (
                "counterfactual_training_paraphrase_from_synthetic_source"
            )
            rows.append(augmented)
    return rows


def _case_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cases": len(rows),
        "steps": sum(len(row.get("expected_decisions") or []) for row in rows),
        "entity_groups": len({str(row.get("entity_id") or "") for row in rows}),
        "categories": dict(
            sorted(Counter(str(row.get("category") or "") for row in rows).items())
        ),
        "exact_queries": len({str(row.get("user_query") or "") for row in rows}),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, default=CASE_DIR)
    parser.add_argument("--step-dir", type=Path, default=STEP_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report: dict[str, Any] = {
        "dataset_id": DATASET_ID,
        "source_dataset_id": SOURCE_ID,
        "reward": STRICT_REWARD,
        "splits": {},
    }
    for split in ("train", "dev", "test"):
        source_path = args.case_dir / f"{SOURCE_ID}_{split}_cases.jsonl"
        source = load_jsonl(source_path)
        cases = _augment_train(source) if split == "train" else [
            _with_strict_reward(case) for case in source
        ]
        case_path = args.case_dir / f"{DATASET_ID}_{split}_cases.jsonl"
        _write_jsonl(case_path, cases)
        steps = build_rows(
            cases,
            indent=-1,
            prompt_format="qwen_chatml",
            append_im_end=True,
        )
        step_path = args.step_dir / f"{split}.jsonl"
        _write_jsonl(step_path, steps)
        report["splits"][split] = {
            "cases": _case_summary(cases),
            "steps": step_stats(steps),
            "source_cases": str(source_path),
            "case_file": str(case_path),
            "step_file": str(step_path),
        }
    args.step_dir.mkdir(parents=True, exist_ok=True)
    (args.step_dir / "metadata.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
