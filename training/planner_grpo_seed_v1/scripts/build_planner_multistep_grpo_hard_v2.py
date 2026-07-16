#!/usr/bin/env python3
"""Build the second calibration version of the Planner model-separation challenge.

V2 keeps the observation counterfactuals from V1 but concentrates the primary
families on the two empirically useful failure mechanisms: state-conditioned
detection-to-migration transitions and the final-stop/typed-argument contract of
image-backed migration advice.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from capa.evaluation.dataset_audit import case_stats, step_stats  # noqa: E402
from training.planner_grpo_seed_v1.scripts import (  # noqa: E402
    build_planner_multistep_grpo_hard_v1 as v1,
)
from training.planner_grpo_seed_v1.scripts.build_planner_sft_data import (  # noqa: E402
    build_rows,
)


DATASET_ID = "planner_multistep_grpo_hard_v2"
CASE_DIR = ROOT / "training/planner_grpo_seed_v1/cases"
STEP_DIR = ROOT / "training/planner_grpo_seed_v1/sft_data_multistep_grpo_hard_v2_chatml"
DATASET_DIR = ROOT / "data/datasets" / DATASET_ID
SEED = 20260715
CALIBRATION_ENTITY_COUNT = 32
DEFAULT_CONFIRMATION_CASES = 600

SCENARIOS = (
    "qwen_box_variance_to_migration",
    "qwen_empty_result_to_migration",
    "qwen_domain_shift_to_migration",
    "rex_box_variance_to_migration",
    "rex_empty_result_to_migration",
    "rex_domain_shift_to_migration",
    "qwen_confident_stop_guardrail",
    "rex_confident_stop_guardrail",
    "image_migration_capability_probe",
    "image_migration_validation_probe",
    "image_migration_risk_probe",
    "image_migration_data_probe",
)

PRIMARY_SCENARIOS = set(SCENARIOS) - {
    "qwen_confident_stop_guardrail",
    "rex_confident_stop_guardrail",
}

SCENARIO_CODES = {
    "qwen_box_variance_to_migration": "QVAR",
    "qwen_empty_result_to_migration": "QEMPTY",
    "qwen_domain_shift_to_migration": "QDOMAIN",
    "rex_box_variance_to_migration": "RVAR",
    "rex_empty_result_to_migration": "REMPTY",
    "rex_domain_shift_to_migration": "RDOMAIN",
    "qwen_confident_stop_guardrail": "QSTOP",
    "rex_confident_stop_guardrail": "RSTOP",
    "image_migration_capability_probe": "ICAP",
    "image_migration_validation_probe": "IVAL",
    "image_migration_risk_probe": "IRISK",
    "image_migration_data_probe": "IDATA",
}

BAD_CASE_TARGETS = {
    "qwen_box_variance_to_migration": ["qwen_to_migration", "final_stop_true", "box_variance_observation"],
    "qwen_empty_result_to_migration": ["qwen_to_migration", "final_stop_true", "empty_result_observation"],
    "qwen_domain_shift_to_migration": ["qwen_to_migration", "final_stop_true", "domain_shift_observation"],
    "rex_box_variance_to_migration": ["named_rex_to_migration", "final_stop_true", "box_variance_observation"],
    "rex_empty_result_to_migration": ["named_rex_to_migration", "final_stop_true", "empty_result_observation"],
    "rex_domain_shift_to_migration": ["named_rex_to_migration", "final_stop_true", "domain_shift_observation"],
    "qwen_confident_stop_guardrail": ["observation_conditioned_stop", "avoid_unnecessary_migration"],
    "rex_confident_stop_guardrail": ["observation_conditioned_stop", "avoid_unnecessary_migration"],
    "image_migration_capability_probe": ["use_image_true", "use_visual_probe_true", "final_stop_true"],
    "image_migration_validation_probe": ["use_image_true", "use_visual_probe_true", "final_stop_true"],
    "image_migration_risk_probe": ["use_image_true", "use_visual_probe_true", "final_stop_true"],
    "image_migration_data_probe": ["use_image_true", "use_visual_probe_true", "final_stop_true"],
}

CALIBRATION_STYLES = (
    "请按结果分支完成这项试点",
    "这是一项必须读取 observation 的串行任务",
    "按能力边界验收规则处理",
    "请严格执行以下条件路由",
)

CONFIRMATION_STYLES = (
    "请依据本轮工具状态推进",
    "按现场验收分支完成请求",
    "这轮需要先取样再判断下一步",
    "请遵循结果驱动的执行约定",
    "按业务边界处理下面的任务",
)


def build_entity_splits() -> dict[str, list[str]]:
    # V1 calibration exposed only its first 24 entities.  V2 starts from the
    # independent V1 confirmation pool and reshuffles it with a new seed.
    entities = list(v1.ENTITY_SPLITS["confirmation"])
    random.Random(SEED).shuffle(entities)
    return {
        "calibration": entities[:CALIBRATION_ENTITY_COUNT],
        "confirmation": entities[CALIBRATION_ENTITY_COUNT:],
    }


ENTITY_SPLITS = build_entity_splits()


def parse_scenarios(raw: str) -> list[str]:
    values = [value.strip() for value in re.split(r"[,\s]+", raw) if value.strip()]
    unknown = sorted(set(values) - set(SCENARIOS))
    if unknown:
        raise ValueError(f"unknown scenarios: {unknown}")
    return list(dict.fromkeys(values))


def fixture_for(split: str, entity_index: int) -> tuple[str, str, str]:
    fixtures = v1.CALIBRATION_FIXTURES if split == "calibration" else v1.CONFIRMATION_FIXTURES
    return fixtures[entity_index % len(fixtures)]


def adaptive_query(
    *, split: str, style: str, project: str, target: str, model_text: str
) -> str:
    if split == "calibration":
        return (
            f"{style}：第一步必须用{model_text}检查附件中的{target}，此步只取样，不能结束。"
            f"若 observation 显示候选框波动、零结果或明显域偏移，禁止重试或换检测模型，"
            f"直接转迁移顾问评估{project}；顾问必须使用当前图片并在内部再做视觉复核，"
            "顾问报告即最终交付。只有 observation 明确说明证据稳定充分时才直接结束。"
        )
    return (
        f"{style}。先由{model_text}对附件里的{target}做中间探测；结果若有框不稳定、"
        f"未返回目标或存在场景域偏移，都不要再次调用检测工具，应进入{project}的迁移评估。"
        "迁移顾问需结合图片并执行内部视觉探针，其输出完成后本轮收口；只有稳定充分的结果可直接结束。"
    )


def direct_query(
    *, split: str, style: str, project: str, target: str, scenario: str
) -> str:
    focus = {
        "image_migration_capability_probe": "现有能力边界和可复用资产",
        "image_migration_validation_probe": "最低成本验证路径和验收步骤",
        "image_migration_risk_probe": "误检漏检风险与工程约束",
        "image_migration_data_probe": "补数规模、长尾样本和标注需求",
    }[scenario]
    if split == "calibration":
        return (
            f"{style}：直接交给迁移顾问判断{project}能否覆盖附件中的{target}，重点给出{focus}。"
            "当前图片必须作为证据，并允许顾问内部进行视觉探针；不要拆成独立 Qwen/Rex 检测。"
            "迁移报告就是最终结果，完成后结束。"
        )
    return (
        f"{style}。请让迁移顾问直接分析{project}接入附件中{target}的方案，需覆盖{focus}。"
        "顾问应使用这张图片并在内部做视觉核验，不要先调用单图检测；报告产出后无需其它步骤。"
    )


def observation_for(
    *, scenario: str, model_text: str, project: str, target: str
) -> dict[str, Any]:
    if "box_variance" in scenario:
        status = "box_variance"
        summary = (
            f"{model_text} 成功返回{target}候选框，但更换等价目标描述后框数与位置明显波动；"
            f"根据原请求不得重试检测，应进入{project}的迁移评估并由顾问内部视觉复核。"
        )
    elif "empty_result" in scenario:
        status = "empty_result"
        summary = (
            f"{model_text} 调用成功但没有返回{target}候选框；这不等于能力不可迁移，"
            f"原请求要求直接进入{project}的迁移可行性评估，不得再次检测。"
        )
    elif "domain_shift" in scenario:
        status = "domain_shift"
        summary = (
            f"{model_text} 返回了{target}结果，但样例与现有部署域差异明显，当前结果不足以确认复用边界；"
            f"应转迁移顾问评估{project}并在顾问内部复核图片。"
        )
    else:
        status = "confident"
        summary = (
            f"{model_text} 对{target}的结果稳定充分，已满足原请求的直接确认分支；"
            "无需迁移顾问或其它工具，可直接结束。"
        )
    return v1.observation(
        1,
        success=True,
        status=status,
        summary=summary,
    )


def make_case(
    *, split: str, entity_index: int, project: str, scenario: str
) -> dict[str, Any]:
    target, fixture, fixture_family = fixture_for(split, entity_index)
    styles = CALIBRATION_STYLES if split == "calibration" else CONFIRMATION_STYLES
    style = styles[entity_index % len(styles)]
    entity_id = f"pmghv2_{split}_entity_{entity_index + 1:03d}"
    observations: list[dict[str, Any]] = []

    if scenario.startswith("qwen_") or scenario.startswith("rex_"):
        is_qwen = scenario.startswith("qwen_")
        action = "qwen_detection" if is_qwen else "rexomni_detection"
        model_text = "Qwen" if is_qwen else "Rex-Omni"
        query = adaptive_query(
            split=split,
            style=style,
            project=project,
            target=target,
            model_text=model_text,
        )
        expected = [
            v1.tool_step(
                action,
                required={"finish_after_tool": False},
                contains={"label": [target]},
            )
        ]
        if scenario.endswith("confident_stop_guardrail"):
            expected.append(
                {
                    "decision_type": "end",
                    "required_args": {},
                    "arg_contains": {},
                }
            )
        else:
            expected.append(
                v1.tool_step(
                    "migration_advisor",
                    required={
                        "use_image": True,
                        "use_visual_probe": True,
                        "finish_after_tool": True,
                    },
                    contains={"user_query": [project, target]},
                )
            )
        observations = [
            observation_for(
                scenario=scenario,
                model_text=model_text,
                project=project,
                target=target,
            )
        ]
        template_id = f"adaptive_{'qwen' if is_qwen else 'rex'}_{split}_{entity_index % len(styles) + 1}"
    else:
        query = direct_query(
            split=split,
            style=style,
            project=project,
            target=target,
            scenario=scenario,
        )
        expected = [
            v1.tool_step(
                "migration_advisor",
                required={
                    "use_image": True,
                    "use_visual_probe": True,
                    "finish_after_tool": True,
                },
                contains={"user_query": [project, target]},
            )
        ]
        template_id = f"{scenario}_{split}_{entity_index % len(styles) + 1}"

    allowed = v1.allowed_actions(expected)
    return {
        "case_id": f"PMGHV2-{split.upper()}-{entity_index + 1:03d}-{SCENARIO_CODES[scenario]}",
        "dataset_id": DATASET_ID,
        "split": split,
        "selection_role": (
            "visible_family_screening_v2"
            if split == "calibration"
            else "frozen_family_confirmation_unfiltered"
        ),
        "entity_id": entity_id,
        "group_id": entity_id,
        "template_id": template_id,
        "scenario_id": scenario,
        "scenario_tier": "primary_challenge" if scenario in PRIMARY_SCENARIOS else "counterfactual_guardrail",
        "category": scenario,
        "bad_case_targets": BAD_CASE_TARGETS[scenario],
        "user_query": query,
        "image_fixture_family": fixture_family,
        "setup": v1.setup(fixture=fixture),
        "expected_decisions": expected,
        "mock_observations": observations,
        "forbidden_actions": [tool for tool in v1.ACTIVE_TOOLS if tool not in allowed],
        "reward_spec": v1.STRICT_REWARD,
        "provenance_class": "v2_synthetic_refinement_from_v1_family_level_calibration",
    }


def build_split(
    *, split: str, accepted_scenarios: list[str], confirmation_cases: int
) -> list[dict[str, Any]]:
    if split == "calibration":
        scenarios = list(SCENARIOS)
        entities = ENTITY_SPLITS[split][:CALIBRATION_ENTITY_COUNT]
    else:
        if not accepted_scenarios:
            raise ValueError("confirmation requires a frozen --accepted-scenarios allowlist")
        scenarios = accepted_scenarios
        required_entities = math.ceil(confirmation_cases / len(scenarios))
        entities = ENTITY_SPLITS[split][:required_entities]
        if len(entities) < required_entities:
            raise ValueError("not enough independent confirmation entities")
    rows = [
        make_case(
            split=split,
            entity_index=entity_index,
            project=project,
            scenario=scenario,
        )
        for entity_index, project in enumerate(entities)
        for scenario in scenarios
    ]
    return rows if split == "calibration" else rows[:confirmation_cases]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", default="calibration")
    parser.add_argument("--accepted-scenarios", default="")
    parser.add_argument("--confirmation-cases", type=int, default=DEFAULT_CONFIRMATION_CASES)
    parser.add_argument("--case-dir", type=Path, default=CASE_DIR)
    parser.add_argument("--step-dir", type=Path, default=STEP_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested = [value.strip() for value in args.splits.split(",") if value.strip()]
    if not requested or set(requested) - {"calibration", "confirmation"}:
        raise ValueError("--splits must contain calibration and/or confirmation")
    accepted = parse_scenarios(args.accepted_scenarios)
    generated: dict[str, list[dict[str, Any]]] = {}
    case_paths: dict[str, Path] = {}
    step_paths: dict[str, Path] = {}
    split_reports: dict[str, Any] = {}
    for split in requested:
        rows = build_split(
            split=split,
            accepted_scenarios=accepted,
            confirmation_cases=args.confirmation_cases,
        )
        errors = v1.validate_rows(rows)
        if errors:
            raise ValueError("dataset validation failed:\n" + "\n".join(errors[:50]))
        case_path = args.case_dir / f"{DATASET_ID}_{split}_cases.jsonl"
        v1.write_jsonl(case_path, rows)
        steps = build_rows(rows, indent=-1, prompt_format="qwen_chatml", append_im_end=True)
        step_path = args.step_dir / f"{split}.jsonl"
        v1.write_jsonl(step_path, steps)
        generated[split] = rows
        case_paths[split] = case_path
        step_paths[split] = step_path
        split_reports[split] = {
            "cases": case_stats(rows),
            "steps": step_stats(steps),
            "entity_groups": len({str(row.get("entity_id") or "") for row in rows}),
            "scenario_tiers": dict(sorted(Counter(str(row.get("scenario_tier") or "") for row in rows).items())),
            "case_file": str(case_path.relative_to(ROOT)),
            "step_file": str(step_path.relative_to(ROOT)),
        }

    current_paths = {path.resolve() for path in case_paths.values()}
    all_rows = [row for split in requested for row in generated[split]]
    leakage = v1.existing_case_overlap(all_rows, current_paths)
    if leakage["status"] != "pass":
        raise ValueError(f"existing dataset overlap detected: {leakage['overlaps']}")
    report = {
        "schema_version": "1.0",
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "parent_calibration": "planner_multistep_grpo_hard_v1",
        "selection_rule": "whole-family calibration only; confirmation is never item-filtered",
        "accepted_scenarios": accepted,
        "splits": split_reports,
        "integrity": {
            "existing_dataset_overlap": leakage,
            "generated_split_isolation": v1.split_integrity(generated),
        },
        "files": {
            **{f"{split}_cases": str(path.relative_to(ROOT)) for split, path in case_paths.items()},
            **{f"{split}_steps": str(path.relative_to(ROOT)) for split, path in step_paths.items()},
        },
    }
    report["sha256"] = {
        name: v1.sha256(ROOT / relative_path) for name, relative_path in report["files"].items()
    }
    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    v1.write_json(args.dataset_dir / "build_report.json", report)
    v1.write_json(args.step_dir / "metadata.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
