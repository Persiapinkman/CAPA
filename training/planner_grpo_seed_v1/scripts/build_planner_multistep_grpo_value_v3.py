#!/usr/bin/env python3
"""Build a route-centric Planner challenge intended to expose GRPO value.

Calibration contains paired observation counterfactuals for every entity.  Whole
families may be admitted from calibration, but confirmation is generated exactly
once from a frozen eight-family allowlist and is never filtered case by case.
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


DATASET_ID = "planner_multistep_grpo_value_v3"
CASE_DIR = ROOT / "training/planner_grpo_seed_v1/cases"
STEP_DIR = ROOT / "training/planner_grpo_seed_v1/sft_data_multistep_grpo_value_v3_qwen35_chatml"
DATASET_DIR = ROOT / "data/datasets" / DATASET_ID
SEED = 20260715
CALIBRATION_ENTITY_COUNT = 24
CONFIRMATION_ENTITY_COUNT = 75
CONFIRMATION_FAMILY_COUNT = 8
DEFAULT_CONFIRMATION_CASES = CONFIRMATION_ENTITY_COUNT * CONFIRMATION_FAMILY_COUNT

BRANCHES = (
    "accept_all_gates_stop",
    "domain_veto_migrate",
    "iou_veto_migrate",
    "empty_veto_migrate",
    "retry_accept_stop",
    "retry_domain_migrate",
    "retry_iou_migrate",
    "hard_failure_migrate",
)
SCENARIOS = tuple(
    f"{detector}_{branch}"
    for branch in BRANCHES
    for detector in ("qwen", "rex")
)
SCENARIO_CODES = {
    scenario: f"{'Q' if scenario.startswith('qwen_') else 'R'}{index + 1:02d}"
    for index, scenario in enumerate(SCENARIOS)
}

VALUE_REWARD = {
    "json_valid": 0.02,
    "decision_type_valid": 0.03,
    "action_match": 0.65,
    "argument_match": 0.10,
    "finish_after_tool": 0.10,
    "no_forbidden_action": 0.10,
    "wrong_action_cap": 0.20,
    "strict_action_match": True,
    "strict_argument_types": True,
    "no_premature_stop": 0.10,
    "no_repeated_tool": 0.10,
    "no_skip_required_probe": 0.10,
    "final_tool_finish": 0.10,
}

CALIBRATION_FIXTURES = (
    ("持竿人员", "examples/images/fisherman.jpg", "v3_calibration_fisherman"),
    ("清运车辆", "examples/images/trash_truck.jpg", "v3_calibration_trash_truck"),
)
CONFIRMATION_FIXTURES = (
    ("烟雾", "examples/images/smoke.jpg", "v3_confirmation_smoke"),
    ("背包", "examples/images/person_with_bag.png", "v3_confirmation_person_with_bag"),
    ("横幅", "examples/images/banner.jpg", "v3_confirmation_banner"),
)

SITE_ROOTS = (
    "海岛储能站",
    "冷轧钢卷库",
    "跨境物流闸口",
    "高原光伏场",
    "港区危化品通道",
    "城市燃气巡检廊",
    "智能制造装配岛",
    "轨道交通检修库",
    "沿江散货码头",
    "森林防火瞭望区",
    "大型算力中心",
    "新能源整车库",
    "生物医药冷库",
    "船闸控制区",
    "水利枢纽廊道",
    "机场特种车辆区",
    "高层建筑设备层",
    "海关查验平台",
    "矿山破碎转运站",
    "城市应急物资库",
)
PROJECT_SUFFIXES = (
    "软边界验收",
    "跨域能力复核",
    "低样本迁移决策",
    "现场灰度准入",
    "视觉能力接入评审",
    "长尾风险验证",
    "存量模型复用审查",
    "边缘部署能力核验",
)

CALIBRATION_STYLES = (
    "请执行本轮分支验收",
    "按当前批次的硬门槛处理",
    "请完成这次结果驱动的串行判断",
    "依据本轮状态表推进，不要套用历史结论",
)
CONFIRMATION_STYLES = (
    "按本次上线检查单推进",
    "请完成当前现场样本的分阶段判定",
    "本轮只认最新工具状态，请据此处理",
    "依据当前验收码执行后续动作",
    "请按这次能力边界合同完成任务",
)

CURRENT_OBSERVATION_FORBIDDEN_HINTS = (
    "迁移",
    "migration_advisor",
    "应结束",
    "直接结束",
    "应重试",
    "再次调用",
)


def build_entities() -> dict[str, list[str]]:
    values = [
        f"{site}{suffix}{index + 701:03d}号"
        for index, (site, suffix) in enumerate(
            (pair for site in SITE_ROOTS for suffix in PROJECT_SUFFIXES for pair in [(site, suffix)])
        )
    ]
    random.Random(SEED).shuffle(values)
    return {
        "calibration": values[:CALIBRATION_ENTITY_COUNT],
        "confirmation": values[
            CALIBRATION_ENTITY_COUNT : CALIBRATION_ENTITY_COUNT + CONFIRMATION_ENTITY_COUNT
        ],
    }


ENTITY_SPLITS = build_entities()


def parse_scenarios(raw: str) -> list[str]:
    values = [value.strip() for value in re.split(r"[,\s]+", raw) if value.strip()]
    unknown = sorted(set(values) - set(SCENARIOS))
    if unknown:
        raise ValueError(f"unknown scenarios: {unknown}")
    return list(dict.fromkeys(values))


def detector_parts(scenario: str) -> tuple[str, str, str]:
    is_qwen = scenario.startswith("qwen_")
    return (
        "qwen_detection" if is_qwen else "rexomni_detection",
        "Qwen" if is_qwen else "Rex-Omni",
        "qwen" if is_qwen else "rex",
    )


def fixture_for(split: str, entity_index: int) -> tuple[str, str, str]:
    fixtures = CALIBRATION_FIXTURES if split == "calibration" else CONFIRMATION_FIXTURES
    return fixtures[entity_index % len(fixtures)]


def policy_query(
    *, split: str, style: str, project: str, target: str, model_text: str
) -> str:
    intro = (
        f"{style}：先用{model_text}检查附件中的{target}，检测只是中间步骤。"
        if split == "calibration"
        else f"{style}。第一步由{model_text}核验附件里的{target}，然后读取最新 observation。"
    )
    return (
        intro
        + f"本任务最终要判断{project}是否需要能力迁移。只使用当前 query 最新一步的字段，"
        "历史批次、overall_badge 和自然语言备注都不能覆盖硬门槛。规则按以下优先级执行："
        "若 gateway_error 不为 none，仅当 retryable=true 且 retry_count=0 时原检测工具重试一次；"
        "不可重试或已经重试过的技术错误转迁移顾问。无技术错误时，只有 candidate_count>=1、"
        "min_confidence>=0.88、cross_prompt_iou>=0.72 且 domain_shift=low 四项同时成立才直接收口；"
        "任何一项不成立都转迁移顾问。重试后的结果仍按同一四项门槛判断，禁止第二次重试。"
        "所有检测步骤 finish_after_tool=false；迁移顾问必须使用当前图片、开启内部视觉探针并在报告后结束。"
    )


def metric_observation(
    after_step: int,
    *,
    status: str,
    candidate_count: int | None,
    min_confidence: float | None,
    cross_prompt_iou: float | None,
    domain_shift: str,
    gateway_error: str,
    retryable: bool,
    retry_count: int,
    badge: str,
) -> dict[str, Any]:
    fields = [
        f"candidate_count={candidate_count if candidate_count is not None else 'NA'}",
        f"min_confidence={min_confidence if min_confidence is not None else 'NA'}",
        f"cross_prompt_iou={cross_prompt_iou if cross_prompt_iou is not None else 'NA'}",
        f"domain_shift={domain_shift}",
        f"gateway_error={gateway_error}",
        f"retryable={'true' if retryable else 'false'}",
        f"retry_count={retry_count}",
        f"overall_badge={badge}",
    ]
    return v1.observation(
        after_step,
        success=gateway_error == "none",
        status=status,
        summary="本轮结构化状态：" + "；".join(fields) + "。备注：badge 仅供看板展示。",
    )


def successful_observation(after_step: int, branch: str, *, retry_count: int) -> dict[str, Any]:
    if branch == "accept":
        return metric_observation(
            after_step,
            status="completed",
            candidate_count=2,
            min_confidence=0.91,
            cross_prompt_iou=0.83,
            domain_shift="low",
            gateway_error="none",
            retryable=False,
            retry_count=retry_count,
            badge="amber" if retry_count else "green",
        )
    if branch == "domain":
        return metric_observation(
            after_step,
            status="completed",
            candidate_count=2,
            min_confidence=0.95,
            cross_prompt_iou=0.86,
            domain_shift="high",
            gateway_error="none",
            retryable=False,
            retry_count=retry_count,
            badge="green",
        )
    if branch == "iou":
        return metric_observation(
            after_step,
            status="completed",
            candidate_count=3,
            min_confidence=0.94,
            cross_prompt_iou=0.43,
            domain_shift="low",
            gateway_error="none",
            retryable=False,
            retry_count=retry_count,
            badge="green",
        )
    if branch == "empty":
        return metric_observation(
            after_step,
            status="completed",
            candidate_count=0,
            min_confidence=0.97,
            cross_prompt_iou=0.81,
            domain_shift="low",
            gateway_error="none",
            retryable=False,
            retry_count=retry_count,
            badge="green",
        )
    raise ValueError(f"unknown success branch: {branch}")


def retryable_observation() -> dict[str, Any]:
    return metric_observation(
        1,
        status="gateway_error",
        candidate_count=None,
        min_confidence=None,
        cross_prompt_iou=None,
        domain_shift="unknown",
        gateway_error="timeout",
        retryable=True,
        retry_count=0,
        badge="red",
    )


def hard_failure_observation() -> dict[str, Any]:
    return metric_observation(
        1,
        status="gateway_error",
        candidate_count=None,
        min_confidence=None,
        cross_prompt_iou=None,
        domain_shift="unknown",
        gateway_error="invalid_payload",
        retryable=False,
        retry_count=0,
        badge="amber",
    )


def stale_history(
    *, scenario: str, detector_action: str, model_text: str, target: str, project: str
) -> list[dict[str, Any]]:
    if scenario.endswith("accept_all_gates_stop"):
        steps = [
            {
                "step_index": 1,
                "action": detector_action,
                "action_input": {"label": target, "finish_after_tool": False},
                "observation": {
                    "status": "completed",
                    "summary": "归档批次：domain_shift=high；overall_badge=red。",
                },
            },
            {
                "step_index": 2,
                "action": "migration_advisor",
                "action_input": {"user_query": project, "finish_after_tool": True},
                "observation": {"status": "completed", "summary": "归档报告已完成。"},
            },
        ]
    elif scenario.endswith("domain_veto_migrate"):
        steps = [
            {
                "step_index": 1,
                "action": detector_action,
                "action_input": {"label": target, "finish_after_tool": False},
                "observation": {
                    "status": "completed",
                    "summary": "归档批次：四项硬门槛均通过；overall_badge=green。",
                },
            }
        ]
    else:
        return []
    return [
        {
            "query_id": f"archived_{scenario}",
            "query": f"归档任务：用{model_text}复核{project}中的{target}",
            "result_summary": "这是历史批次，不代表当前 query 的结果。",
            "steps": steps,
        }
    ]


def migration_step(project: str) -> dict[str, Any]:
    return v1.tool_step(
        "migration_advisor",
        required={
            "use_image": True,
            "use_visual_probe": True,
            "finish_after_tool": True,
        },
        contains={"user_query": [project]},
    )


def detection_step(action: str, target: str) -> dict[str, Any]:
    return v1.tool_step(
        action,
        required={"finish_after_tool": False},
        contains={"label": [target]},
    )


def branch_spec(
    *, scenario: str, action: str, target: str, project: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    first = detection_step(action, target)
    if scenario.endswith("accept_all_gates_stop"):
        return [first, v1.end_step()], [successful_observation(1, "accept", retry_count=0)], 2
    if scenario.endswith("domain_veto_migrate"):
        return [first, migration_step(project)], [successful_observation(1, "domain", retry_count=0)], 2
    if scenario.endswith("iou_veto_migrate"):
        return [first, migration_step(project)], [successful_observation(1, "iou", retry_count=0)], 2
    if scenario.endswith("empty_veto_migrate"):
        return [first, migration_step(project)], [successful_observation(1, "empty", retry_count=0)], 2
    if scenario.endswith("retry_accept_stop"):
        return (
            [first, detection_step(action, target), v1.end_step()],
            [retryable_observation(), successful_observation(2, "accept", retry_count=1)],
            3,
        )
    if scenario.endswith("retry_domain_migrate"):
        return (
            [first, detection_step(action, target), migration_step(project)],
            [retryable_observation(), successful_observation(2, "domain", retry_count=1)],
            3,
        )
    if scenario.endswith("retry_iou_migrate"):
        return (
            [first, detection_step(action, target), migration_step(project)],
            [retryable_observation(), successful_observation(2, "iou", retry_count=1)],
            3,
        )
    if scenario.endswith("hard_failure_migrate"):
        return [first, migration_step(project)], [hard_failure_observation()], 2
    raise ValueError(f"unknown scenario: {scenario}")


def make_case(
    *, split: str, entity_index: int, project: str, scenario: str
) -> dict[str, Any]:
    target, fixture, fixture_family = fixture_for(split, entity_index)
    styles = CALIBRATION_STYLES if split == "calibration" else CONFIRMATION_STYLES
    style = styles[entity_index % len(styles)]
    action, model_text, detector = detector_parts(scenario)
    expected, observations, target_step = branch_spec(
        scenario=scenario,
        action=action,
        target=target,
        project=project,
    )
    entity_id = f"pmgv3_{split}_entity_{entity_index + 1:03d}"
    allowed = v1.allowed_actions(expected)
    return {
        "case_id": f"PMGV3-{split.upper()}-{entity_index + 1:03d}-{SCENARIO_CODES[scenario]}",
        "dataset_id": DATASET_ID,
        "split": split,
        "selection_role": (
            "visible_whole_family_calibration"
            if split == "calibration"
            else "frozen_confirmation_unfiltered"
        ),
        "entity_id": entity_id,
        "group_id": entity_id,
        "counterfactual_bundle_id": f"{entity_id}_{detector}",
        "template_id": f"policy_{detector}_{split}_{entity_index % len(styles) + 1}",
        "scenario_id": scenario,
        "category": scenario,
        "detector_family": detector,
        "branch_policy": scenario.removeprefix(f"{detector}_"),
        "grpo_target_step": target_step,
        "user_query": policy_query(
            split=split,
            style=style,
            project=project,
            target=target,
            model_text=model_text,
        ),
        "image_fixture_family": fixture_family,
        "setup": {
            **v1.setup(fixture=fixture),
            "query_trajectories": stale_history(
                scenario=scenario,
                detector_action=action,
                model_text=model_text,
                target=target,
                project=project,
            ),
        },
        "expected_decisions": expected,
        "mock_observations": observations,
        "forbidden_actions": [tool for tool in v1.ACTIVE_TOOLS if tool not in allowed],
        "reward_spec": VALUE_REWARD,
        "provenance_class": "v3_route_policy_counterfactual_from_v2_aggregate_failure_taxonomy",
    }


def build_split(
    *, split: str, accepted_scenarios: list[str], confirmation_cases: int
) -> list[dict[str, Any]]:
    if split == "calibration":
        scenarios = list(SCENARIOS)
        entities = ENTITY_SPLITS[split]
    else:
        if len(accepted_scenarios) != CONFIRMATION_FAMILY_COUNT:
            raise ValueError(
                f"confirmation requires exactly {CONFIRMATION_FAMILY_COUNT} frozen scenarios"
            )
        if confirmation_cases != DEFAULT_CONFIRMATION_CASES:
            raise ValueError(f"confirmation is frozen at {DEFAULT_CONFIRMATION_CASES} cases")
        scenarios = accepted_scenarios
        entities = ENTITY_SPLITS[split]
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
    if split == "confirmation" and len(rows) != confirmation_cases:
        raise ValueError(f"expected {confirmation_cases} confirmation rows, got {len(rows)}")
    return rows


def validate_rows(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    ids = [str(row.get("case_id") or "") for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        errors.append("case IDs must be unique and non-empty")
    for row in rows:
        cid = str(row.get("case_id") or "")
        expected = row.get("expected_decisions")
        observations = row.get("mock_observations")
        if not isinstance(expected, list) or len(expected) not in {2, 3}:
            errors.append(f"{cid}: expected_decisions must contain 2 or 3 steps")
            continue
        if not isinstance(observations, list) or len(observations) != len(expected) - 1:
            errors.append(f"{cid}: observation count must equal step count minus one")
        else:
            for index, item in enumerate(observations, start=1):
                if int(item.get("after_step") or 0) != index:
                    errors.append(f"{cid}: observation {index} must follow step {index}")
                summary = str((item.get("observation") or {}).get("summary") or "")
                for hint in CURRENT_OBSERVATION_FORBIDDEN_HINTS:
                    if hint in summary:
                        errors.append(f"{cid}: current observation leaks action hint {hint!r}")
        fixture = ROOT / str((row.get("setup") or {}).get("image_fixture") or "")
        if not fixture.is_file():
            errors.append(f"{cid}: missing fixture {fixture}")
        for step_index, step in enumerate(expected, start=1):
            if not isinstance(step, dict):
                errors.append(f"{cid}: step {step_index} must be an object")
                continue
            if str(step.get("decision_type") or "tool") == "tool":
                required = step.get("required_args") if isinstance(step.get("required_args"), dict) else {}
                if not isinstance(required.get("finish_after_tool"), bool):
                    errors.append(f"{cid}: step {step_index} requires boolean finish_after_tool")
    return errors


def qwen35_step_rows(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = build_rows(cases, indent=-1, prompt_format="qwen_chatml", append_im_end=True)
    old_suffix = "<|im_start|>assistant\n"
    marker = "<think>\n\n</think>\n\n"
    for row in rows:
        prompt = str(row.get("prompt") or "")
        if not prompt.endswith(old_suffix):
            raise ValueError(f"unexpected ChatML prompt suffix for {row.get('case_id')}")
        row["prompt"] = prompt + marker
    return rows


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
        errors = validate_rows(rows)
        if errors:
            raise ValueError("dataset validation failed:\n" + "\n".join(errors[:50]))
        case_path = args.case_dir / f"{DATASET_ID}_{split}_cases.jsonl"
        v1.write_jsonl(case_path, rows)
        steps = qwen35_step_rows(rows)
        step_path = args.step_dir / f"{split}.jsonl"
        v1.write_jsonl(step_path, steps)
        generated[split] = rows
        case_paths[split] = case_path
        step_paths[split] = step_path
        split_reports[split] = {
            "cases": case_stats(rows),
            "steps": step_stats(steps),
            "entity_groups": len({str(row.get("entity_id") or "") for row in rows}),
            "families": dict(sorted(Counter(str(row.get("scenario_id") or "") for row in rows).items())),
            "step_counts": dict(sorted(Counter(len(row["expected_decisions"]) for row in rows).items())),
            "case_file": str(case_path.relative_to(ROOT)),
            "step_file": str(step_path.relative_to(ROOT)),
        }

    current_paths = {path.resolve() for path in case_paths.values()}
    all_rows = [row for split in requested for row in generated[split]]
    overlap = v1.existing_case_overlap(all_rows, current_paths)
    if overlap["status"] != "pass":
        raise ValueError(f"existing dataset overlap detected: {overlap['overlaps']}")
    report = {
        "schema_version": "1.0",
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "selection_rule": "calibration-only whole-family selection; confirmation is unfiltered",
        "accepted_scenarios": accepted,
        "splits": split_reports,
        "integrity": {
            "existing_dataset_overlap": overlap,
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
