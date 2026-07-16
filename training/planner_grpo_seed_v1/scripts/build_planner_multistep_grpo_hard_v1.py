#!/usr/bin/env python3
"""Build a model-separation challenge for GRPO-suitable Planner routing cases.

The calibration split is intentionally visible and may be used to admit or reject
whole scenario families.  The confirmation split must be generated only from the
frozen family allowlist and must never be filtered at the individual-case level.
"""

from __future__ import annotations

import argparse
import hashlib
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

from capa.evaluation.dataset_audit import case_stats, normalize_query, step_stats  # noqa: E402
from training.planner_grpo_seed_v1.scripts.build_planner_sft_data import (  # noqa: E402
    build_rows,
)


DATASET_ID = "planner_multistep_grpo_hard_v1"
CASE_DIR = ROOT / "training/planner_grpo_seed_v1/cases"
STEP_DIR = ROOT / "training/planner_grpo_seed_v1/sft_data_multistep_grpo_hard_v1_chatml"
DATASET_DIR = ROOT / "data/datasets" / DATASET_ID
SEED = 20260714
CALIBRATION_ENTITY_COUNT = 24
DEFAULT_CONFIRMATION_CASES = 600

ACTIVE_TOOLS = [
    "rag_answer",
    "re_question",
    "answerer",
    "flux-image-generation",
    "qwen_detection",
    "rexomni_detection",
    "pipeline_eval",
    "migration_advisor",
]

STRICT_REWARD = {
    "json_valid": 0.02,
    "decision_type_valid": 0.03,
    "action_match": 0.60,
    "argument_match": 0.20,
    "finish_after_tool": 0.10,
    "no_forbidden_action": 0.05,
    "wrong_action_cap": 0.20,
    "strict_action_match": True,
    "strict_argument_types": True,
    "no_premature_stop": 0.10,
    "no_repeated_tool": 0.10,
    "no_skip_required_probe": 0.10,
    "final_tool_finish": 0.10,
}

SCENARIOS = (
    "qwen_uncertain_to_migration",
    "qwen_confident_stop",
    "qwen_failure_retry",
    "rex_uncertain_to_migration",
    "rex_confident_stop",
    "rex_failure_retry",
    "image_migration_internal_probe",
    "image_migration_no_probe",
    "text_migration",
    "pipeline_full_eval",
    "flux_generation_only",
    "visual_intent_clarify",
)

PRIMARY_SCENARIOS = {
    "qwen_uncertain_to_migration",
    "qwen_confident_stop",
    "qwen_failure_retry",
    "rex_uncertain_to_migration",
    "rex_confident_stop",
    "rex_failure_retry",
    "image_migration_internal_probe",
    "image_migration_no_probe",
    "text_migration",
    "pipeline_full_eval",
}

SCENARIO_CODES = {
    "qwen_uncertain_to_migration": "QUM",
    "qwen_confident_stop": "QCS",
    "qwen_failure_retry": "QFR",
    "rex_uncertain_to_migration": "RUM",
    "rex_confident_stop": "RCS",
    "rex_failure_retry": "RFR",
    "image_migration_internal_probe": "IMIP",
    "image_migration_no_probe": "IMNP",
    "text_migration": "TMIG",
    "pipeline_full_eval": "PIPE",
    "flux_generation_only": "FLUX",
    "visual_intent_clarify": "CLAR",
}

BAD_CASE_TARGETS = {
    "qwen_uncertain_to_migration": [
        "finish_after_tool_false_then_true",
        "observation_conditioned_transition",
        "qwen_to_migration",
    ],
    "qwen_confident_stop": [
        "finish_after_tool_false_then_end",
        "observation_conditioned_stop",
        "avoid_unnecessary_migration",
    ],
    "qwen_failure_retry": [
        "tool_failure_retry",
        "typed_finish_boolean",
        "allowed_same_tool_retry",
    ],
    "rex_uncertain_to_migration": [
        "finish_after_tool_false_then_true",
        "observation_conditioned_transition",
        "named_rex_to_migration",
    ],
    "rex_confident_stop": [
        "finish_after_tool_false_then_end",
        "observation_conditioned_stop",
        "avoid_unnecessary_migration",
    ],
    "rex_failure_retry": [
        "tool_failure_retry",
        "typed_finish_boolean",
        "allowed_same_tool_retry",
    ],
    "image_migration_internal_probe": [
        "use_image_true",
        "use_visual_probe_true",
        "exact_final_stop",
    ],
    "image_migration_no_probe": [
        "use_image_true",
        "use_visual_probe_false",
        "exact_final_stop",
    ],
    "text_migration": [
        "use_image_false",
        "use_visual_probe_false",
        "exact_final_stop",
    ],
    "pipeline_full_eval": [
        "pipeline_vs_probe_boundary",
        "pipeline_vs_flux_boundary",
        "exact_final_stop",
    ],
    "flux_generation_only": ["flux_vs_pipeline_guardrail", "side_effect_boundary"],
    "visual_intent_clarify": ["clarify_before_side_effect", "intent_ambiguity"],
}

CALIBRATION_FIXTURES = (
    ("手持钓竿的人员", "examples/images/fisherman.jpg", "fisherman"),
    ("垃圾清运车", "examples/images/trash_truck.jpg", "trash_truck"),
)

CONFIRMATION_FIXTURES = (
    ("可见烟雾", "examples/images/smoke.jpg", "smoke"),
    ("肩背包", "examples/images/person_with_bag.png", "person_with_bag"),
    ("悬挂横幅", "examples/images/banner.jpg", "banner"),
)

SITE_ROOTS = (
    "海上风电运维区",
    "城市地下管廊",
    "港口冷藏堆场",
    "医院物流通道",
    "机场行李分拣区",
    "铁路货运编组站",
    "锂电池生产车间",
    "数据中心机房",
    "山区输电走廊",
    "粮食筒仓作业区",
    "水泥厂装卸区",
    "天然气调压站",
    "污水处理厂",
    "半导体洁净走廊",
    "汽车电池仓库",
    "跨海大桥检修道",
    "大型商业综合体",
    "航空维修机库",
    "自动化立体仓库",
    "化工园区罐区",
    "隧道机电设备间",
    "水电站设备层",
    "市政泵站",
    "沿海防波堤",
    "露天矿运输道路",
    "农业温室大棚",
    "大型会展中心",
    "食品冷链仓库",
    "地铁车辆段",
    "船舶涂装车间",
)

PROJECT_SUFFIXES = (
    "巡检能力扩展",
    "开放集识别升级",
    "低成本迁移验证",
    "视觉告警改造",
    "样例驱动能力评估",
    "边缘端部署预研",
    "长尾目标补充",
    "现有模型复用",
    "风险识别增强",
    "新类别上线评审",
    "业务边界核验",
    "小样本能力迁移",
    "异常目标覆盖",
    "检测策略迭代",
    "试点验收准备",
    "模型能力盘点",
    "现场样例验证",
    "工程成本评估",
    "增量场景接入",
    "视觉方案选型",
)

SPLIT_STYLES = {
    "calibration": (
        "请按下面的条件执行",
        "这次必须遵守分支约束",
        "请完成这个串行判断",
        "按现场试点规则处理",
    ),
    "confirmation": (
        "这轮按业务条件推进",
        "请依据工具返回状态继续",
        "按验收约定完成该请求",
        "请处理以下分阶段任务",
        "本次按结果分支执行",
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def tool_step(
    action: str,
    *,
    required: dict[str, Any],
    contains: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    return {
        "decision_type": "tool",
        "action": action,
        "required_args": required,
        "arg_contains": contains or {},
    }


def end_step() -> dict[str, Any]:
    return {
        "decision_type": "end",
        "required_args": {"end_reason": "memory_hit"},
        "arg_contains": {},
    }


def clarify_step() -> dict[str, Any]:
    return {
        "decision_type": "clarify",
        "required_args": {},
        "arg_contains": {},
    }


def setup(*, fixture: str = "") -> dict[str, Any]:
    return {
        "has_image": bool(fixture),
        "image_fixture": fixture,
        "query_trajectories": [],
    }


def observation(after_step: int, *, success: bool, summary: str, status: str) -> dict[str, Any]:
    return {
        "after_step": after_step,
        "observation": {
            "success": success,
            "status": status,
            "summary": summary,
        },
    }


def allowed_actions(expected: list[dict[str, Any]]) -> set[str]:
    return {
        str(step.get("action") or "")
        for step in expected
        if str(step.get("decision_type") or "tool") == "tool"
        and str(step.get("action") or "")
    }


def fixture_for(split: str, entity_index: int) -> tuple[str, str, str]:
    fixtures = CALIBRATION_FIXTURES if split == "calibration" else CONFIRMATION_FIXTURES
    return fixtures[entity_index % len(fixtures)]


def build_entities() -> dict[str, list[str]]:
    entities = [
        f"{site}{suffix}{index + 1:03d}号项目"
        for index, (site, suffix) in enumerate(
            (pair for site in SITE_ROOTS for pair_suffix in PROJECT_SUFFIXES for pair in [(site, pair_suffix)])
        )
    ]
    random.Random(SEED).shuffle(entities)
    return {
        "calibration": entities[:CALIBRATION_ENTITY_COUNT],
        "confirmation": entities[CALIBRATION_ENTITY_COUNT:],
    }


ENTITY_SPLITS = build_entities()


def adaptive_probe_query(
    *, split: str, style: str, project: str, target: str, model_text: str
) -> str:
    if split == "calibration":
        return (
            f"{style}：先用{model_text}对附件里的{target}做一次轻量探针，探针只是中间步骤，"
            f"不要提前收口。若返回结果不稳定或不足以支持结论，继续为{project}给出迁移可行性方案；"
            "若结果稳定明确，就直接使用探针结果结束；若工具发生技术失败，只重试同一检测模型一次后结束。"
        )
    return (
        f"{style}。附件中的{target}先交给{model_text}快速核验，本次调用后仍要读取状态再决定："
        f"低置信或表述波动时转入{project}的能力迁移评估，证据充分时直接收口，"
        "瞬时调用失败时仅允许原检测工具重试一次。"
    )


def direct_query(
    *, split: str, scenario: str, style: str, project: str, target: str
) -> str:
    if split == "calibration":
        templates = {
            "image_migration_internal_probe": (
                "{style}：直接让迁移顾问评估{project}能否覆盖附件中的{target}，图片要作为样例证据，"
                "并允许迁移顾问内部做轻量视觉探针；不要先单独调用检测工具。"
            ),
            "image_migration_no_probe": (
                "{style}：附件只作为{target}现象说明，交给迁移顾问分析{project}的成本、风险与数据需求；"
                "禁止再运行任何视觉探针或独立检测。"
            ),
            "text_migration": (
                "{style}：不依赖图片，评估{project}新增{target}能力的复用边界、补数规模、风险和低成本验证路径。"
            ),
            "pipeline_full_eval": (
                "{style}：以附件中的{target}为参考扩增样本，同时比较 Qwen 与 Rex-Omni，"
                "输出误检漏检、精度结论和完整评测报告。"
            ),
            "flux_generation_only": (
                "{style}：只基于附件生成一张{target}在{project}中的变化图，不做检测、模型比较或迁移分析。"
            ),
            "visual_intent_clarify": (
                "{style}：把附件里的{target}处理一下，让它适合{project}，具体是检测、生成新图还是做迁移方案由你看着办。"
            ),
        }
    else:
        templates = {
            "image_migration_internal_probe": (
                "{style}。请由迁移顾问直接判断{project}接入{target}的可行性，需使用当前图片，"
                "并可在顾问内部进行视觉探测；不要拆出单图检测步骤。"
            ),
            "image_migration_no_probe": (
                "{style}。结合附件说明的{target}需求，为{project}做迁移边界与工程代价报告，"
                "但本轮不得执行视觉探针，图片只作上下文证据。"
            ),
            "text_migration": (
                "{style}。当前没有样例图，请给出{project}支持{target}所需的数据、成本、风险及能力迁移判断。"
            ),
            "pipeline_full_eval": (
                "{style}。围绕附件里的{target}完成端到端评测：补生成样本、运行两种检测模型、"
                "汇总 precision/recall 与误检漏检报告。"
            ),
            "flux_generation_only": (
                "{style}。参考当前图片仅制作一张{project}场景的{target}合成图，生成完即止，"
                "不要启动检测流水线。"
            ),
            "visual_intent_clarify": (
                "{style}。关于附件中的{target}，帮我做成{project}需要的样子；我还没决定是要识别、"
                "生成图片还是评估能力迁移。"
            ),
        }
    return templates[scenario].format(
        style=style,
        project=project,
        target=target,
    )


def make_case(
    *, split: str, entity_index: int, project: str, scenario: str
) -> dict[str, Any]:
    target, fixture, fixture_family = fixture_for(split, entity_index)
    style_values = SPLIT_STYLES[split]
    style = style_values[entity_index % len(style_values)]
    entity_id = f"pmghv1_{split}_entity_{entity_index + 1:03d}"
    expected: list[dict[str, Any]]
    observations: list[dict[str, Any]] = []
    image_fixture = fixture

    if scenario.startswith("qwen_") or scenario.startswith("rex_"):
        is_qwen = scenario.startswith("qwen_")
        action = "qwen_detection" if is_qwen else "rexomni_detection"
        model_text = "Qwen" if is_qwen else "Rex-Omni"
        query = adaptive_probe_query(
            split=split,
            style=style,
            project=project,
            target=target,
            model_text=model_text,
        )
        expected = [
            tool_step(
                action,
                required={"finish_after_tool": False},
                contains={"label": [target]},
            )
        ]
        if scenario.endswith("uncertain_to_migration"):
            expected.append(
                tool_step(
                    "migration_advisor",
                    required={
                        "use_image": True,
                        "use_visual_probe": True,
                        "finish_after_tool": True,
                    },
                    contains={"user_query": [project, target, "迁移"]},
                )
            )
            observations = [
                observation(
                    1,
                    success=True,
                    status="uncertain",
                    summary=(
                        f"{model_text} 已返回候选框，但不同目标表述下结果明显波动，"
                        f"当前证据不足以判断{project}的能力边界，需要进入迁移可行性评估。"
                    ),
                )
            ]
        elif scenario.endswith("confident_stop"):
            expected.append(end_step())
            observations = [
                observation(
                    1,
                    success=True,
                    status="confident",
                    summary=(
                        f"{model_text} 对{target}的结果稳定明确，已满足原请求中的直接确认分支；"
                        "无需迁移分析或其它工具，可使用当前结果收口。"
                    ),
                )
            ]
        else:
            expected.append(
                tool_step(
                    action,
                    required={"finish_after_tool": True},
                    contains={"label": [target]},
                )
            )
            observations = [
                observation(
                    1,
                    success=False,
                    status="retryable_failure",
                    summary=(
                        f"{model_text} 因瞬时网关错误未产生{target}结果；"
                        "原请求允许仅重试同一检测工具一次。"
                    ),
                )
            ]
        template_id = (
            f"adaptive_{'qwen' if is_qwen else 'rex'}_{split}_"
            f"{entity_index % len(style_values) + 1}"
        )
    else:
        query = direct_query(
            split=split,
            scenario=scenario,
            style=style,
            project=project,
            target=target,
        )
        template_id = f"{scenario}_{split}_{entity_index % len(style_values) + 1}"
        if scenario == "image_migration_internal_probe":
            expected = [
                tool_step(
                    "migration_advisor",
                    required={
                        "use_image": True,
                        "use_visual_probe": True,
                        "finish_after_tool": True,
                    },
                    contains={"user_query": [project, target]},
                )
            ]
        elif scenario == "image_migration_no_probe":
            expected = [
                tool_step(
                    "migration_advisor",
                    required={
                        "use_image": True,
                        "use_visual_probe": False,
                        "finish_after_tool": True,
                    },
                    contains={"user_query": [project, target]},
                )
            ]
        elif scenario == "text_migration":
            image_fixture = ""
            expected = [
                tool_step(
                    "migration_advisor",
                    required={
                        "use_image": False,
                        "use_visual_probe": False,
                        "finish_after_tool": True,
                    },
                    contains={"user_query": [project, target]},
                )
            ]
        elif scenario == "pipeline_full_eval":
            expected = [
                tool_step(
                    "pipeline_eval",
                    required={"finish_after_tool": True},
                    contains={"task_text": [target, "评测"]},
                )
            ]
        elif scenario == "flux_generation_only":
            expected = [
                tool_step(
                    "flux-image-generation",
                    required={
                        "source_image_required": True,
                        "num_images": 1,
                        "finish_after_tool": True,
                    },
                    contains={"task_text": [project, target]},
                )
            ]
        else:
            expected = [clarify_step()]

    allowed = allowed_actions(expected)
    return {
        "case_id": (
            f"PMGHV1-{split.upper()}-{entity_index + 1:03d}-"
            f"{SCENARIO_CODES[scenario]}"
        ),
        "dataset_id": DATASET_ID,
        "split": split,
        "selection_role": (
            "visible_family_screening"
            if split == "calibration"
            else "frozen_family_confirmation_unfiltered"
        ),
        "entity_id": entity_id,
        "group_id": entity_id,
        "template_id": template_id,
        "scenario_id": scenario,
        "scenario_tier": "primary_challenge" if scenario in PRIMARY_SCENARIOS else "guardrail",
        "category": scenario,
        "bad_case_targets": BAD_CASE_TARGETS[scenario],
        "user_query": query,
        "image_fixture_family": fixture_family if image_fixture else "none",
        "setup": setup(fixture=image_fixture),
        "expected_decisions": expected,
        "mock_observations": observations,
        "forbidden_actions": [tool for tool in ACTIVE_TOOLS if tool not in allowed],
        "reward_spec": STRICT_REWARD,
        "provenance_class": "synthetic_from_aggregate_compound245_failure_taxonomy",
    }


def build_split(
    *,
    split: str,
    accepted_scenarios: list[str],
    confirmation_cases: int,
) -> list[dict[str, Any]]:
    if split == "calibration":
        scenarios = list(SCENARIOS)
        entities = ENTITY_SPLITS[split][:CALIBRATION_ENTITY_COUNT]
        return [
            make_case(
                split=split,
                entity_index=entity_index,
                project=project,
                scenario=scenario,
            )
            for entity_index, project in enumerate(entities)
            for scenario in scenarios
        ]

    scenarios = accepted_scenarios
    if not scenarios:
        raise ValueError(
            "confirmation requires --accepted-scenarios from the frozen calibration family gate"
        )
    needed_entities = math.ceil(confirmation_cases / len(scenarios))
    entities = ENTITY_SPLITS[split][:needed_entities]
    if len(entities) < needed_entities:
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
    return rows[:confirmation_cases]


def existing_case_overlap(rows: list[dict[str, Any]], current_paths: set[Path]) -> dict[str, Any]:
    case_ids = {str(row.get("case_id") or "") for row in rows}
    queries = {normalize_query(str(row.get("user_query") or "")) for row in rows}
    overlaps: list[dict[str, Any]] = []
    for path in sorted(CASE_DIR.glob("*_cases.jsonl")):
        if path.resolve() in current_paths:
            continue
        other_ids: set[str] = set()
        other_queries: set[str] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                continue
            other_ids.add(str(value.get("case_id") or ""))
            other_queries.add(normalize_query(str(value.get("user_query") or "")))
        id_count = len(case_ids & other_ids)
        query_count = len(queries & other_queries)
        if id_count or query_count:
            overlaps.append(
                {
                    "file": str(path.relative_to(ROOT)),
                    "case_id_overlap": id_count,
                    "normalized_query_overlap": query_count,
                }
            )
    return {
        "status": "pass" if not overlaps else "fail",
        "overlaps": overlaps,
        "files_scanned": len(list(CASE_DIR.glob("*_cases.jsonl"))) - len(current_paths),
    }


def validate_rows(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    ids = [str(row.get("case_id") or "") for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate case_id")
    for row in rows:
        cid = str(row.get("case_id") or "")
        expected = row.get("expected_decisions")
        if not isinstance(expected, list) or not expected or len(expected) > 2:
            errors.append(f"{cid}: expected_decisions must have 1-2 steps")
            continue
        setup_data = row.get("setup") if isinstance(row.get("setup"), dict) else {}
        if bool(setup_data.get("has_image")):
            fixture = ROOT / str(setup_data.get("image_fixture") or "")
            if not fixture.is_file():
                errors.append(f"{cid}: missing image fixture {fixture}")
        observations = row.get("mock_observations")
        if len(expected) == 2 and (
            not isinstance(observations, list)
            or len(observations) != 1
            or int(observations[0].get("after_step") or 0) != 1
        ):
            errors.append(f"{cid}: two-step case must have one after_step=1 observation")
        for step_index, step in enumerate(expected, start=1):
            if not isinstance(step, dict):
                errors.append(f"{cid}: step {step_index} is not an object")
                continue
            if str(step.get("decision_type") or "tool") == "tool":
                required = step.get("required_args") if isinstance(step.get("required_args"), dict) else {}
                if "finish_after_tool" not in required:
                    errors.append(f"{cid}: step {step_index} lacks finish_after_tool gold")
                elif not isinstance(required.get("finish_after_tool"), bool):
                    errors.append(f"{cid}: step {step_index} finish_after_tool is not bool")
    return errors


def split_integrity(generated: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    if set(generated) != {"calibration", "confirmation"}:
        return {"status": "not_applicable", "reason": "both splits were not generated together"}
    calibration = generated["calibration"]
    confirmation = generated["confirmation"]
    checks: dict[str, int] = {}
    for key in ("case_id", "entity_id", "template_id"):
        checks[f"{key}_overlap"] = len(
            {str(row.get(key) or "") for row in calibration}
            & {str(row.get(key) or "") for row in confirmation}
        )
    checks["normalized_query_overlap"] = len(
        {normalize_query(str(row.get("user_query") or "")) for row in calibration}
        & {normalize_query(str(row.get("user_query") or "")) for row in confirmation}
    )
    checks["fixture_family_overlap"] = len(
        {str(row.get("image_fixture_family") or "") for row in calibration if row.get("image_fixture_family") != "none"}
        & {str(row.get("image_fixture_family") or "") for row in confirmation if row.get("image_fixture_family") != "none"}
    )
    return {
        "status": "pass" if all(value == 0 for value in checks.values()) else "fail",
        **checks,
    }


def parse_scenarios(raw: str) -> list[str]:
    values = [value.strip() for value in re.split(r"[,\s]+", raw) if value.strip()]
    unknown = sorted(set(values) - set(SCENARIOS))
    if unknown:
        raise ValueError(f"unknown scenarios: {unknown}")
    return list(dict.fromkeys(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        default="calibration",
        help="Comma-separated: calibration and/or confirmation",
    )
    parser.add_argument(
        "--accepted-scenarios",
        default="",
        help="Frozen family allowlist required for confirmation",
    )
    parser.add_argument(
        "--confirmation-cases",
        type=int,
        default=DEFAULT_CONFIRMATION_CASES,
    )
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
    if args.confirmation_cases < 1:
        raise ValueError("--confirmation-cases must be positive")

    generated: dict[str, list[dict[str, Any]]] = {}
    output_paths: dict[str, Path] = {}
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
        write_jsonl(case_path, rows)
        steps = build_rows(
            rows,
            indent=-1,
            prompt_format="qwen_chatml",
            append_im_end=True,
        )
        step_path = args.step_dir / f"{split}.jsonl"
        write_jsonl(step_path, steps)
        generated[split] = rows
        output_paths[split] = case_path
        step_paths[split] = step_path
        split_reports[split] = {
            "cases": case_stats(rows),
            "steps": step_stats(steps),
            "entity_groups": len({str(row.get("entity_id") or "") for row in rows}),
            "scenario_tiers": dict(
                sorted(Counter(str(row.get("scenario_tier") or "") for row in rows).items())
            ),
            "fixture_families": dict(
                sorted(Counter(str(row.get("image_fixture_family") or "") for row in rows).items())
            ),
            "case_file": str(case_path.relative_to(ROOT)),
            "step_file": str(step_path.relative_to(ROOT)),
        }

    current_paths = {path.resolve() for path in output_paths.values()}
    all_rows = [row for split in requested for row in generated[split]]
    leakage = existing_case_overlap(all_rows, current_paths)
    if leakage["status"] != "pass":
        raise ValueError(f"existing dataset overlap detected: {leakage['overlaps']}")

    report = {
        "schema_version": "1.0",
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "role": "visible_family_calibration_and_frozen_confirmation_challenge",
        "selection_rule": (
            "Admit/reject whole scenario families using calibration only; never filter "
            "individual confirmation cases."
        ),
        "accepted_scenarios": accepted,
        "splits": split_reports,
        "integrity": {
            "existing_dataset_overlap": leakage,
            "generated_split_isolation": split_integrity(generated),
        },
        "files": {
            **{f"{split}_cases": str(path.relative_to(ROOT)) for split, path in output_paths.items()},
            **{f"{split}_steps": str(path.relative_to(ROOT)) for split, path in step_paths.items()},
        },
    }
    report["sha256"] = {
        name: sha256(ROOT / relative_path) for name, relative_path in report["files"].items()
    }
    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.dataset_dir / "build_report.json", report)
    write_json(args.step_dir / "metadata.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
