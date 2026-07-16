#!/usr/bin/env python3
"""Build the preregistered residual-focused Planner V7 GRPO dataset.

V7 keeps entity as the independent unit.  Each entity is crossed with both
detectors and nine matched state scenarios.  Query wording, badge, fixture,
and error aliases are assigned by an explicit seeded blocked layout rather
than by reusing one entity-index modulus for every nuisance factor.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
for import_root in (ROOT / "src", ROOT, ROOT / "demo"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from training.planner_grpo_seed_v1.scripts import (  # noqa: E402
    build_planner_multistep_grpo_hard_v1 as v1,
)
from training.planner_grpo_seed_v1.scripts import (  # noqa: E402
    build_planner_retry_migrate_v6 as v6,
)


DATASET_ID = "planner_retry_migrate_residual_v7"
SCHEMA_VERSION = "1.0"
SEED = 2026071607
CREATED_AT = "2026-07-16T15:30:00+00:00"
STUDY_ID = "planner_retry_migrate_residual_v7_qwen35_4b_v1"
CASE_DIR = ROOT / "training/planner_grpo_seed_v1/cases"
DATASET_DIR = ROOT / "data/datasets" / DATASET_ID
FIXTURE_DIR = ROOT / "examples/images" / DATASET_ID
STEP_DIR = ROOT / "training/planner_grpo_seed_v1/step_data"
STUDY_DIR = ROOT / "experiments/studies" / STUDY_ID
DEFAULT_MODEL = Path("/raid/zkq/models/Qwen3.5-4B")
EXPECTED_EOS_ID = 248046
EXPECTED_PAD_ID = 248044
MAX_PROMPT_TOKENS = 4608

PRIMARY_SCENARIOS = (
    "fresh_retry_step2",
    "post_retry_success_step3",
    "post_retry_error_step3",
    "post_retry_metric_veto_step3",
    "current_success_step2",
    "conflicting_state_step2",
)
CONTROL_SCENARIOS = (
    "nonretryable_step2",
    "budget_exhausted_step2",
    "missing_required_state_step2",
)
ALL_SCENARIOS = PRIMARY_SCENARIOS + CONTROL_SCENARIOS
BADGES = ("red", "amber", "missing")
DETECTORS = v6.DETECTORS

SPLIT_SPECS: dict[str, dict[str, Any]] = {
    "grpo_train": {
        "entities": 24,
        "code": "GT",
        "training_only": True,
        "evaluation_only": False,
        "exclude_from_training": False,
        "sealed": False,
        "role": "optimization_only",
    },
    "grpo_dev": {
        "entities": 12,
        "code": "GD",
        "training_only": False,
        "evaluation_only": True,
        "exclude_from_training": True,
        "sealed": False,
        "role": "support_gate_and_model_selection_only",
    },
    "test": {
        "entities": 12,
        "code": "TE",
        "training_only": False,
        "evaluation_only": True,
        "exclude_from_training": True,
        "sealed": True,
        "role": "sealed_final_evaluation_only",
    },
}

LEXICONS: dict[str, dict[str, tuple[str, ...]]] = {
    "grpo_train": {
        "roots": (
            "海底电缆登陆站",
            "氢能压缩加注岛",
            "半导体湿法清洗间",
            "高原气象雷达站",
            "港区自动系泊台",
            "工业蒸汽调压廊",
        ),
        "suffixes": (
            "失联恢复复核",
            "迁移前置裁决",
            "剩余预算审查",
            "故障回执终判",
        ),
        "styles": (
            "请对最新故障回执完成最终处置判定",
            "核验失联探针与剩余预算后裁决后续路径",
            "这是一张迁移前置审查单，请按当前状态推进",
            "告警文字提到预算耗尽，但本轮只按结构化字段决定",
        ),
    },
    "grpo_dev": {
        "roots": (
            "潮汐能变流器舱",
            "隧道应急广播间",
            "航空发动机试车台",
            "冷链机器人充电区",
        ),
        "suffixes": (
            "异常链路复核",
            "探针预算终审",
            "恢复动作裁决",
        ),
        "styles": (
            "读取当前故障状态并完成处置终审",
            "请在失联告警背景下核销本轮重试预算",
            "迁移评审已排队，先依据最后一条结构化回执选择动作",
            "标题写着预算用尽不代表字段已耗尽，请按显式状态裁决",
        ),
    },
    "test": {
        "roots": (
            "月面样品封装舱",
            "深海浮标维护台",
            "磁悬浮道岔机房",
            "核医学物流缓冲区",
        ),
        "suffixes": (
            "回执边界终验",
            "链路处置会签",
            "能力恢复定级",
        ),
        "styles": (
            "请完成这张异常处置单的最终路由",
            "失联与故障只是工单背景，核对最新字段后继续",
            "在迁移候选已登记的前提下判断当前动作",
            "预算耗尽出现在备注中，请以结构化重试计数为准",
        ),
    },
}

POLICY_WORDING: dict[str, tuple[str, ...]] = {
    "grpo_train": (
        "工单名称中的故障、失联、迁移、耗尽均是背景词，不能替代当前 query 最后一条 observation 的显式字段。",
        "gateway_error 非 none 时，只有 retryable=true 且 retry_count=0 才允许同一检测器再执行一次；其余情况进入迁移顾问。",
        "gateway_error=none 时，candidate_count>=1、min_confidence>=0.88、cross_prompt_iou>=0.72、domain_shift=low 四项全过才结束。",
        "字段缺失或互相冲突按不安全状态处理；重试后的 observation 只能结束或迁移，不允许第二次重试。",
    ),
    "grpo_dev": (
        "自然语言里的故障等级和预算描述没有路由优先级，只读取本 query 最新 observation 的结构化键值。",
        "错误状态仅在 retryable=true 与 retry_count=0 同时成立时复用原检测器一次，否则调用迁移顾问。",
        "正常状态需要候选数、置信度、跨提示一致性和域偏移四个门槛全部通过才结束，任一否决项都转迁移。",
        "缺省、非法、冲突状态采取保守分支；已经重试一次的结果不能再次触发检测器。",
    ),
    "test": (
        "工单标题与备注只提供语境，裁决必须完全服从当前 query 最后一个 observation 的显式字段。",
        "gateway_error 存在时，原检测器的唯一重试条件是 retryable=true 且 retry_count=0；其他错误进入迁移顾问。",
        "gateway_error=none 时必须同时满足 candidate_count、min_confidence、cross_prompt_iou 与 domain_shift 门槛才结束。",
        "字段缺失或逻辑冲突均视为安全失败；重试后不允许形成循环，只能结束或迁移。",
    ),
}

ERROR_ALIASES: dict[str, dict[str, tuple[str, str]]] = {
    "grpo_train": {
        "timeout": ("sensor_reply_epoch_overrun", "vision_probe_watch_expired"),
        "transport": ("feature_uplink_session_lost", "detector_control_plane_unreachable"),
        "quota": ("inspection_retry_credit_empty", "vision_execution_budget_locked"),
        "payload": ("probe_result_envelope_invalid", "detector_request_schema_diverged"),
    },
    "grpo_dev": {
        "timeout": ("visual_response_clock_elapsed", "probe_ack_deadline_closed"),
        "transport": ("detector_return_channel_missing", "vision_service_route_unbound"),
        "quota": ("probe_retry_allowance_spent", "detector_capacity_token_absent"),
        "payload": ("vision_state_packet_malformed", "detector_input_contract_unknown"),
    },
    "test": {
        "timeout": ("inspection_result_horizon_passed", "visual_worker_reply_window_shut"),
        "transport": ("feature_exchange_path_unavailable", "detector_session_carrier_broken"),
        "quota": ("vision_retry_lease_depleted", "probe_admission_credit_withheld"),
        "payload": ("detector_state_manifest_corrupt", "visual_request_shape_unrecognized"),
    },
}

FIXTURES: dict[str, tuple[dict[str, Any], ...]] = {
    "grpo_train": (
        {"target": "青绿双环联轴器", "slug": "teal_double_ring_coupler", "family": "prrv7_gt_teal_rings", "shape": "rings", "fg": (45, 170, 155), "bg": (43, 49, 56)},
        {"target": "金色五星检修片", "slug": "gold_five_star_service_tab", "family": "prrv7_gt_gold_star", "shape": "star", "fg": (221, 171, 42), "bg": (224, 229, 232)},
    ),
    "grpo_dev": (
        {"target": "绯红十字定位块", "slug": "crimson_cross_locator", "family": "prrv7_gd_crimson_cross", "shape": "cross", "fg": (188, 55, 67), "bg": (224, 219, 210)},
        {"target": "碧蓝三辐调节轮", "slug": "azure_three_spoke_adjuster", "family": "prrv7_gd_azure_wheel", "shape": "wheel", "fg": (47, 137, 201), "bg": (42, 48, 58)},
    ),
    "test": (
        {"target": "白铜弧顶隔离座", "slug": "nickel_arch_isolator", "family": "prrv7_te_nickel_arch", "shape": "arch", "fg": (184, 181, 164), "bg": (45, 51, 61)},
        {"target": "朱红六角巡检片", "slug": "vermilion_hex_inspection_tab", "family": "prrv7_te_vermilion_hex", "shape": "hexagon", "fg": (211, 75, 50), "bg": (224, 226, 218)},
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(jsonl_text(rows), encoding="utf-8")


def fixture_path(spec: dict[str, Any]) -> Path:
    return FIXTURE_DIR / f"{spec['slug']}.png"


def fixture_relative_path(spec: dict[str, Any]) -> str:
    return str(fixture_path(spec).relative_to(ROOT))


def write_fixture_images() -> list[Path]:
    paths: list[Path] = []
    for index, spec in enumerate(spec for split in SPLIT_SPECS for spec in FIXTURES[split]):
        path = fixture_path(spec)
        v6.draw_fixture(path, spec, index=index + 20)
        paths.append(path)
    hashes = [sha256_file(path) for path in paths]
    if len(hashes) != len(set(hashes)):
        raise ValueError("V7 fixture images are not content-unique")
    historical = {
        sha256_file(path)
        for path in (ROOT / "examples/images").rglob("*.png")
        if path not in set(paths)
    }
    if historical.intersection(hashes):
        raise ValueError("V7 fixture image content overlaps an existing fixture")
    return paths


def build_projects(split: str) -> list[str]:
    lexicon = LEXICONS[split]
    pairs = list(itertools.product(lexicon["roots"], lexicon["suffixes"]))
    expected = int(SPLIT_SPECS[split]["entities"])
    if len(pairs) != expected:
        raise ValueError(f"{split}: expected {expected} root/suffix pairs, found {len(pairs)}")
    random.Random(SEED + 101 + sum(map(ord, split))).shuffle(pairs)
    return [
        f"{root}{suffix}{1000 + int(sha256_text(f'{split}|{index}|{root}|{suffix}')[:6], 16) % 9000:04d}号"
        for index, (root, suffix) in enumerate(pairs)
    ]


PROJECTS = {split: build_projects(split) for split in SPLIT_SPECS}


def factor_layout(split: str) -> list[dict[str, Any]]:
    styles = range(len(LEXICONS[split]["styles"]))
    badges = range(len(BADGES))
    if split == "grpo_train":
        combinations = list(itertools.product(styles, badges, range(len(FIXTURES[split]))))
    else:
        combinations = [
            (style_index, badge_index, (style_index + badge_index + int(split == "test")) % 2)
            for style_index, badge_index in itertools.product(styles, badges)
        ]
    if len(combinations) != int(SPLIT_SPECS[split]["entities"]):
        raise ValueError(f"{split}: factor layout size mismatch")
    random.Random(SEED + 211 + sum(map(ord, split))).shuffle(combinations)
    return [
        {
            "style_index": style_index,
            "badge_index": badge_index,
            "fixture_index": fixture_index,
        }
        for style_index, badge_index, fixture_index in combinations
    ]


FACTOR_LAYOUTS = {split: factor_layout(split) for split in SPLIT_SPECS}


def alias_layout(split: str) -> list[tuple[str, str]]:
    aliases = [
        (family, alias)
        for family, family_aliases in ERROR_ALIASES[split].items()
        for alias in family_aliases
    ]
    bundles = int(SPLIT_SPECS[split]["entities"]) * len(DETECTORS)
    if bundles % len(aliases):
        raise ValueError(f"{split}: {bundles} detector bundles do not balance {len(aliases)} aliases")
    layout = aliases * (bundles // len(aliases))
    random.Random(SEED + 307 + sum(map(ord, split))).shuffle(layout)
    return layout


ALIAS_LAYOUTS = {split: alias_layout(split) for split in SPLIT_SPECS}


def entity_id(split: str, entity_index: int) -> str:
    return f"prrv7_{SPLIT_SPECS[split]['code'].lower()}_entity_{entity_index + 1:03d}"


def policy_query(split: str, style: str, project: str, target: str, model_text: str) -> str:
    return (
        f"{style}：先使用{model_text}检测附件中的{target}，检测结果仅作为中间状态。"
        f"最终需要判断{project}应继续现有能力、结束，还是进入迁移评估。"
        + "".join(POLICY_WORDING[split])
        + "检测器必须设置 finish_after_tool=false；迁移顾问使用当前图片和视觉探针并设置 finish_after_tool=true。"
    )


def base_case(
    *,
    split: str,
    entity_index: int,
    detector_index: int,
    scenario_id: str,
    suffix: str,
    query: str,
    expected: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    target_step: int,
    target_action_class: str,
    error_family: str,
    error_alias: str,
    bundle_error_alias: str,
    badge: str,
    fixture: dict[str, Any],
    style_index: int,
    bundle_id: str,
    post_retry_outcome: str = "",
) -> dict[str, Any]:
    detector, _, _ = DETECTORS[detector_index]
    spec = SPLIT_SPECS[split]
    case_id = f"PRRV7-{spec['code']}-{entity_index + 1:03d}-{detector.upper()}-{suffix}"
    allowed = v1.allowed_actions(expected)
    return {
        "case_id": case_id,
        "dataset_id": DATASET_ID,
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "data_stage": "grpo",
        "selection_role": spec["role"],
        "training_only": spec["training_only"],
        "evaluation_only": spec["evaluation_only"],
        "exclude_from_training": spec["exclude_from_training"],
        "sealed": spec["sealed"],
        "entity_id": entity_id(split, entity_index),
        "group_id": bundle_id,
        "counterfactual_bundle_id": bundle_id,
        "project_entity": PROJECTS[split][entity_index],
        "target_entity": str(fixture["target"]),
        "template_id": f"prrv7_{split}_{detector}_hard_{style_index + 1}",
        "scenario_id": scenario_id,
        "category": scenario_id,
        "optimization_scope": "primary_residual" if scenario_id in PRIMARY_SCENARIOS else "stability_control",
        "detector_family": detector,
        "target_action_class": target_action_class,
        "decision_under_test_step": target_step,
        "grpo_target_step": target_step,
        "grpo_eligible": split != "test",
        "sft_eligible": False,
        "guardrail": scenario_id not in {
            "fresh_retry_step2",
            "post_retry_success_step3",
            "post_retry_error_step3",
            "post_retry_metric_veto_step3",
            "nonretryable_step2",
            "budget_exhausted_step2",
        },
        "error_family": error_family,
        "error_alias": error_alias,
        "bundle_error_alias": bundle_error_alias,
        "badge_condition": badge,
        "post_retry_outcome": post_retry_outcome,
        "query_style_index": style_index,
        "user_query": query,
        "image_fixture_family": str(fixture["family"]),
        "setup": {
            **v1.setup(fixture=fixture_relative_path(fixture)),
            "max_steps": 3,
            "query_trajectories": [],
        },
        "expected_decisions": expected,
        "mock_observations": observations,
        "forbidden_actions": [tool for tool in v1.ACTIVE_TOOLS if tool not in allowed],
        "reward_spec": dict(v6.VALUE_REWARD),
        "provenance_class": "independent_synthetic_residual_factorial_v7",
    }


def make_bundle(split: str, entity_index: int, detector_index: int) -> list[dict[str, Any]]:
    detector, action, model_text = DETECTORS[detector_index]
    factors = FACTOR_LAYOUTS[split][entity_index]
    style_index = int(factors["style_index"])
    badge = BADGES[int(factors["badge_index"])]
    fixture = FIXTURES[split][int(factors["fixture_index"])]
    project = PROJECTS[split][entity_index]
    family, alias = ALIAS_LAYOUTS[split][entity_index * len(DETECTORS) + detector_index]
    query = policy_query(
        split,
        LEXICONS[split]["styles"][style_index],
        project,
        str(fixture["target"]),
        model_text,
    )
    bundle_id = f"{entity_id(split, entity_index)}_{detector}_{alias}"
    first = v6.detection_step(action, str(fixture["target"]))
    retry = v6.detection_step(action, str(fixture["target"]))
    migrate = v6.migration_step(project)
    end = v1.end_step()
    fresh_error = v6.error_observation(
        1, alias=alias, retryable=True, retry_count=0, badge_mode=badge
    )
    post_success = v6.success_observation(2, badge_mode=badge, retry_count=1)
    post_error = v6.error_observation(
        2, alias=alias, retryable=True, retry_count=1, badge_mode=badge
    )
    post_veto = v6.veto_observation(
        2,
        badge_mode=badge,
        retry_count=1,
        veto_index=entity_index * len(DETECTORS) + detector_index,
    )

    common = {
        "split": split,
        "entity_index": entity_index,
        "detector_index": detector_index,
        "query": query,
        "bundle_error_alias": alias,
        "badge": badge,
        "fixture": fixture,
        "style_index": style_index,
        "bundle_id": bundle_id,
    }
    return [
        base_case(
            **common,
            scenario_id="fresh_retry_step2",
            suffix="FR2",
            expected=[first, retry, end],
            observations=[fresh_error, post_success],
            target_step=2,
            target_action_class="retry",
            error_family=family,
            error_alias=alias,
            post_retry_outcome="success",
        ),
        base_case(
            **common,
            scenario_id="post_retry_success_step3",
            suffix="PS3",
            expected=[first, retry, end],
            observations=[fresh_error, post_success],
            target_step=3,
            target_action_class="end",
            error_family=family,
            error_alias=alias,
            post_retry_outcome="success",
        ),
        base_case(
            **common,
            scenario_id="post_retry_error_step3",
            suffix="PE3",
            expected=[first, retry, migrate],
            observations=[fresh_error, post_error],
            target_step=3,
            target_action_class="migrate",
            error_family=family,
            error_alias=alias,
            post_retry_outcome="error",
        ),
        base_case(
            **common,
            scenario_id="post_retry_metric_veto_step3",
            suffix="PV3",
            expected=[first, retry, migrate],
            observations=[fresh_error, post_veto],
            target_step=3,
            target_action_class="migrate",
            error_family="none",
            error_alias="none",
            post_retry_outcome="metric_veto",
        ),
        base_case(
            **common,
            scenario_id="current_success_step2",
            suffix="CS2",
            expected=[first, end],
            observations=[v6.success_observation(1, badge_mode=badge, retry_count=0)],
            target_step=2,
            target_action_class="end",
            error_family="none",
            error_alias="none",
        ),
        base_case(
            **common,
            scenario_id="conflicting_state_step2",
            suffix="CF2",
            expected=[first, migrate],
            observations=[
                v6.structured_observation(
                    1,
                    success=True,
                    status="completed",
                    candidate_count=None,
                    min_confidence=None,
                    cross_prompt_iou=None,
                    domain_shift="unknown",
                    gateway_error="none",
                    retryable=True,
                    retry_count=0,
                    badge_mode=badge,
                )
            ],
            target_step=2,
            target_action_class="migrate",
            error_family="none",
            error_alias="none",
        ),
        base_case(
            **common,
            scenario_id="nonretryable_step2",
            suffix="NR2",
            expected=[first, migrate],
            observations=[
                v6.error_observation(
                    1, alias=alias, retryable=False, retry_count=0, badge_mode=badge
                )
            ],
            target_step=2,
            target_action_class="migrate",
            error_family=family,
            error_alias=alias,
        ),
        base_case(
            **common,
            scenario_id="budget_exhausted_step2",
            suffix="BE2",
            expected=[first, migrate],
            observations=[
                v6.error_observation(
                    1,
                    alias=alias,
                    retryable=True,
                    retry_count=1 + ((entity_index + detector_index) % 2),
                    badge_mode=badge,
                )
            ],
            target_step=2,
            target_action_class="migrate",
            error_family=family,
            error_alias=alias,
        ),
        base_case(
            **common,
            scenario_id="missing_required_state_step2",
            suffix="MS2",
            expected=[first, migrate],
            observations=[
                v6.structured_observation(
                    1,
                    success=False,
                    status="gateway_error",
                    candidate_count=None,
                    min_confidence=None,
                    cross_prompt_iou=None,
                    domain_shift="unknown",
                    gateway_error=alias,
                    retryable=None,
                    retry_count=0,
                    badge_mode=badge,
                    omit_fields=("retryable",),
                )
            ],
            target_step=2,
            target_action_class="migrate",
            error_family=family,
            error_alias=alias,
        ),
    ]


def build_split_cases(split: str) -> list[dict[str, Any]]:
    rows = [
        case
        for entity_index in range(int(SPLIT_SPECS[split]["entities"]))
        for detector_index in range(len(DETECTORS))
        for case in make_bundle(split, entity_index, detector_index)
    ]
    random.Random(SEED + 401 + sum(map(ord, split))).shuffle(rows)
    return rows


def build_all_cases() -> dict[str, list[dict[str, Any]]]:
    return {split: build_split_cases(split) for split in SPLIT_SPECS}


def factor_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    entities: dict[str, dict[str, Any]] = {}
    bundles: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        entities.setdefault(
            str(row["entity_id"]),
            {
                "style": int(row["query_style_index"]),
                "badge": str(row["badge_condition"]),
                "fixture": str(row["image_fixture_family"]),
            },
        )
        bundles[str(row["counterfactual_bundle_id"])].append(row)
    return {
        "entities": len(entities),
        "entity_style_badge_fixture": dict(
            sorted(
                Counter(
                    f"style{value['style'] + 1}|{value['badge']}|{value['fixture']}"
                    for value in entities.values()
                ).items()
            )
        ),
        "styles": dict(sorted(Counter(value["style"] for value in entities.values()).items())),
        "badges": dict(sorted(Counter(value["badge"] for value in entities.values()).items())),
        "fixtures": dict(sorted(Counter(value["fixture"] for value in entities.values()).items())),
        "bundle_aliases": dict(
            sorted(Counter(str(values[0]["bundle_error_alias"]) for values in bundles.values()).items())
        ),
        "scenarios": dict(sorted(Counter(str(row["scenario_id"]) for row in rows).items())),
        "detectors": dict(sorted(Counter(str(row["detector_family"]) for row in rows).items())),
        "target_actions": dict(sorted(Counter(str(row["target_action_class"]) for row in rows).items())),
        "target_steps": dict(sorted(Counter(int(row["grpo_target_step"]) for row in rows).items())),
    }


def protected_values(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    fields = (
        "case_id",
        "entity_id",
        "project_entity",
        "target_entity",
        "image_fixture_family",
        "bundle_error_alias",
    )
    return {
        field: {str(row.get(field) or "") for row in rows if str(row.get(field) or "")}
        for field in fields
    }


def validate_cases(cases_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    errors: list[str] = []
    expected_scenarios = set(ALL_SCENARIOS)
    for split, rows in cases_by_split.items():
        entity_count = int(SPLIT_SPECS[split]["entities"])
        expected_rows = entity_count * len(DETECTORS) * len(ALL_SCENARIOS)
        if len(rows) != expected_rows:
            errors.append(f"{split}: expected {expected_rows} rows, found {len(rows)}")
        if len({str(row["case_id"]) for row in rows}) != len(rows):
            errors.append(f"{split}: duplicate case IDs")
        bundles: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            bundles[str(row["counterfactual_bundle_id"])].append(row)
        if len(bundles) != entity_count * len(DETECTORS):
            errors.append(f"{split}: detector bundle count mismatch")
        for bundle_id, values in bundles.items():
            if len(values) != len(ALL_SCENARIOS):
                errors.append(f"{split}/{bundle_id}: expected nine scenarios")
                continue
            if {str(row["scenario_id"]) for row in values} != expected_scenarios:
                errors.append(f"{split}/{bundle_id}: incomplete scenario set")
            for field in (
                "user_query",
                "badge_condition",
                "image_fixture_family",
                "query_style_index",
                "bundle_error_alias",
            ):
                if len({str(row[field]) for row in values}) != 1:
                    errors.append(f"{split}/{bundle_id}: {field} changed within block")
        scenario_counts = Counter(str(row["scenario_id"]) for row in rows)
        if set(scenario_counts.values()) != {entity_count * len(DETECTORS)}:
            errors.append(f"{split}: scenario counts are not balanced: {scenario_counts}")
        entity_rows = {
            str(row["entity_id"]): row
            for row in rows
        }
        if split == "grpo_train":
            combinations = Counter(
                (
                    int(row["query_style_index"]),
                    str(row["badge_condition"]),
                    str(row["image_fixture_family"]),
                )
                for row in entity_rows.values()
            )
            if len(combinations) != 24 or set(combinations.values()) != {1}:
                errors.append("grpo_train: style×badge×fixture is not a full factorial")
        else:
            style_badge = Counter(
                (int(row["query_style_index"]), str(row["badge_condition"]))
                for row in entity_rows.values()
            )
            fixture_counts = Counter(str(row["image_fixture_family"]) for row in entity_rows.values())
            if len(style_badge) != 12 or set(style_badge.values()) != {1}:
                errors.append(f"{split}: style×badge layout is not complete")
            if set(fixture_counts.values()) != {entity_count // 2}:
                errors.append(f"{split}: fixtures are not globally balanced")
        alias_counts = Counter(
            str(values[0]["bundle_error_alias"]) for values in bundles.values()
        )
        if len(alias_counts) != 8 or len(set(alias_counts.values())) != 1:
            errors.append(f"{split}: alias spellings are not balanced: {alias_counts}")
        for row in rows:
            if not bool(v6.score_case(row)["passed"]):
                errors.append(f"{row['case_id']}: canonical full trajectory failed")
            expected = row["expected_decisions"]
            detector_action = v6.expected_action_name(expected[0])
            for observation in row["mock_observations"]:
                after_step = int(observation["after_step"])
                oracle = v6.independent_oracle_action(str(observation["observation"]["summary"]))
                gold = v6.expected_action_class(expected[after_step], detector_action=detector_action)
                if oracle != gold:
                    errors.append(f"{row['case_id']} step {after_step}: oracle={oracle}, gold={gold}")

    all_ids = [str(row["case_id"]) for rows in cases_by_split.values() for row in rows]
    if len(all_ids) != len(set(all_ids)):
        errors.append("case IDs overlap across V7 splits")
    split_names = list(cases_by_split)
    split_overlap: dict[str, Any] = {}
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            left_values = protected_values(cases_by_split[left])
            right_values = protected_values(cases_by_split[right])
            overlap = {
                field: sorted(left_values[field] & right_values[field])
                for field in left_values
            }
            split_overlap[f"{left}__{right}"] = {
                field: len(values) for field, values in overlap.items()
            }
            if any(overlap.values()):
                errors.append(f"{left}/{right}: protected values overlap")

    historical_rows: list[dict[str, Any]] = []
    for path in CASE_DIR.glob("planner_retry_migrate_v6_*_cases.jsonl"):
        historical_rows.extend(
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    historical = protected_values(historical_rows)
    current = protected_values([row for rows in cases_by_split.values() for row in rows])
    repository_overlap = {
        field: len(current[field] & historical[field]) for field in current
    }
    if any(repository_overlap.values()):
        errors.append(f"V7 protected values overlap V6: {repository_overlap}")
    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "split_overlap": split_overlap,
        "v6_repository_overlap": repository_overlap,
        "factors": {split: factor_report(rows) for split, rows in cases_by_split.items()},
    }


def load_tokenizer(model_path: Path) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=False,
        use_fast=True,
        padding_side="left",
    )
    if tokenizer.eos_token_id != EXPECTED_EOS_ID or tokenizer.pad_token_id != EXPECTED_PAD_ID:
        raise ValueError(
            f"tokenizer contract changed: eos={tokenizer.eos_token_id}, pad={tokenizer.pad_token_id}"
        )
    return tokenizer


def build_step_rows(cases: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in sorted(cases, key=lambda row: str(row["case_id"])):
        step_index = int(case["grpo_target_step"])
        prompt, prompt_tokens = v6._render_row_prompt(case, step_index, tokenizer)
        expected = case["expected_decisions"]
        rows.append(
            {
                "prompt": prompt,
                "case_id": str(case["case_id"]),
                "dataset_id": DATASET_ID,
                "split": str(case["split"]),
                "category": str(case["category"]),
                "scenario_id": str(case["scenario_id"]),
                "optimization_scope": str(case["optimization_scope"]),
                "step_index": step_index,
                "expected_step": json.dumps(expected[step_index - 1], ensure_ascii=False, sort_keys=True),
                "forbidden_actions": json.dumps(case["forbidden_actions"], ensure_ascii=False, sort_keys=True),
                "reward_spec": json.dumps(case["reward_spec"], ensure_ascii=False, sort_keys=True),
                "previous_action": v6.expected_action_name(expected[step_index - 2]),
                "entity_id": str(case["entity_id"]),
                "group_id": str(case["group_id"]),
                "counterfactual_bundle_id": str(case["counterfactual_bundle_id"]),
                "template_id": str(case["template_id"]),
                "detector_family": str(case["detector_family"]),
                "target_action_class": str(case["target_action_class"]),
                "badge_condition": str(case["badge_condition"]),
                "image_fixture_family": str(case["image_fixture_family"]),
                "bundle_error_alias": str(case["bundle_error_alias"]),
                "full_expected_actions": json.dumps(
                    [v6.expected_action_name(item) for item in expected],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "data_stage": "grpo",
                "prompt_token_count": prompt_tokens,
                "prompt_sha256": sha256_text(prompt),
            }
        )
    return rows


def prompt_repository_overlap(rows: list[dict[str, Any]], excluded: set[Path]) -> dict[str, Any]:
    current = {str(row["prompt_sha256"]) for row in rows}
    historical: set[str] = set()
    paths: list[str] = []
    for path in STEP_DIR.glob("*.jsonl"):
        if path.resolve() in excluded:
            continue
        found = False
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            prompt_hash = str(value.get("prompt_sha256") or "")
            if prompt_hash:
                historical.add(prompt_hash)
                found = True
        if found:
            paths.append(str(path.relative_to(ROOT)))
    return {
        "checked_files": paths,
        "historical_prompt_hashes": len(historical),
        "current_prompt_hashes": len(current),
        "overlap": len(current & historical),
    }


def summarize_step_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = sorted(int(row["prompt_token_count"]) for row in rows)
    return {
        "rows": len(rows),
        "entities": len({str(row["entity_id"]) for row in rows}),
        "categories": dict(sorted(Counter(str(row["category"]) for row in rows).items())),
        "optimization_scope": dict(
            sorted(Counter(str(row["optimization_scope"]) for row in rows).items())
        ),
        "target_steps": dict(sorted(Counter(int(row["step_index"]) for row in rows).items())),
        "target_actions": dict(
            sorted(Counter(str(row["target_action_class"]) for row in rows).items())
        ),
        "prompt_tokens": {
            "min": min(lengths),
            "mean": statistics.fmean(lengths),
            "p50": lengths[len(lengths) // 2],
            "p95": lengths[int((len(lengths) - 1) * 0.95)],
            "max": max(lengths),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--case-dir", type=Path, default=CASE_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--step-dir", type=Path, default=STEP_DIR)
    parser.add_argument("--materialize-test", action="store_true")
    parser.add_argument("--confirm-materialize-test", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_name_or_path.resolve()
    case_dir = args.case_dir if args.case_dir.is_absolute() else ROOT / args.case_dir
    dataset_dir = args.dataset_dir if args.dataset_dir.is_absolute() else ROOT / args.dataset_dir
    step_dir = args.step_dir if args.step_dir.is_absolute() else ROOT / args.step_dir
    if args.materialize_test and args.confirm_materialize_test != "OPEN_V7_TEST":
        raise PermissionError("test materialization requires --confirm-materialize-test OPEN_V7_TEST")

    fixture_paths = write_fixture_images()
    cases_by_split = build_all_cases()
    validation = validate_cases(cases_by_split)
    if validation["status"] != "pass":
        raise ValueError("V7 case validation failed:\n" + "\n".join(validation["errors"][:100]))

    case_paths: dict[str, Path] = {}
    for split in ("grpo_train", "grpo_dev"):
        path = case_dir / f"{DATASET_ID}_{split}_cases.jsonl"
        write_jsonl(path, cases_by_split[split])
        case_paths[split] = path
    test_commitment = sha256_text(jsonl_text(cases_by_split["test"]))
    test_path: Path | None = None
    if args.materialize_test:
        test_path = STUDY_DIR / "sealed_test_data/test_cases.jsonl"
        write_jsonl(test_path, cases_by_split["test"])
        if sha256_file(test_path) != test_commitment:
            raise ValueError("materialized V7 test does not match preregistered commitment")

    tokenizer = load_tokenizer(model_path)
    step_paths: dict[str, Path] = {}
    step_rows: dict[str, list[dict[str, Any]]] = {}
    for split in ("grpo_train", "grpo_dev"):
        rows = build_step_rows(cases_by_split[split], tokenizer)
        audit = v6.prompt_audit(rows)
        if audit["status"] != "pass":
            raise ValueError(f"{split} prompt audit failed: {json.dumps(audit, ensure_ascii=False)}")
        path = step_dir / f"{DATASET_ID}_{split}_qwen35_4b_nothinking_mixed_steps.jsonl"
        write_jsonl(path, rows)
        step_rows[split] = rows
        step_paths[split] = path

    all_step_rows = [row for rows in step_rows.values() for row in rows]
    if len({str(row["prompt_sha256"]) for row in all_step_rows}) != len(all_step_rows):
        raise ValueError("V7 train/dev formatted prompts are not unique")
    if {str(row["prompt_sha256"]) for row in step_rows["grpo_train"]} & {
        str(row["prompt_sha256"]) for row in step_rows["grpo_dev"]
    }:
        raise ValueError("V7 train/dev formatted prompt hashes overlap")
    repository_prompt_audit = prompt_repository_overlap(
        all_step_rows,
        {path.resolve() for path in step_paths.values()},
    )
    if repository_prompt_audit["overlap"]:
        raise ValueError(f"V7 prompts overlap repository history: {repository_prompt_audit}")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    audit_path = dataset_dir / "audit_report.json"
    audit_payload = {
        "schema_version": "1.0",
        "created_at": CREATED_AT,
        "dataset_id": DATASET_ID,
        "case_validation": validation,
        "step_prompt_audits": {
            split: v6.prompt_audit(rows) for split, rows in step_rows.items()
        },
        "repository_prompt_audit": repository_prompt_audit,
        "test_commitment": {
            "rows": len(cases_by_split["test"]),
            "sha256": test_commitment,
            "materialized": bool(test_path),
        },
    }
    write_json(audit_path, audit_payload)

    step_manifests: dict[str, Path] = {}
    for split, path in step_paths.items():
        manifest_path = path.with_suffix(".manifest.json")
        manifest = {
            "schema_version": "1.0",
            "created_at": CREATED_AT,
            "dataset_id": DATASET_ID,
            "role": SPLIT_SPECS[split]["role"],
            "model_name_or_path": str(model_path),
            "rows": len(step_rows[split]),
            "allowed_step_indices": [2, 3],
            "prompt_contract": {
                "chat_template": "native_qwen35",
                "enable_thinking": False,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
                "max_prompt_tokens": MAX_PROMPT_TOKENS,
            },
            "distribution": summarize_step_rows(step_rows[split]),
            "files": {
                "cases": str(case_paths[split].relative_to(ROOT)),
                "step_data": str(path.relative_to(ROOT)),
            },
            "sha256": {
                "cases": sha256_file(case_paths[split]),
                "step_data": sha256_file(path),
                "config": sha256_file(model_path / "config.json"),
                "tokenizer_config": sha256_file(model_path / "tokenizer_config.json"),
                "chat_template": sha256_file(model_path / "chat_template.jinja"),
            },
        }
        write_json(manifest_path, manifest)
        step_manifests[split] = manifest_path

    manifest_path = dataset_dir / "manifest.json"
    manifest = {
        "schema_version": "1.0",
        "created_at": CREATED_AT,
        "dataset_id": DATASET_ID,
        "study_id": STUDY_ID,
        "status": "frozen_train_dev_test_committed",
        "seed": SEED,
        "experimental_unit": "entity × detector counterfactual bundle",
        "independent_replication_unit": "entity",
        "scenarios": {
            "primary_residual": list(PRIMARY_SCENARIOS),
            "stability_control": list(CONTROL_SCENARIOS),
        },
        "case_rows": {split: len(rows) for split, rows in cases_by_split.items()},
        "step_rows": {split: summarize_step_rows(rows) for split, rows in step_rows.items()},
        "test_commitment": {
            "rows": len(cases_by_split["test"]),
            "sha256": test_commitment,
            "materialized": bool(test_path),
        },
        "integrity": {
            "case_validation": validation["status"],
            "repository_prompt_overlap": repository_prompt_audit["overlap"],
            "independent_human_review": "pending",
        },
        "files": {
            "builder": str(Path(__file__).resolve().relative_to(ROOT)),
            "preregistration": str((STUDY_DIR / "preregistration.json").relative_to(ROOT)),
            "audit_report": str(audit_path.relative_to(ROOT)),
            **{f"cases_{split}": str(path.relative_to(ROOT)) for split, path in case_paths.items()},
            **{f"step_data_{split}": str(path.relative_to(ROOT)) for split, path in step_paths.items()},
            **{f"step_manifest_{split}": str(path.relative_to(ROOT)) for split, path in step_manifests.items()},
            **{f"fixture_{index + 1:02d}": str(path.relative_to(ROOT)) for index, path in enumerate(fixture_paths)},
        },
    }
    manifest["sha256"] = {
        name: sha256_file(ROOT / path) for name, path in manifest["files"].items()
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
