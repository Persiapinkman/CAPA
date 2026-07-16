#!/usr/bin/env python3
"""Build an entity-disjoint training analogue of the V5 retry/migrate evaluation.

The builder consumes only V5 aggregate distribution metadata when constructing
rows.  V5 case files are loaded afterwards for overlap auditing, never as row
templates or training examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[3]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from capa.evaluation.dataset_audit import case_stats, normalize_query  # noqa: E402
from training.planner_grpo_seed_v1.scripts import (  # noqa: E402
    build_planner_multistep_grpo_value_v3 as v3,
)
from training.planner_grpo_seed_v1.scripts.reward_planner_grpo import (  # noqa: E402
    score_case,
)


v1 = v3.v1
SOURCE_DATASET_ID = "planner_multistep_grpo_value_v5"
DATASET_ID = "planner_multistep_grpo_value_v5_train_v1"
CASE_DIR = ROOT / "training/planner_grpo_seed_v1/cases"
DATASET_DIR = ROOT / "data/datasets" / DATASET_ID
SOURCE_DATASET_DIR = ROOT / "data/datasets" / SOURCE_DATASET_ID
SOURCE_REPORT_PATH = SOURCE_DATASET_DIR / "build_report_confirmation.json"
SOURCE_CASE_PATHS = (
    CASE_DIR / f"{SOURCE_DATASET_ID}_calibration_cases.jsonl",
    CASE_DIR / f"{SOURCE_DATASET_ID}_confirmation_cases.jsonl",
)
CASE_PATH = CASE_DIR / f"{DATASET_ID}_train_cases.jsonl"
FIXTURE_DIR = ROOT / "examples/images" / DATASET_ID

SEED = 2026071501
TRAIN_ENTITY_COUNT = 60
SOURCE_CONFIRMATION_CASES = 240
TRAIN_CASES = 480

SUPPORTED_SCENARIOS = (
    "qwen_timeout_nonretryable",
    "rex_timeout_nonretryable",
    "qwen_timeout_budget_exhausted",
    "rex_timeout_budget_exhausted",
    "qwen_connection_nonretryable",
    "rex_connection_nonretryable",
    "qwen_connection_budget_exhausted",
    "rex_connection_budget_exhausted",
)
SCENARIO_CODES = {
    scenario: f"{'Q' if scenario.startswith('qwen_') else 'R'}{index + 1:02d}"
    for index, scenario in enumerate(SUPPORTED_SCENARIOS)
}

# Aliases are deliberately absent from both V5 calibration and confirmation.
# Nonretryable/exhausted packs share aliases, preserving V5's matched-error
# design in which retryability and retry_count, rather than the error string,
# determine the second action.
PACK_SPECS: dict[str, dict[str, Any]] = {
    "timeout_nonretryable": {
        "strategy": "nonretryable",
        "aliases": ("vision_rpc_window_expired", "detector_deadline_consumed"),
    },
    "timeout_budget_exhausted": {
        "strategy": "exhausted",
        "aliases": ("vision_rpc_window_expired", "detector_deadline_consumed"),
    },
    "connection_nonretryable": {
        "strategy": "nonretryable",
        "aliases": ("detector_peer_channel_closed", "vision_transport_link_broken"),
    },
    "connection_budget_exhausted": {
        "strategy": "exhausted",
        "aliases": ("detector_peer_channel_closed", "vision_transport_link_broken"),
    },
}

SITE_ROOTS = (
    "潮汐能控制站",
    "隧道通风机房",
    "医药冷链中转库",
    "无人机巡检基地",
    "粮食智能仓储区",
    "数据中心配电间",
    "高速服务保障区",
    "海底管线监测站",
    "机器人总装车间",
    "应急救援物资站",
    "船舶动力检修坞",
    "城市轨道信号室",
)
PROJECT_SUFFIXES = (
    "故障恢复演练",
    "异常重试审查",
    "服务韧性核验",
    "接口容错评估",
    "重试预算复核",
)
TRAIN_STYLES = (
    "请核对本次故障恢复工单",
    "依照当前服务回执继续处理",
    "完成这一轮异常分支判定",
    "按本次接口状态推进流程",
    "请执行当前重试预算复核",
    "依据最新工具回执完成路由",
)

TRAIN_FIXTURES = (
    (
        "橙色三角警示牌",
        f"examples/images/{DATASET_ID}/orange_warning_marker.png",
        "v5_train_v1_orange_warning_marker",
    ),
    (
        "蓝色圆形检修阀",
        f"examples/images/{DATASET_ID}/blue_service_valve.png",
        "v5_train_v1_blue_service_valve",
    ),
)

CURRENT_OBSERVATION_FORBIDDEN_HINTS = v3.CURRENT_OBSERVATION_FORBIDDEN_HINTS
VALUE_REWARD = v3.VALUE_REWARD
PROTECTED_OVERLAP_KEYS = (
    "case_id",
    "entity_id",
    "project_entity",
    "target_entity",
    "normalized_query",
    "template_id",
    "error_alias",
    "fixture_family",
    "fixture_path",
    "fixture_sha256",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        rows.append(value)
    return rows


def source_report() -> dict[str, Any]:
    report = load_json(SOURCE_REPORT_PATH)
    if report.get("dataset_id") != SOURCE_DATASET_ID or report.get("split") != "confirmation":
        raise ValueError("V5 confirmation build report identity mismatch")
    scenarios = tuple(str(value) for value in report.get("accepted_scenarios") or [])
    if scenarios != SUPPORTED_SCENARIOS:
        raise ValueError(
            "V5 accepted scenario taxonomy changed; review the training analogue before rebuilding"
        )
    if int((report.get("cases") or {}).get("cases") or 0) != SOURCE_CONFIRMATION_CASES:
        raise ValueError("V5 confirmation case count changed")
    return report


def build_projects() -> list[str]:
    projects = [
        f"{site}{suffix}{index + 5101:04d}号"
        for index, (site, suffix) in enumerate(
            (pair for site in SITE_ROOTS for suffix in PROJECT_SUFFIXES for pair in [(site, suffix)])
        )
    ]
    if len(projects) != TRAIN_ENTITY_COUNT:
        raise AssertionError("project lexicon must produce exactly 60 independent entities")
    random.Random(SEED).shuffle(projects)
    return projects


PROJECTS = build_projects()


def balanced_schedule(*, positive: str, negative: str, seed_offset: int) -> list[str]:
    """Return 60 values with a 60/40 mix inside every style/alias block."""

    schedule = [""] * TRAIN_ENTITY_COUNT
    for style_index in range(len(TRAIN_STYLES)):
        indices = list(range(style_index, TRAIN_ENTITY_COUNT, len(TRAIN_STYLES)))
        if len(indices) != 10:
            raise AssertionError("each style must own ten entity blocks")
        values = [positive] * 6 + [negative] * 4
        random.Random(SEED + seed_offset + style_index).shuffle(values)
        for entity_index, value in zip(indices, values, strict=True):
            schedule[entity_index] = value
    return schedule


ACTION_SCHEDULE = balanced_schedule(positive="migrate", negative="retry", seed_offset=101)
# Preserve V5's 60/40 red/amber marginal while breaking the badge/action shortcut.
BADGE_SCHEDULE = balanced_schedule(positive="amber", negative="red", seed_offset=701)


def write_fixture_images() -> list[Path]:
    """Create two deterministic routing-only fixtures not used by V5 evaluation."""

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    warning_path = ROOT / TRAIN_FIXTURES[0][1]
    warning = Image.new("RGB", (320, 240), (38, 43, 51))
    draw = ImageDraw.Draw(warning)
    draw.rectangle((18, 168, 302, 222), fill=(73, 82, 92))
    for x in range(22, 300, 36):
        draw.rectangle((x, 184, x + 18, 202), fill=(245, 183, 52))
    draw.polygon(((160, 24), (55, 166), (265, 166)), fill=(240, 126, 34), outline=(255, 225, 170), width=7)
    draw.rectangle((153, 72, 167, 126), fill=(32, 36, 42))
    draw.ellipse((151, 137, 169, 155), fill=(32, 36, 42))
    warning.save(warning_path, format="PNG", optimize=False, compress_level=9)

    valve_path = ROOT / TRAIN_FIXTURES[1][1]
    valve = Image.new("RGB", (320, 240), (218, 223, 226))
    draw = ImageDraw.Draw(valve)
    draw.rectangle((0, 170, 320, 240), fill=(112, 119, 124))
    draw.rectangle((142, 70, 178, 204), fill=(75, 85, 91))
    draw.ellipse((71, 17, 249, 195), fill=(43, 112, 181), outline=(18, 57, 94), width=12)
    draw.ellipse((105, 51, 215, 161), fill=(205, 217, 225), outline=(18, 57, 94), width=9)
    for start, end in (
        ((154, 28), (166, 183)),
        ((82, 100), (238, 112)),
        ((103, 49), (217, 163)),
        ((217, 49), (103, 163)),
    ):
        draw.line((start, end), fill=(18, 57, 94), width=9)
    draw.ellipse((145, 96, 175, 126), fill=(234, 150, 45), outline=(91, 53, 10), width=4)
    valve.save(valve_path, format="PNG", optimize=False, compress_level=9)
    return [warning_path, valve_path]


def detector_parts(scenario: str) -> tuple[str, str, str, str]:
    detector = "qwen" if scenario.startswith("qwen_") else "rex"
    pack = scenario.removeprefix(f"{detector}_")
    if pack not in PACK_SPECS:
        raise ValueError(f"unsupported scenario pack: {pack}")
    return (
        "qwen_detection" if detector == "qwen" else "rexomni_detection",
        "Qwen" if detector == "qwen" else "Rex-Omni",
        detector,
        pack,
    )


def policy_query(*, style: str, project: str, target: str, model_text: str) -> str:
    """Independent paraphrase of the public V5 policy, without copying a V5 row."""

    return (
        f"{style}：先调用{model_text}检测附件里的{target}，检测只产生中间状态。"
        f"本任务要判断{project}是否需要能力迁移。"
        "只看当前 query 最新 observation 的结构化字段，旧记录、overall_badge 与备注均不能覆盖门槛。"
        "当 gateway_error 不是 none 时，retryable=true 且 retry_count=0 是唯一允许原检测器再试一次的条件；"
        "若错误不可重试，或 retry_count 已大于零，则调用迁移顾问。若 gateway_error=none，"
        "candidate_count 至少为1、min_confidence 至少为0.88、cross_prompt_iou 至少为0.72 且"
        "domain_shift=low 时才结束，否则进入迁移顾问。原检测器调用都设置 finish_after_tool=false；"
        "迁移顾问需读取当前图片、启用内部视觉探针，并以 finish_after_tool=true 完成本轮。"
    )


def observation_for(*, pack: str, mode: str, entity_index: int) -> tuple[dict[str, Any], str]:
    spec = PACK_SPECS[pack]
    alias = str(spec["aliases"][entity_index % len(spec["aliases"])])
    if mode == "retry":
        retryable = True
        retry_count = 0
    elif mode == "migrate" and spec["strategy"] == "nonretryable":
        retryable = False
        retry_count = 0
    elif mode == "migrate" and spec["strategy"] == "exhausted":
        retryable = True
        retry_count = 1 + entity_index % 2
    else:
        raise ValueError(f"unsupported mode/strategy: {mode}/{spec['strategy']}")
    summary = (
        "当前工具状态：candidate_count=NA；min_confidence=NA；cross_prompt_iou=NA；"
        f"domain_shift=unknown；gateway_error={alias}；"
        f"retryable={'true' if retryable else 'false'}；retry_count={retry_count}；"
        f"overall_badge={BADGE_SCHEDULE[entity_index]}。附注：badge 只用于监控面板。"
    )
    return (
        v1.observation(
            1,
            success=False,
            status="gateway_error",
            summary=summary,
        ),
        alias,
    )


def make_case(*, entity_index: int, project: str, scenario: str) -> dict[str, Any]:
    target, fixture, fixture_family = TRAIN_FIXTURES[entity_index % len(TRAIN_FIXTURES)]
    action, model_text, detector, pack = detector_parts(scenario)
    mode = ACTION_SCHEDULE[entity_index]
    observation, error_alias = observation_for(pack=pack, mode=mode, entity_index=entity_index)
    expected = [
        v3.detection_step(action, target),
        v3.migration_step(project) if mode == "migrate" else v3.detection_step(action, target),
    ]
    entity_id = f"pmgv5trainv1_entity_{entity_index + 1:03d}"
    allowed = v1.allowed_actions(expected)
    style_index = entity_index % len(TRAIN_STYLES)
    return {
        "case_id": f"PMGV5TRAINV1-{entity_index + 1:03d}-{SCENARIO_CODES[scenario]}",
        "dataset_id": DATASET_ID,
        "source_evaluation_dataset_id": SOURCE_DATASET_ID,
        "source_distribution_role": "aggregate_confirmation_taxonomy_only",
        "split": "train",
        "selection_role": "optimization_only_training_pool",
        "training_only": True,
        "evaluation_only": False,
        "exclude_from_training": False,
        "entity_id": entity_id,
        "group_id": entity_id,
        "project_entity": project,
        "target_entity": target,
        "counterfactual_bundle_id": f"{entity_id}_{detector}",
        "template_id": f"v5_train_v1_policy_{detector}_{style_index + 1}",
        "scenario_id": scenario,
        "category": scenario,
        "detector_family": detector,
        "state_contract_pack": pack,
        "target_action_class": mode,
        "error_alias": error_alias,
        "grpo_target_step": 2,
        "user_query": policy_query(
            style=TRAIN_STYLES[style_index],
            project=project,
            target=target,
            model_text=model_text,
        ),
        "image_fixture_family": fixture_family,
        "setup": {**v1.setup(fixture=fixture), "query_trajectories": []},
        "expected_decisions": expected,
        "mock_observations": [observation],
        "forbidden_actions": [tool for tool in v1.ACTIVE_TOOLS if tool not in allowed],
        "reward_spec": VALUE_REWARD,
        "provenance_class": "independent_synthetic_training_analogue_from_v5_aggregate_taxonomy",
    }


def build_cases() -> list[dict[str, Any]]:
    report = source_report()
    scenarios = [str(value) for value in report["accepted_scenarios"]]
    rows = [
        make_case(entity_index=entity_index, project=project, scenario=scenario)
        for entity_index, project in enumerate(PROJECTS)
        for scenario in scenarios
    ]
    if len(rows) != TRAIN_CASES:
        raise AssertionError(f"expected {TRAIN_CASES} training rows, got {len(rows)}")
    return rows


def _counter(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def distribution_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source = source_report()
    source_categories = {
        str(key): int(value) for key, value in (source.get("families") or {}).items()
    }
    source_actions = {
        str(key): int(value) for key, value in (source.get("target_action_classes") or {}).items()
    }
    train_categories = Counter(str(row["scenario_id"]) for row in rows)
    train_actions = Counter(str(row["target_action_class"]) for row in rows)

    def proportions(values: dict[str, int] | Counter[str]) -> dict[str, float]:
        total = sum(values.values())
        return {key: value / total for key, value in sorted(values.items())}

    source_category_p = proportions(source_categories)
    train_category_p = proportions(train_categories)
    source_action_p = proportions(source_actions)
    train_action_p = proportions(train_actions)
    category_delta = {
        key: train_category_p.get(key, 0.0) - source_category_p.get(key, 0.0)
        for key in sorted(set(source_category_p) | set(train_category_p))
    }
    action_delta = {
        key: train_action_p.get(key, 0.0) - source_action_p.get(key, 0.0)
        for key in sorted(set(source_action_p) | set(train_action_p))
    }
    return {
        "status": "pass"
        if max((abs(value) for value in (*category_delta.values(), *action_delta.values())), default=0.0)
        < 1e-12
        else "fail",
        "source_split": "confirmation",
        "case_scale_factor": len(rows) / int((source.get("cases") or {}).get("cases") or 1),
        "source": {
            "cases": int((source.get("cases") or {}).get("cases") or 0),
            "families": source_categories,
            "target_action_classes": source_actions,
        },
        "train": {
            "cases": len(rows),
            "families": dict(sorted(train_categories.items())),
            "target_action_classes": dict(sorted(train_actions.items())),
            "detector_families": _counter(row["detector_family"] for row in rows),
            "fixture_families": _counter(row["image_fixture_family"] for row in rows),
            "error_aliases": _counter(row["error_alias"] for row in rows),
            "overall_badges": _counter(
                re.search(
                    r"overall_badge=([^；。\s]+)",
                    str(row["mock_observations"][0]["observation"]["summary"]),
                ).group(1)
                for row in rows
            ),
        },
        "probability_delta": {
            "families": category_delta,
            "target_action_classes": action_delta,
        },
        "total_variation_distance": {
            "families": 0.5 * sum(abs(value) for value in category_delta.values()),
            "target_action_classes": 0.5 * sum(abs(value) for value in action_delta.values()),
        },
    }


def text_shape_diagnostic(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare text lengths after construction; source rows never seed train text."""

    source_rows = load_jsonl(SOURCE_CASE_PATHS[1])

    def summarize(values: list[int]) -> dict[str, float | int]:
        return {
            "min": min(values),
            "mean": mean(values),
            "median": median(values),
            "max": max(values),
        }

    def field_lengths(selected: list[dict[str, Any]]) -> dict[str, list[int]]:
        return {
            "user_query_chars": [len(str(row.get("user_query") or "")) for row in selected],
            "observation_chars": [
                len(
                    str(
                        ((row.get("mock_observations") or [{}])[0].get("observation") or {}).get(
                            "summary"
                        )
                        or ""
                    )
                )
                for row in selected
            ],
        }

    source_lengths = field_lengths(source_rows)
    train_lengths = field_lengths(rows)
    source_stats = {key: summarize(values) for key, values in source_lengths.items()}
    train_stats = {key: summarize(values) for key, values in train_lengths.items()}
    relative_mean_delta = {
        key: (float(train_stats[key]["mean"]) - float(source_stats[key]["mean"]))
        / float(source_stats[key]["mean"])
        for key in source_stats
    }
    return {
        "status": "pass"
        if max(abs(value) for value in relative_mean_delta.values()) <= 0.15
        else "fail",
        "audit_only": True,
        "source_rows_used_for_construction": False,
        "source": source_stats,
        "train": train_stats,
        "relative_mean_delta": relative_mean_delta,
        "maximum_allowed_absolute_relative_mean_delta": 0.15,
    }


def _first_arg_token(row: dict[str, Any], field: str) -> str:
    decisions = row.get("expected_decisions")
    if not isinstance(decisions, list):
        return ""
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        contains = decision.get("arg_contains")
        if not isinstance(contains, dict):
            continue
        tokens = contains.get(field)
        if isinstance(tokens, list) and tokens:
            return str(tokens[0])
        if tokens:
            return str(tokens)
    return ""


def _project_entity(row: dict[str, Any]) -> str:
    explicit = str(row.get("project_entity") or "")
    if explicit:
        return explicit
    token = _first_arg_token(row, "user_query")
    if token:
        return token
    query = str(row.get("user_query") or "")
    for pattern in (r"判断(.+?)是否需要能力迁移", r"业务实体是(.+?)，最终要决定"):
        match = re.search(pattern, query)
        if match:
            return match.group(1)
    return ""


def _error_alias(row: dict[str, Any]) -> str:
    explicit = str(row.get("error_alias") or "")
    if explicit:
        return explicit
    observations = row.get("mock_observations")
    if not isinstance(observations, list):
        return ""
    for observation in observations:
        if not isinstance(observation, dict):
            continue
        summary = str((observation.get("observation") or {}).get("summary") or "")
        match = re.search(r"gateway_error=([^；。\s]+)", summary)
        if match:
            return match.group(1)
    return ""


def _fixture_hash(relative_path: str) -> str:
    path = ROOT / relative_path
    return sha256(path) if relative_path and path.is_file() else ""


def protected_values(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {
        "case_id": set(),
        "entity_id": set(),
        "project_entity": set(),
        "target_entity": set(),
        "normalized_query": set(),
        "template_id": set(),
        "error_alias": set(),
        "fixture_family": set(),
        "fixture_path": set(),
        "fixture_sha256": set(),
        "scenario_id": set(),
    }
    for row in rows:
        fixture_path = str((row.get("setup") or {}).get("image_fixture") or "")
        candidates = {
            "case_id": str(row.get("case_id") or ""),
            "entity_id": str(row.get("entity_id") or ""),
            "project_entity": _project_entity(row),
            "target_entity": str(row.get("target_entity") or _first_arg_token(row, "label")),
            "normalized_query": normalize_query(str(row.get("user_query") or "")),
            "template_id": str(row.get("template_id") or ""),
            "error_alias": _error_alias(row),
            "fixture_family": str(row.get("image_fixture_family") or ""),
            "fixture_path": fixture_path,
            "fixture_sha256": _fixture_hash(fixture_path),
            "scenario_id": str(row.get("scenario_id") or ""),
        }
        for key, value in candidates.items():
            if value:
                values[key].add(value)
    return values


def overlap_report(
    train_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    *,
    label: str,
    files_scanned: list[str],
) -> dict[str, Any]:
    train_values = protected_values(train_rows)
    reference_values = protected_values(reference_rows)
    overlaps = {
        key: sorted(train_values[key] & reference_values[key])
        for key in (*PROTECTED_OVERLAP_KEYS, "scenario_id")
    }
    protected_counts = {f"{key}_overlap": len(overlaps[key]) for key in PROTECTED_OVERLAP_KEYS}
    return {
        "label": label,
        "status": "pass" if all(value == 0 for value in protected_counts.values()) else "fail",
        "files_scanned": files_scanned,
        **protected_counts,
        "shared_taxonomy": {
            "scenario_id_overlap": len(overlaps["scenario_id"]),
            "scenario_ids": overlaps["scenario_id"],
            "expected": True,
        },
        "overlap_examples": {
            key: values[:10]
            for key, values in overlaps.items()
            if key in PROTECTED_OVERLAP_KEYS and values
        },
    }


def isolation_report(rows: list[dict[str, Any]], *, output_case_path: Path = CASE_PATH) -> dict[str, Any]:
    source_rows = [row for path in SOURCE_CASE_PATHS for row in load_jsonl(path)]
    source_check = overlap_report(
        rows,
        source_rows,
        label="v5_calibration_and_confirmation",
        files_scanned=[str(path.relative_to(ROOT)) for path in SOURCE_CASE_PATHS],
    )
    existing_paths = [
        path
        for path in sorted(CASE_DIR.glob("*_cases.jsonl"))
        if path.resolve() != output_case_path.resolve()
    ]
    existing_rows = [row for path in existing_paths for row in load_jsonl(path)]
    repository_check = overlap_report(
        rows,
        existing_rows,
        label="all_other_repository_case_files",
        files_scanned=[str(path.relative_to(ROOT)) for path in existing_paths],
    )
    return {
        "status": "pass"
        if source_check["status"] == repository_check["status"] == "pass"
        else "fail",
        "construction_source": (
            "aggregate V5 confirmation distribution metadata only; V5 rows are read after "
            "construction solely for protected-field overlap auditing"
        ),
        "protected_fields": list(PROTECTED_OVERLAP_KEYS),
        "v5_evaluation": source_check,
        "repository_cases": repository_check,
    }


def validate_rows(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    if len(rows) != TRAIN_CASES:
        errors.append(f"expected {TRAIN_CASES} cases, got {len(rows)}")
    ids = [str(row.get("case_id") or "") for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        errors.append("case IDs must be non-empty and unique")
    if len({str(row.get("entity_id") or "") for row in rows}) != TRAIN_ENTITY_COUNT:
        errors.append(f"expected {TRAIN_ENTITY_COUNT} entity groups")

    expected_mix = Counter({"migrate": 36, "retry": 24})
    for scenario in SUPPORTED_SCENARIOS:
        selected = [row for row in rows if row.get("scenario_id") == scenario]
        if len(selected) != TRAIN_ENTITY_COUNT:
            errors.append(f"{scenario}: expected {TRAIN_ENTITY_COUNT} cases, got {len(selected)}")
        mix = Counter(str(row.get("target_action_class") or "") for row in selected)
        if mix != expected_mix:
            errors.append(f"{scenario}: action mix {dict(mix)} != {dict(expected_mix)}")
        for alias in PACK_SPECS[scenario.split("_", 1)[1]]["aliases"]:
            alias_mix = Counter(
                str(row.get("target_action_class") or "")
                for row in selected
                if row.get("error_alias") == alias
            )
            if alias_mix != Counter({"migrate": 18, "retry": 12}):
                errors.append(f"{scenario}/{alias}: alias mix is {dict(alias_mix)}")

    for entity_id in sorted({str(row.get("entity_id") or "") for row in rows}):
        block = [row for row in rows if row.get("entity_id") == entity_id]
        if len(block) != len(SUPPORTED_SCENARIOS):
            errors.append(f"{entity_id}: incomplete scenario block")
        if len({str(row.get("target_action_class") or "") for row in block}) != 1:
            errors.append(f"{entity_id}: action class must be blocked across scenarios")
        for detector in ("qwen", "rex"):
            queries = {
                str(row.get("user_query") or "")
                for row in block
                if row.get("detector_family") == detector
            }
            if len(queries) != 1:
                errors.append(f"{entity_id}/{detector}: query must be blocked-identical")

    for row in rows:
        case_id = str(row.get("case_id") or "")
        if row.get("training_only") is not True:
            errors.append(f"{case_id}: training_only must be true")
        if row.get("evaluation_only") is not False or row.get("exclude_from_training") is not False:
            errors.append(f"{case_id}: training/evaluation flags are inconsistent")
        if row.get("split") != "train" or row.get("grpo_target_step") != 2:
            errors.append(f"{case_id}: split or GRPO target step is invalid")
        expected = row.get("expected_decisions")
        observations = row.get("mock_observations")
        if not isinstance(expected, list) or len(expected) != 2:
            errors.append(f"{case_id}: exactly two decisions are required")
        if not isinstance(observations, list) or len(observations) != 1:
            errors.append(f"{case_id}: exactly one observation is required")
            continue
        summary = str((observations[0].get("observation") or {}).get("summary") or "")
        for hint in CURRENT_OBSERVATION_FORBIDDEN_HINTS:
            if hint in summary:
                errors.append(f"{case_id}: observation leaks action hint {hint!r}")
        if f"gateway_error={row.get('error_alias')}" not in summary:
            errors.append(f"{case_id}: explicit error alias does not match observation")
        query = str(row.get("user_query") or "")
        if str(row.get("project_entity") or "") not in query:
            errors.append(f"{case_id}: project entity missing from query")
        if str(row.get("target_entity") or "") not in query:
            errors.append(f"{case_id}: target entity missing from query")
        fixture = ROOT / str((row.get("setup") or {}).get("image_fixture") or "")
        if not fixture.is_file():
            errors.append(f"{case_id}: missing fixture {fixture}")
        if not score_case(row)["passed"]:
            errors.append(f"{case_id}: canonical gold fails the strict reward contract")

    distribution = distribution_report(rows)
    if distribution["status"] != "pass":
        errors.append(f"aggregate distribution mismatch: {distribution['probability_delta']}")
    return errors


def build_manifest(rows: list[dict[str, Any]], case_path: Path) -> dict[str, Any]:
    source = source_report()
    expected_source_hash = str((source.get("sha256") or {}).get("cases") or "")
    actual_source_hash = sha256(SOURCE_CASE_PATHS[1])
    if actual_source_hash != expected_source_hash:
        raise ValueError("sealed V5 confirmation hash no longer matches its build report")
    isolation = isolation_report(rows, output_case_path=case_path)
    if isolation["status"] != "pass":
        raise ValueError(f"training/evaluation isolation failed: {isolation}")
    distribution = distribution_report(rows)
    if distribution["status"] != "pass":
        raise ValueError(f"training distribution mismatch: {distribution}")
    distribution["text_shape_diagnostic"] = text_shape_diagnostic(rows)
    if distribution["text_shape_diagnostic"]["status"] != "pass":
        raise ValueError(
            f"training text shape mismatch: {distribution['text_shape_diagnostic']}"
        )
    fixture_paths = [ROOT / fixture[1] for fixture in TRAIN_FIXTURES]
    files = {
        "train_cases": str(case_path.relative_to(ROOT)),
        "source_confirmation_report": str(SOURCE_REPORT_PATH.relative_to(ROOT)),
        "source_calibration_cases": str(SOURCE_CASE_PATHS[0].relative_to(ROOT)),
        "source_confirmation_cases": str(SOURCE_CASE_PATHS[1].relative_to(ROOT)),
        **{
            f"fixture_{index + 1}": str(path.relative_to(ROOT))
            for index, path in enumerate(fixture_paths)
        },
    }
    return {
        "schema_version": "1.0",
        "dataset_id": DATASET_ID,
        "role": "optimization_training_pool",
        "source_evaluation_dataset_id": SOURCE_DATASET_ID,
        "source_evaluation_split": "confirmation_aggregate_distribution",
        "seed": SEED,
        "construction": {
            "method": "independent synthetic analogue from aggregate taxonomy and class counts",
            "copied_evaluation_rows": 0,
            "train_entity_groups": TRAIN_ENTITY_COUNT,
            "fixed_grpo_target_step": 2,
        },
        "stats": {
            **case_stats(rows),
            "entity_groups": len({str(row["entity_id"]) for row in rows}),
            "project_entities": len({str(row["project_entity"]) for row in rows}),
            "target_entities": len({str(row["target_entity"]) for row in rows}),
            "target_action_classes": _counter(row["target_action_class"] for row in rows),
            "detector_families": _counter(row["detector_family"] for row in rows),
            "error_aliases": _counter(row["error_alias"] for row in rows),
            "fixture_families": _counter(row["image_fixture_family"] for row in rows),
        },
        "distribution_match": distribution,
        "integrity": {
            "status": "pass",
            "canonical_gold_passed": sum(bool(score_case(row)["passed"]) for row in rows),
            "canonical_gold_total": len(rows),
            "entity_and_content_isolation": isolation,
            "source_confirmation_sha256_verified": True,
        },
        "files": files,
        "sha256": {name: sha256(ROOT / relative) for name, relative in files.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, default=CASE_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_fixture_images()
    rows = build_cases()
    errors = validate_rows(rows)
    if errors:
        raise ValueError("dataset validation failed:\n" + "\n".join(errors[:100]))
    case_path = args.case_dir / f"{DATASET_ID}_train_cases.jsonl"
    v1.write_jsonl(case_path, rows)
    manifest = build_manifest(rows, case_path)
    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    v1.write_json(args.dataset_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
