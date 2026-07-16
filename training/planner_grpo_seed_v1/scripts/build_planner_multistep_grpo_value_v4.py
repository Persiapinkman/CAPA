#!/usr/bin/env python3
"""Build the V4 sealed Planner evaluation for GRPO-value qualification.

V4 is deliberately narrow: after a required detector call, the Planner must
apply a structured technical-error contract and choose migration, one retry of
the same detector, or end.  Calibration selects complete policy-pack families;
confirmation uses disjoint entities, fixtures, styles, and error aliases.
"""

from __future__ import annotations

import argparse
import json
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

from capa.evaluation.dataset_audit import case_stats  # noqa: E402
from training.planner_grpo_seed_v1.scripts import (  # noqa: E402
    build_planner_multistep_grpo_hard_v1 as v1,
)


DATASET_ID = "planner_multistep_grpo_value_v4"
CASE_DIR = ROOT / "training/planner_grpo_seed_v1/cases"
DATASET_DIR = ROOT / "data/datasets" / DATASET_ID
SEED = 20260715
CALIBRATION_ENTITY_COUNT = 20
CONFIRMATION_ENTITY_COUNT = 30
CONFIRMATION_FAMILY_COUNT = 8
DEFAULT_CONFIRMATION_CASES = CONFIRMATION_ENTITY_COUNT * CONFIRMATION_FAMILY_COUNT

PACKS = (
    "payload_contract",
    "authorization_contract",
    "quota_contract",
    "response_schema_contract",
    "transport_contract",
    "upstream_contract",
)
SCENARIOS = tuple(
    f"{detector}_{pack}"
    for pack in PACKS
    for detector in ("qwen", "rex")
)
SCENARIO_CODES = {
    scenario: f"{'Q' if scenario.startswith('qwen_') else 'R'}{index + 1:02d}"
    for index, scenario in enumerate(SCENARIOS)
}

PACK_ERROR_ALIASES: dict[str, dict[str, dict[str, tuple[str, ...]]]] = {
    "payload_contract": {
        "calibration": {
            "migrate": ("invalid_payload", "malformed_request_body"),
            "retry": ("payload_gateway_timeout", "payload_backend_busy"),
        },
        "confirmation": {
            "migrate": ("request_contract_violation", "unsupported_media_envelope"),
            "retry": ("payload_proxy_timeout", "transient_payload_queue_busy"),
        },
    },
    "authorization_contract": {
        "calibration": {
            "migrate": ("authorization_denied", "credential_scope_denied"),
            "retry": ("identity_provider_timeout", "temporary_token_introspection_busy"),
        },
        "confirmation": {
            "migrate": ("policy_scope_rejected", "tenant_permission_denied"),
            "retry": ("auth_proxy_timeout", "transient_identity_backend_reset"),
        },
    },
    "quota_contract": {
        "calibration": {
            "migrate": ("tenant_quota_exceeded", "capacity_policy_rejected"),
            "retry": ("quota_service_timeout", "temporary_capacity_probe_busy"),
        },
        "confirmation": {
            "migrate": ("project_limit_reached", "admission_quota_denied"),
            "retry": ("limit_registry_timeout", "transient_admission_queue_busy"),
        },
    },
    "response_schema_contract": {
        "calibration": {
            "migrate": ("invalid_response_schema", "decoder_contract_broken"),
            "retry": ("schema_registry_timeout", "temporary_decoder_busy"),
        },
        "confirmation": {
            "migrate": ("response_envelope_mismatch", "malformed_gateway_body"),
            "retry": ("contract_registry_timeout", "transient_decoder_reset"),
        },
    },
    "transport_contract": {
        "calibration": {
            "migrate": ("tls_policy_rejected", "network_route_forbidden"),
            "retry": ("transient_connection_reset", "temporary_transport_timeout"),
        },
        "confirmation": {
            "migrate": ("certificate_policy_denied", "egress_route_rejected"),
            "retry": ("transient_socket_reset", "temporary_proxy_timeout"),
        },
    },
    "upstream_contract": {
        "calibration": {
            "migrate": ("upstream_service_unavailable", "backend_contract_rejected"),
            "retry": ("transient_upstream_timeout", "temporary_backend_busy"),
        },
        "confirmation": {
            "migrate": ("provider_capability_unavailable", "backend_policy_denied"),
            "retry": ("transient_provider_reset", "temporary_upstream_queue_busy"),
        },
    },
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
    ("持竿人员", "examples/images/fisherman.jpg", "v4_calibration_fisherman"),
    ("清运车辆", "examples/images/trash_truck.jpg", "v4_calibration_trash_truck"),
    ("烟雾", "examples/images/smoke.jpg", "v4_calibration_smoke"),
)
CONFIRMATION_FIXTURES = (
    ("背包", "examples/images/person_with_bag.png", "v4_confirmation_person_with_bag"),
    ("横幅", "examples/images/banner.jpg", "v4_confirmation_banner"),
)

SITE_ROOTS = (
    "临港储能试验区",
    "山区输电巡检段",
    "精密仪器中转库",
    "地下综合管廊",
    "离岸风电运维站",
    "航空货运安检区",
    "城市防汛物资点",
    "化工园区缓冲带",
    "轨道车辆调试线",
    "冷链药品交接仓",
    "智能港机作业面",
    "新能源电池暂存区",
    "水文遥测维护点",
    "高架桥梁检修面",
    "大型会展卸货区",
)
PROJECT_SUFFIXES = (
    "边界合同核验",
    "视觉服务准入",
    "工具链异常处置",
    "现场能力复核",
    "灰度接入审查",
    "跨域上线检查",
)

CALIBRATION_STYLES = (
    "按本轮异常处置合同推进",
    "请完成当前批次的两阶段路由",
    "只依据这次工具状态执行",
    "按最新 observation 完成验收",
)
CONFIRMATION_STYLES = (
    "依据本次现场检查单处理",
    "请执行当前样本的串行判定",
    "本轮以最新结构化状态为准",
    "按这次服务边界协议推进",
    "完成当前任务的工具状态路由",
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
        f"{site}{suffix}{index + 1701:04d}号"
        for index, (site, suffix) in enumerate(
            (pair for site in SITE_ROOTS for suffix in PROJECT_SUFFIXES for pair in [(site, suffix)])
        )
    ]
    random.Random(SEED + 4).shuffle(values)
    required = CALIBRATION_ENTITY_COUNT + CONFIRMATION_ENTITY_COUNT
    if len(values) < required:
        raise ValueError("insufficient V4 entity names")
    return {
        "calibration": values[:CALIBRATION_ENTITY_COUNT],
        "confirmation": values[CALIBRATION_ENTITY_COUNT:required],
    }


ENTITY_SPLITS = build_entities()


def parse_scenarios(raw: str) -> list[str]:
    values = [value.strip() for value in re.split(r"[,\s]+", raw) if value.strip()]
    unknown = sorted(set(values) - set(SCENARIOS))
    if unknown:
        raise ValueError(f"unknown scenarios: {unknown}")
    return list(dict.fromkeys(values))


def detector_parts(scenario: str) -> tuple[str, str, str, str]:
    is_qwen = scenario.startswith("qwen_")
    detector = "qwen" if is_qwen else "rex"
    return (
        "qwen_detection" if is_qwen else "rexomni_detection",
        "Qwen" if is_qwen else "Rex-Omni",
        detector,
        scenario.removeprefix(f"{detector}_"),
    )


def fixture_for(split: str, entity_index: int) -> tuple[str, str, str]:
    fixtures = CALIBRATION_FIXTURES if split == "calibration" else CONFIRMATION_FIXTURES
    return fixtures[entity_index % len(fixtures)]


def action_schedule(split: str) -> list[str]:
    if split == "calibration":
        values = ["migrate"] * 12 + ["retry"] * 6 + ["end"] * 2
        seed = SEED + 401
    else:
        values = ["migrate"] * 18 + ["retry"] * 9 + ["end"] * 3
        seed = SEED + 402
    random.Random(seed).shuffle(values)
    return values


ACTION_SCHEDULES = {
    "calibration": action_schedule("calibration"),
    "confirmation": action_schedule("confirmation"),
}


def policy_query(*, style: str, project: str, target: str, model_text: str) -> str:
    return (
        f"{style}：第一步必须用{model_text}检查附件中的{target}，且该检测步骤 "
        "finish_after_tool=false。拿到 observation 后，只读取当前 query 最新一步的结构化字段；"
        "历史批次、error_note、overall_badge 和看板颜色均不能覆盖以下规则。"
        "若 gateway_error 不为 none，仅在 retryable=true 且 retry_count=0 时原检测工具重试一次；"
        "retryable=false 或 retry_count>=1 时调用迁移顾问。若 gateway_error=none，只有 "
        "candidate_count>=1、min_confidence>=0.88、cross_prompt_iou>=0.72 且 domain_shift=low "
        "四项同时成立才结束，否则调用迁移顾问。迁移顾问必须携带当前任务描述，使用当前图片、"
        f"开启内部视觉探针并 finish_after_tool=true。当前任务标识为{project}。"
    )


def metric_observation(
    *,
    gateway_error: str,
    retryable: bool,
    retry_count: int,
    candidate_count: int | None,
    min_confidence: float | None,
    cross_prompt_iou: float | None,
    domain_shift: str,
    badge: str,
    error_note: str,
) -> dict[str, Any]:
    fields = (
        f"candidate_count={candidate_count if candidate_count is not None else 'NA'}；"
        f"min_confidence={min_confidence if min_confidence is not None else 'NA'}；"
        f"cross_prompt_iou={cross_prompt_iou if cross_prompt_iou is not None else 'NA'}；"
        f"domain_shift={domain_shift}；gateway_error={gateway_error}；"
        f"retryable={'true' if retryable else 'false'}；retry_count={retry_count}；"
        f"overall_badge={badge}；error_note={error_note}"
    )
    return v1.observation(
        1,
        success=gateway_error == "none",
        status="completed" if gateway_error == "none" else "gateway_error",
        summary="当前工具状态：" + fields + "。",
    )


def observation_for(
    *, split: str, pack: str, mode: str, entity_index: int
) -> dict[str, Any]:
    aliases = PACK_ERROR_ALIASES[pack][split]
    if mode == "end":
        return metric_observation(
            gateway_error="none",
            retryable=False,
            retry_count=0,
            candidate_count=2 + entity_index % 2,
            min_confidence=0.91 + 0.01 * (entity_index % 3),
            cross_prompt_iou=0.80 + 0.02 * (entity_index % 3),
            domain_shift="low",
            badge="amber" if entity_index % 2 else "green",
            error_note="archived_warning_visible" if entity_index % 2 else "none",
        )
    alias = aliases[mode][entity_index % len(aliases[mode])]
    if mode == "retry":
        return metric_observation(
            gateway_error=alias,
            retryable=True,
            retry_count=0,
            candidate_count=None,
            min_confidence=None,
            cross_prompt_iou=None,
            domain_shift="unknown",
            badge="red",
            error_note="operator_note_says_abort" if entity_index % 2 else "transient",
        )
    if mode != "migrate":
        raise ValueError(f"unknown action mode: {mode}")
    exhausted = entity_index % 3 != 0
    decoy_metrics = entity_index % 2 == 0
    return metric_observation(
        gateway_error=alias,
        retryable=exhausted,
        retry_count=(1 + entity_index % 2) if exhausted else 0,
        candidate_count=2 if decoy_metrics else None,
        min_confidence=0.95 if decoy_metrics else None,
        cross_prompt_iou=0.84 if decoy_metrics else None,
        domain_shift="low" if decoy_metrics else "unknown",
        badge="green" if decoy_metrics else "amber",
        error_note="dashboard_says_continue" if decoy_metrics else "manual_review_pending",
    )


def detection_step(action: str, target: str) -> dict[str, Any]:
    return v1.tool_step(
        action,
        required={"finish_after_tool": False},
        contains={"label": [target]},
    )


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


def make_case(
    *, split: str, entity_index: int, project: str, scenario: str
) -> dict[str, Any]:
    target, fixture, fixture_family = fixture_for(split, entity_index)
    styles = CALIBRATION_STYLES if split == "calibration" else CONFIRMATION_STYLES
    style = styles[entity_index % len(styles)]
    action, model_text, detector, pack = detector_parts(scenario)
    mode = ACTION_SCHEDULES[split][entity_index]
    first = detection_step(action, target)
    if mode == "migrate":
        second = migration_step(project)
    elif mode == "retry":
        second = detection_step(action, target)
    else:
        second = v1.end_step()
    expected = [first, second]
    entity_id = f"pmgv4_{split}_entity_{entity_index + 1:03d}"
    allowed = v1.allowed_actions(expected)
    return {
        "case_id": f"PMGV4-{split.upper()}-{entity_index + 1:03d}-{SCENARIO_CODES[scenario]}",
        "dataset_id": DATASET_ID,
        "split": split,
        "selection_role": (
            "visible_whole_family_calibration"
            if split == "calibration"
            else "sealed_confirmation_unfiltered"
        ),
        "entity_id": entity_id,
        "group_id": entity_id,
        "counterfactual_bundle_id": f"{entity_id}_{detector}",
        "template_id": f"v4_policy_{detector}_{split}_{entity_index % len(styles) + 1}",
        "scenario_id": scenario,
        "category": scenario,
        "detector_family": detector,
        "policy_pack": pack,
        "target_action_class": mode,
        "grpo_target_step": 2,
        "user_query": policy_query(
            style=style,
            project=project,
            target=target,
            model_text=model_text,
        ),
        "image_fixture_family": fixture_family,
        "setup": {**v1.setup(fixture=fixture), "query_trajectories": []},
        "expected_decisions": expected,
        "mock_observations": [
            observation_for(
                split=split,
                pack=pack,
                mode=mode,
                entity_index=entity_index,
            )
        ],
        "forbidden_actions": [tool for tool in v1.ACTIVE_TOOLS if tool not in allowed],
        "reward_spec": VALUE_REWARD,
        "provenance_class": "v4_policy_pack_from_v3_hard_failure_calibration_gap",
        "evaluation_only": True,
        "exclude_from_training": True,
    }


def build_split(
    *, split: str, accepted_scenarios: list[str], confirmation_cases: int
) -> list[dict[str, Any]]:
    if split == "calibration":
        scenarios = list(SCENARIOS)
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
    expected = len(entities) * len(scenarios)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} rows, got {len(rows)}")
    return rows


def validate_rows(rows: list[dict[str, Any]], *, split: str) -> list[str]:
    errors: list[str] = []
    ids = [str(row.get("case_id") or "") for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        errors.append("case IDs must be unique and non-empty")
    family_counts = Counter(str(row.get("scenario_id") or "") for row in rows)
    expected_per_family = CALIBRATION_ENTITY_COUNT if split == "calibration" else CONFIRMATION_ENTITY_COUNT
    if any(value != expected_per_family for value in family_counts.values()):
        errors.append(f"every family must contain {expected_per_family} entity cases")
    for scenario in family_counts:
        mode_counts = Counter(
            str(row.get("target_action_class") or "")
            for row in rows
            if str(row.get("scenario_id") or "") == scenario
        )
        expected_modes = (
            {"migrate": 12, "retry": 6, "end": 2}
            if split == "calibration"
            else {"migrate": 18, "retry": 9, "end": 3}
        )
        if dict(mode_counts) != expected_modes:
            errors.append(f"{scenario}: action mix {dict(mode_counts)} != {expected_modes}")
    for row in rows:
        cid = str(row.get("case_id") or "")
        expected = row.get("expected_decisions")
        observations = row.get("mock_observations")
        if not isinstance(expected, list) or len(expected) != 2:
            errors.append(f"{cid}: expected_decisions must contain exactly two steps")
            continue
        if not isinstance(observations, list) or len(observations) != 1:
            errors.append(f"{cid}: exactly one observation is required")
            continue
        if int(observations[0].get("after_step") or 0) != 1:
            errors.append(f"{cid}: observation must follow step 1")
        summary = str((observations[0].get("observation") or {}).get("summary") or "")
        for hint in CURRENT_OBSERVATION_FORBIDDEN_HINTS:
            if hint in summary:
                errors.append(f"{cid}: current observation leaks action hint {hint!r}")
        fixture = ROOT / str((row.get("setup") or {}).get("image_fixture") or "")
        if not fixture.is_file():
            errors.append(f"{cid}: missing fixture {fixture}")
        if row.get("evaluation_only") is not True or row.get("exclude_from_training") is not True:
            errors.append(f"{cid}: sealed evaluation flags are missing")
    for entity_id in {str(row.get("entity_id") or "") for row in rows}:
        block = [row for row in rows if str(row.get("entity_id") or "") == entity_id]
        for detector in ("qwen", "rex"):
            queries = {
                str(row.get("user_query") or "")
                for row in block
                if str(row.get("detector_family") or "") == detector
            }
            if len(queries) != 1:
                errors.append(f"{entity_id}/{detector}: policy query must be blocked-identical")
    return errors


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("calibration", "confirmation"), default="calibration")
    parser.add_argument("--accepted-scenarios", default="")
    parser.add_argument("--confirmation-cases", type=int, default=DEFAULT_CONFIRMATION_CASES)
    parser.add_argument("--case-dir", type=Path, default=CASE_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    accepted = parse_scenarios(args.accepted_scenarios)
    rows = build_split(
        split=args.split,
        accepted_scenarios=accepted,
        confirmation_cases=args.confirmation_cases,
    )
    errors = validate_rows(rows, split=args.split)
    if errors:
        raise ValueError("dataset validation failed:\n" + "\n".join(errors[:80]))

    case_path = args.case_dir / f"{DATASET_ID}_{args.split}_cases.jsonl"
    v1.write_jsonl(case_path, rows)
    calibration_path = args.case_dir / f"{DATASET_ID}_calibration_cases.jsonl"
    confirmation_path = args.case_dir / f"{DATASET_ID}_confirmation_cases.jsonl"
    generated = {
        key: value
        for key, value in {
            "calibration": load_jsonl(calibration_path),
            "confirmation": load_jsonl(confirmation_path),
        }.items()
        if value
    }
    current_paths = {calibration_path.resolve(), confirmation_path.resolve()}
    overlap = v1.existing_case_overlap(rows, current_paths)
    if overlap["status"] != "pass":
        raise ValueError(f"existing dataset overlap detected: {overlap['overlaps']}")
    split_isolation = v1.split_integrity(generated)
    if split_isolation["status"] not in {"pass", "not_applicable"}:
        raise ValueError(f"V4 split isolation failed: {split_isolation}")

    report = {
        "schema_version": "1.0",
        "dataset_id": DATASET_ID,
        "split": args.split,
        "seed": SEED,
        "selection_rule": "whole-family calibration only; confirmation unfiltered",
        "accepted_scenarios": accepted,
        "evaluation_only": True,
        "exclude_from_training": True,
        "cases": case_stats(rows),
        "entity_groups": len({str(row.get("entity_id") or "") for row in rows}),
        "families": dict(sorted(Counter(str(row.get("scenario_id") or "") for row in rows).items())),
        "target_action_classes": dict(
            sorted(Counter(str(row.get("target_action_class") or "") for row in rows).items())
        ),
        "integrity": {
            "existing_dataset_overlap": overlap,
            "generated_split_isolation": split_isolation,
        },
        "files": {"cases": str(case_path.relative_to(ROOT))},
        "sha256": {"cases": v1.sha256(case_path)},
    }
    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    v1.write_json(args.dataset_dir / f"build_report_{args.split}.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
