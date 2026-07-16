#!/usr/bin/env python3
"""Build V5 matched-state retry-versus-migrate Planner evaluation cases."""

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
    build_planner_multistep_grpo_value_v3 as v3,
)


v1 = v3.v1
DATASET_ID = "planner_multistep_grpo_value_v5"
CASE_DIR = ROOT / "training/planner_grpo_seed_v1/cases"
DATASET_DIR = ROOT / "data/datasets" / DATASET_ID
SEED = 20260715
CALIBRATION_ENTITY_COUNT = 20
CONFIRMATION_ENTITY_COUNT = 30
CONFIRMATION_FAMILY_COUNT = 8
DEFAULT_CONFIRMATION_CASES = CONFIRMATION_ENTITY_COUNT * CONFIRMATION_FAMILY_COUNT

PACK_SPECS: dict[str, dict[str, Any]] = {
    "timeout_nonretryable": {
        "strategy": "nonretryable",
        "calibration": ("timeout", "request_timeout"),
        "confirmation": ("deadline_exceeded", "gateway_deadline"),
    },
    "timeout_budget_exhausted": {
        "strategy": "exhausted",
        "calibration": ("timeout", "request_timeout"),
        "confirmation": ("deadline_exceeded", "gateway_deadline"),
    },
    "connection_nonretryable": {
        "strategy": "nonretryable",
        "calibration": ("connection_reset", "transport_reset"),
        "confirmation": ("socket_reset", "proxy_connection_lost"),
    },
    "connection_budget_exhausted": {
        "strategy": "exhausted",
        "calibration": ("connection_reset", "transport_reset"),
        "confirmation": ("socket_reset", "proxy_connection_lost"),
    },
    "payload_nonretryable": {
        "strategy": "nonretryable",
        "calibration": ("invalid_payload", "malformed_payload"),
        "confirmation": ("request_contract_error", "payload_schema_violation"),
    },
    "gateway_busy_budget_exhausted": {
        "strategy": "exhausted",
        "calibration": ("gateway_busy", "upstream_busy"),
        "confirmation": ("provider_overloaded", "gateway_capacity_busy"),
    },
}
PACKS = tuple(PACK_SPECS)
SCENARIOS = tuple(
    f"{detector}_{pack}"
    for pack in PACKS
    for detector in ("qwen", "rex")
)
SCENARIO_CODES = {
    scenario: f"{'Q' if scenario.startswith('qwen_') else 'R'}{index + 1:02d}"
    for index, scenario in enumerate(SCENARIOS)
}

CALIBRATION_FIXTURES = (
    ("持竿人员", "examples/images/fisherman.jpg", "v5_calibration_fisherman"),
    ("清运车辆", "examples/images/trash_truck.jpg", "v5_calibration_trash_truck"),
    ("烟雾", "examples/images/smoke.jpg", "v5_calibration_smoke"),
)
CONFIRMATION_FIXTURES = (
    ("背包", "examples/images/person_with_bag.png", "v5_confirmation_person_with_bag"),
    ("横幅", "examples/images/banner.jpg", "v5_confirmation_banner"),
)

SITE_ROOTS = (
    "近海能源运维区",
    "山地应急通信点",
    "高值器械周转仓",
    "城市地下泵站",
    "航空维修器材区",
    "危化品车辆待检区",
    "智能仓储分拨面",
    "轨道供电维护段",
    "沿江防汛巡查点",
    "低温生物样本库",
    "港口自动化堆场",
    "新能源装备测试线",
    "大型活动安保区",
    "水务远程监测站",
    "高架道路养护面",
)
PROJECT_SUFFIXES = (
    "异常预算合同",
    "工具路由复核",
    "服务边界验收",
    "现场异常处置",
    "跨域能力审查",
    "灰度接入核验",
)

CALIBRATION_STYLES = (
    "请执行本轮分支验收",
    "按当前批次的硬门槛处理",
    "请完成这次结果驱动的串行判断",
    "依据本轮状态表推进，不要套用历史结论",
)
CONFIRMATION_STYLES = (
    "完成本次异常预算检查",
    "按当前服务状态执行串行路由",
    "请处理这次工具返回后的后续动作",
    "依据本轮结构化状态继续任务",
    "只按当前批次字段完成能力复核",
)

CURRENT_OBSERVATION_FORBIDDEN_HINTS = v3.CURRENT_OBSERVATION_FORBIDDEN_HINTS
VALUE_REWARD = v3.VALUE_REWARD


def build_entities() -> dict[str, list[str]]:
    values = [
        f"{site}{suffix}{index + 2701:04d}号"
        for index, (site, suffix) in enumerate(
            (pair for site in SITE_ROOTS for suffix in PROJECT_SUFFIXES for pair in [(site, suffix)])
        )
    ]
    random.Random(SEED + 5).shuffle(values)
    required = CALIBRATION_ENTITY_COUNT + CONFIRMATION_ENTITY_COUNT
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
        values = ["migrate"] * 12 + ["retry"] * 8
        seed = SEED + 501
    else:
        values = ["migrate"] * 18 + ["retry"] * 12
        seed = SEED + 502
    random.Random(seed).shuffle(values)
    return values


ACTION_SCHEDULES = {
    "calibration": action_schedule("calibration"),
    "confirmation": action_schedule("confirmation"),
}


def policy_query(*, style: str, project: str, target: str, model_text: str) -> str:
    # Keep the policy structure that V3 showed the reference can apply while
    # changing entity/style text and removing stale-history confounds.
    return v3.policy_query(
        split="calibration",
        style=style,
        project=project,
        target=target,
        model_text=model_text,
    )


def observation_for(
    *, split: str, pack: str, mode: str, entity_index: int
) -> dict[str, Any]:
    spec = PACK_SPECS[pack]
    aliases = spec[split]
    alias = aliases[entity_index % len(aliases)]
    if mode == "retry":
        retryable = True
        retry_count = 0
        badge = "red"
    elif mode == "migrate":
        retryable = spec["strategy"] == "exhausted"
        retry_count = 1 + entity_index % 2 if retryable else 0
        badge = "amber"
    else:
        raise ValueError(f"unknown target mode: {mode}")
    return v3.metric_observation(
        1,
        status="gateway_error",
        candidate_count=None,
        min_confidence=None,
        cross_prompt_iou=None,
        domain_shift="unknown",
        gateway_error=alias,
        retryable=retryable,
        retry_count=retry_count,
        badge=badge,
    )


def detection_step(action: str, target: str) -> dict[str, Any]:
    return v3.detection_step(action, target)


def migration_step(project: str) -> dict[str, Any]:
    return v3.migration_step(project)


def make_case(
    *, split: str, entity_index: int, project: str, scenario: str
) -> dict[str, Any]:
    target, fixture, fixture_family = fixture_for(split, entity_index)
    styles = CALIBRATION_STYLES if split == "calibration" else CONFIRMATION_STYLES
    style = styles[entity_index % len(styles)]
    action, model_text, detector, pack = detector_parts(scenario)
    mode = ACTION_SCHEDULES[split][entity_index]
    expected = [
        detection_step(action, target),
        migration_step(project) if mode == "migrate" else detection_step(action, target),
    ]
    entity_id = f"pmgv5_{split}_entity_{entity_index + 1:03d}"
    allowed = v1.allowed_actions(expected)
    return {
        "case_id": f"PMGV5-{split.upper()}-{entity_index + 1:03d}-{SCENARIO_CODES[scenario]}",
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
        "template_id": f"v5_policy_{detector}_{split}_{entity_index % len(styles) + 1}",
        "scenario_id": scenario,
        "category": scenario,
        "detector_family": detector,
        "state_contract_pack": pack,
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
        "provenance_class": "v5_matched_state_from_v3_hard_failure_gap",
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


def validate_rows(rows: list[dict[str, Any]], *, split: str) -> list[str]:
    errors: list[str] = []
    ids = [str(row.get("case_id") or "") for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        errors.append("case IDs must be unique and non-empty")
    expected_per_family = CALIBRATION_ENTITY_COUNT if split == "calibration" else CONFIRMATION_ENTITY_COUNT
    expected_mix = (
        {"migrate": 12, "retry": 8}
        if split == "calibration"
        else {"migrate": 18, "retry": 12}
    )
    families = Counter(str(row.get("scenario_id") or "") for row in rows)
    if any(count != expected_per_family for count in families.values()):
        errors.append(f"every family must contain {expected_per_family} cases")
    for scenario in families:
        mix = Counter(
            str(row.get("target_action_class") or "")
            for row in rows
            if str(row.get("scenario_id") or "") == scenario
        )
        if dict(mix) != expected_mix:
            errors.append(f"{scenario}: action mix {dict(mix)} != {expected_mix}")
    for row in rows:
        cid = str(row.get("case_id") or "")
        expected = row.get("expected_decisions")
        observations = row.get("mock_observations")
        if not isinstance(expected, list) or len(expected) != 2:
            errors.append(f"{cid}: exactly two decisions are required")
        if not isinstance(observations, list) or len(observations) != 1:
            errors.append(f"{cid}: exactly one observation is required")
            continue
        summary = str((observations[0].get("observation") or {}).get("summary") or "")
        for hint in CURRENT_OBSERVATION_FORBIDDEN_HINTS:
            if hint in summary:
                errors.append(f"{cid}: observation leaks action hint {hint!r}")
        fixture = ROOT / str((row.get("setup") or {}).get("image_fixture") or "")
        if not fixture.is_file():
            errors.append(f"{cid}: missing fixture {fixture}")
        if row.get("evaluation_only") is not True or row.get("exclude_from_training") is not True:
            errors.append(f"{cid}: evaluation-only flags missing")
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
    isolation = v1.split_integrity(generated)
    if isolation["status"] not in {"pass", "not_applicable"}:
        raise ValueError(f"V5 split isolation failed: {isolation}")
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
            "generated_split_isolation": isolation,
        },
        "files": {"cases": str(case_path.relative_to(ROOT))},
        "sha256": {"cases": v1.sha256(case_path)},
    }
    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    v1.write_json(args.dataset_dir / f"build_report_{args.split}.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
