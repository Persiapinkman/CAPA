#!/usr/bin/env python3
"""Build preregistered V8 retry/safe-end residual data with sealed test cohorts."""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


from training.planner_grpo_seed_v1.scripts import (
    build_planner_retry_migrate_residual_v7 as v7,
)


ROOT = Path(__file__).resolve().parents[3]
DATASET_ID = "planner_retry_safe_end_residual_v8"
STUDY_ID = "planner_retry_migrate_residual_v8_qwen35_4b_v1"
SCHEMA_VERSION = "1.0"
SEED = 2026071608
CREATED_AT = "2026-07-16T18:05:00Z"
CASE_DIR = ROOT / "training/planner_grpo_seed_v1/cases"
STEP_DIR = ROOT / "training/planner_grpo_seed_v1/step_data"
DATASET_DIR = ROOT / "data/datasets" / DATASET_ID
FIXTURE_DIR = ROOT / "examples/images" / DATASET_ID
STUDY_DIR = ROOT / "experiments/studies" / STUDY_ID
DEFAULT_MODEL = Path("/raid/zkq/models/Qwen3.5-4B")

PRIMARY_SCENARIOS = (
    "current_success_step2",
    "fresh_retry_step2",
    "post_retry_success_step3",
)
CONTROL_SCENARIOS = (
    "post_retry_error_step3",
    "post_retry_metric_veto_step3",
    "conflicting_state_step2",
    "nonretryable_step2",
    "budget_exhausted_step2",
    "missing_required_state_step2",
)
ALL_SCENARIOS = PRIMARY_SCENARIOS + CONTROL_SCENARIOS
MATERIALIZED_SPLITS = ("grpo_train", "support_dev", "selection_dev")
SEALED_SPLITS = ("sealed_test_a", "sealed_test_b")

SPLIT_SPECS: dict[str, dict[str, Any]] = {
    "grpo_train": {
        "entities": 24,
        "code": "GT",
        "training_only": True,
        "evaluation_only": False,
        "exclude_from_training": False,
        "sealed": False,
        "role": "optimizer_only",
    },
    "support_dev": {
        "entities": 12,
        "code": "SD",
        "training_only": False,
        "evaluation_only": True,
        "exclude_from_training": True,
        "sealed": False,
        "role": "stochastic_support_gate_only",
    },
    "selection_dev": {
        "entities": 12,
        "code": "VD",
        "training_only": False,
        "evaluation_only": True,
        "exclude_from_training": True,
        "sealed": False,
        "role": "checkpoint_selection_only",
    },
    "sealed_test_a": {
        "entities": 12,
        "code": "TA",
        "training_only": False,
        "evaluation_only": True,
        "exclude_from_training": True,
        "sealed": True,
        "role": "sealed_confirmation_cohort_a",
    },
    "sealed_test_b": {
        "entities": 12,
        "code": "TB",
        "training_only": False,
        "evaluation_only": True,
        "exclude_from_training": True,
        "sealed": True,
        "role": "sealed_confirmation_cohort_b",
    },
}

LEXICONS: dict[str, dict[str, tuple[str, ...]]] = {
    "grpo_train": {
        "roots": (
            "近海风机变桨柜",
            "生物反应器取样岛",
            "高铁接触网监测间",
            "锂电化成消防区",
            "光刻胶冷藏站",
            "城市排涝泵控室",
        ),
        "suffixes": ("链路恢复判定", "视觉能力续用终审", "重试窗口复核", "迁移边界核销"),
        "styles": (
            "请按最新结构化回执完成本轮动作",
            "工单标题含故障字样，但只依据当前字段裁决",
            "迁移候选已登记，请先核验这次探针状态",
            "预算备注可能过期，以显式 retry_count 为准",
        ),
    },
    "support_dev": {
        "roots": ("极地通信中继舱", "晶圆缺陷复判台", "智能粮仓通风廊", "矿井提升机监控间"),
        "suffixes": ("重试资格会签", "当前能力终判", "异常回执核销"),
        "styles": (
            "读取最后一条 observation 后选择唯一后续动作",
            "不要被失联工单名干扰，按结构化重试条件处理",
            "先判断本轮是否仍有一次同检测器重试机会",
            "请区分当前成功、首次可重试错误与重试后成功",
        ),
    },
    "selection_dev": {
        "roots": ("海上平台消防泵舱", "量子测控低温机柜", "机场行李分拣环线", "地热回灌计量站"),
        "suffixes": ("状态边界验收", "探针恢复复议", "能力续用裁决"),
        "styles": (
            "依据最新机器回执完成动作终审",
            "历史告警只作背景，不得覆盖当前键值",
            "判断应结束、重试原检测器还是进入迁移",
            "请用 retryable 与 retry_count 的联合状态裁决",
        ),
    },
    "sealed_test_a": {
        "roots": ("高空飞艇能源舱", "深地实验排水站", "轨道交通信号电源室", "海洋牧场投喂平台"),
        "suffixes": ("回执闭环终验", "重试策略定级", "迁移条件复核"),
        "styles": (
            "完成当前视觉探针链路的最终路由",
            "工单背景出现迁移字样，仍须服从最新状态",
            "核对错误、重试次数与成功指标后继续",
            "只允许一次原检测器重试，随后必须收敛",
        ),
    },
    "sealed_test_b": {
        "roots": ("荒漠光热储能岛", "水下机器人维护坞", "航空货站冷链库", "城市地下管廊节点"),
        "suffixes": ("能力恢复会审", "异常状态终裁", "重试闭环核验"),
        "styles": (
            "请给出这次探针状态对应的唯一下一步",
            "标题中的耗尽与故障不能替代结构化字段",
            "根据当前回执判断继续、结束或迁移",
            "重试成功应结束，重试错误不得形成循环",
        ),
    },
}

POLICY_WORDING: dict[str, tuple[str, ...]] = {
    split: (
        "只读取当前 query 最后一条 observation；标题、历史记录、badge 和备注没有路由优先级。",
        "若 gateway_error 不是 none，仅当 retryable=true 且 retry_count=0 时重试同一检测器一次，否则进入迁移顾问。",
        "若 gateway_error=none，candidate_count>=1、min_confidence>=0.88、cross_prompt_iou>=0.72 且 domain_shift=low 时结束，否则迁移。",
        "重试后的回执只能结束或迁移，禁止第二次重试；字段缺失、非法或冲突均进入迁移顾问。",
    )
    for split in SPLIT_SPECS
}

ERROR_ALIASES: dict[str, dict[str, tuple[str, str]]] = {
    "grpo_train": {
        "timeout": ("probe_response_epoch_missed", "vision_ack_clock_overrun"),
        "transport": ("detector_stream_route_lost", "feature_return_tunnel_closed"),
        "quota": ("inspection_retry_slot_empty", "visual_compute_lease_denied"),
        "payload": ("probe_state_envelope_broken", "detector_contract_shape_invalid"),
    },
    "support_dev": {
        "timeout": ("sensor_result_window_elapsed", "vision_reply_tick_expired"),
        "transport": ("feature_backhaul_path_missing", "detector_control_link_dropped"),
        "quota": ("probe_replay_credit_spent", "vision_admission_slot_absent"),
        "payload": ("detector_state_frame_malformed", "inspection_request_schema_unknown"),
    },
    "selection_dev": {
        "timeout": ("visual_result_deadline_passed", "probe_response_horizon_closed"),
        "transport": ("detector_feedback_channel_unbound", "feature_exchange_session_lost"),
        "quota": ("vision_retry_token_depleted", "inspection_capacity_lease_locked"),
        "payload": ("probe_manifest_format_diverged", "detector_input_packet_corrupt"),
    },
    "sealed_test_a": {
        "timeout": ("inspection_ack_interval_ended", "visual_worker_clock_exceeded"),
        "transport": ("detector_result_carrier_unavailable", "feature_uplink_route_broken"),
        "quota": ("probe_retry_budget_withheld", "vision_execution_credit_exhausted"),
        "payload": ("detector_reply_bundle_invalid", "visual_probe_contract_unreadable"),
    },
    "sealed_test_b": {
        "timeout": ("vision_feedback_period_lapsed", "probe_completion_timer_closed"),
        "transport": ("feature_return_bus_detached", "detector_service_path_unreachable"),
        "quota": ("inspection_retry_lease_consumed", "visual_job_credit_missing"),
        "payload": ("probe_result_schema_fragmented", "detector_state_payload_unrecognized"),
    },
}

FIXTURES: dict[str, tuple[dict[str, Any], ...]] = {
    "grpo_train": (
        {"target": "翠绿双孔定位环", "slug": "emerald_double_aperture_ring", "family": "prsv8_gt_emerald_ring", "shape": "rings", "fg": (38, 155, 105), "bg": (229, 225, 214)},
        {"target": "靛蓝五星校验片", "slug": "indigo_five_star_check_tab", "family": "prsv8_gt_indigo_star", "shape": "star", "fg": (67, 82, 176), "bg": (43, 48, 57)},
    ),
    "support_dev": (
        {"target": "橙黄十字锁定块", "slug": "amber_cross_lock_block", "family": "prsv8_sd_amber_cross", "shape": "cross", "fg": (222, 145, 38), "bg": (42, 50, 60)},
        {"target": "紫罗兰三辐手轮", "slug": "violet_three_spoke_wheel", "family": "prsv8_sd_violet_wheel", "shape": "wheel", "fg": (139, 83, 184), "bg": (227, 224, 215)},
    ),
    "selection_dev": (
        {"target": "青铜弧顶隔离座", "slug": "bronze_arch_isolator", "family": "prsv8_vd_bronze_arch", "shape": "arch", "fg": (166, 118, 72), "bg": (44, 51, 60)},
        {"target": "湖蓝六角巡检片", "slug": "lake_blue_hex_tab", "family": "prsv8_vd_blue_hex", "shape": "hexagon", "fg": (49, 151, 190), "bg": (228, 224, 213)},
    ),
    "sealed_test_a": (
        {"target": "银灰双环耦合件", "slug": "silver_double_ring_coupler", "family": "prsv8_ta_silver_rings", "shape": "rings", "fg": (160, 169, 177), "bg": (39, 47, 56)},
        {"target": "玫红五星检修牌", "slug": "magenta_star_service_tag", "family": "prsv8_ta_magenta_star", "shape": "star", "fg": (194, 62, 124), "bg": (229, 226, 216)},
    ),
    "sealed_test_b": (
        {"target": "松绿十字定位销", "slug": "pine_cross_locator", "family": "prsv8_tb_pine_cross", "shape": "cross", "fg": (47, 128, 102), "bg": (225, 222, 211)},
        {"target": "珊瑚红三辐调节轮", "slug": "coral_three_spoke_adjuster", "family": "prsv8_tb_coral_wheel", "shape": "wheel", "fg": (211, 91, 78), "bg": (42, 49, 58)},
    ),
}


def v8_entity_id(split: str, entity_index: int) -> str:
    return f"prsv8_{SPLIT_SPECS[split]['code'].lower()}_entity_{entity_index + 1:03d}"


@contextlib.contextmanager
def configured_v7() -> Iterator[None]:
    names = (
        "DATASET_ID",
        "SCHEMA_VERSION",
        "SEED",
        "CREATED_AT",
        "STUDY_ID",
        "DATASET_DIR",
        "FIXTURE_DIR",
        "STUDY_DIR",
        "PRIMARY_SCENARIOS",
        "CONTROL_SCENARIOS",
        "ALL_SCENARIOS",
        "SPLIT_SPECS",
        "LEXICONS",
        "POLICY_WORDING",
        "ERROR_ALIASES",
        "FIXTURES",
        "PROJECTS",
        "FACTOR_LAYOUTS",
        "ALIAS_LAYOUTS",
        "entity_id",
        "base_case",
    )
    saved = {name: getattr(v7, name) for name in names}
    original_base_case = v7.base_case
    try:
        v7.DATASET_ID = DATASET_ID
        v7.SCHEMA_VERSION = SCHEMA_VERSION
        v7.SEED = SEED
        v7.CREATED_AT = CREATED_AT
        v7.STUDY_ID = STUDY_ID
        v7.DATASET_DIR = DATASET_DIR
        v7.FIXTURE_DIR = FIXTURE_DIR
        v7.STUDY_DIR = STUDY_DIR
        v7.PRIMARY_SCENARIOS = PRIMARY_SCENARIOS
        v7.CONTROL_SCENARIOS = CONTROL_SCENARIOS
        v7.ALL_SCENARIOS = ALL_SCENARIOS
        v7.SPLIT_SPECS = SPLIT_SPECS
        v7.LEXICONS = LEXICONS
        v7.POLICY_WORDING = POLICY_WORDING
        v7.ERROR_ALIASES = ERROR_ALIASES
        v7.FIXTURES = FIXTURES
        v7.entity_id = v8_entity_id
        v7.PROJECTS = {split: v7.build_projects(split) for split in SPLIT_SPECS}
        v7.FACTOR_LAYOUTS = {split: v7.factor_layout(split) for split in SPLIT_SPECS}
        v7.ALIAS_LAYOUTS = {split: v7.alias_layout(split) for split in SPLIT_SPECS}

        def base_case_v8(**kwargs: Any) -> dict[str, Any]:
            row = original_base_case(**kwargs)
            row["case_id"] = str(row["case_id"]).replace("PRRV7-", "PRSV8-", 1)
            row["template_id"] = str(row["template_id"]).replace("prrv7_", "prsv8_", 1)
            row["grpo_eligible"] = row["split"] == "grpo_train" and row["scenario_id"] in PRIMARY_SCENARIOS
            row["provenance_class"] = "independent_synthetic_retry_safe_end_factorial_v8"
            return row

        v7.base_case = base_case_v8
        yield
    finally:
        for name, value in saved.items():
            setattr(v7, name, value)


def build_cases_in_memory() -> dict[str, list[dict[str, Any]]]:
    with configured_v7():
        return v7.build_all_cases()


def historical_overlap(cases_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    current = v7.protected_values([row for rows in cases_by_split.values() for row in rows])
    historical_rows: list[dict[str, Any]] = []
    for path in CASE_DIR.glob("planner_retry_*_cases.jsonl"):
        if DATASET_ID in path.name:
            continue
        historical_rows.extend(v7.load_jsonl(path) if hasattr(v7, "load_jsonl") else [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        ])
    historical = v7.protected_values(historical_rows)
    return {field: len(current[field] & historical[field]) for field in current}


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = sorted(int(row["prompt_token_count"]) for row in rows)
    return {
        "rows": len(rows),
        "entities": len({str(row["entity_id"]) for row in rows}),
        "scenarios": dict(sorted(Counter(str(row["scenario_id"]) for row in rows).items())),
        "steps": dict(sorted(Counter(int(row["step_index"]) for row in rows).items())),
        "actions": dict(sorted(Counter(str(row["target_action_class"]) for row in rows).items())),
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
    parser.add_argument("--materialize-test", action="store_true")
    parser.add_argument("--confirm-materialize-test", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.materialize_test and args.confirm_materialize_test != "OPEN_V8_TEST":
        raise PermissionError("test materialization requires --confirm-materialize-test OPEN_V8_TEST")
    model_path = args.model_name_or_path.resolve()
    with configured_v7():
        fixture_paths = v7.write_fixture_images()
        cases_by_split = v7.build_all_cases()
        validation = v7.validate_cases(cases_by_split)
        overlap = historical_overlap(cases_by_split)
        if any(overlap.values()):
            validation["errors"].append(f"V8 protected values overlap prior retry datasets: {overlap}")
            validation["status"] = "fail"
        if validation["status"] != "pass":
            raise ValueError("V8 case validation failed:\n" + "\n".join(validation["errors"][:100]))

        case_paths: dict[str, Path] = {}
        for split in MATERIALIZED_SPLITS:
            path = CASE_DIR / f"{DATASET_ID}_{split}_cases.jsonl"
            v7.write_jsonl(path, cases_by_split[split])
            case_paths[split] = path

        sealed_rows = [row for split in SEALED_SPLITS for row in cases_by_split[split]]
        test_commitment = v7.sha256_text(v7.jsonl_text(sealed_rows))
        sealed_paths: dict[str, Path] = {}
        if args.materialize_test:
            for split in SEALED_SPLITS:
                path = STUDY_DIR / "sealed_test_data" / f"{split}_cases.jsonl"
                v7.write_jsonl(path, cases_by_split[split])
                sealed_paths[split] = path
            combined_path = STUDY_DIR / "sealed_test_data" / "combined_test_cases.jsonl"
            v7.write_jsonl(combined_path, sealed_rows)
            sealed_paths["combined"] = combined_path
            if v7.sha256_file(combined_path) != test_commitment:
                raise ValueError("materialized V8 test does not match frozen commitment")

        tokenizer = v7.load_tokenizer(model_path)
        step_rows: dict[str, list[dict[str, Any]]] = {}
        step_paths: dict[str, Path] = {}
        for split in MATERIALIZED_SPLITS:
            source = cases_by_split[split]
            if split in {"grpo_train", "support_dev"}:
                source = [row for row in source if row["scenario_id"] in PRIMARY_SCENARIOS]
            rows = v7.build_step_rows(source, tokenizer)
            audit = v7.v6.prompt_audit(rows)
            if audit["status"] != "pass":
                raise ValueError(f"{split} prompt audit failed: {audit}")
            path = STEP_DIR / f"{DATASET_ID}_{split}_qwen35_4b_nothinking_mixed_steps.jsonl"
            v7.write_jsonl(path, rows)
            step_rows[split] = rows
            step_paths[split] = path

        all_steps = [row for rows in step_rows.values() for row in rows]
        if len({row["prompt_sha256"] for row in all_steps}) != len(all_steps):
            raise ValueError("V8 formatted train/dev prompts are not unique")
        prompt_overlap = v7.prompt_repository_overlap(
            all_steps, {path.resolve() for path in step_paths.values()}
        )
        if prompt_overlap["overlap"]:
            raise ValueError(f"V8 prompts overlap repository history: {prompt_overlap}")

        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        audit_path = DATASET_DIR / "audit_report.json"
        v7.write_json(
            audit_path,
            {
                "schema_version": SCHEMA_VERSION,
                "created_at": CREATED_AT,
                "dataset_id": DATASET_ID,
                "case_validation": validation,
                "historical_protected_value_overlap": overlap,
                "repository_prompt_audit": prompt_overlap,
                "step_prompt_audits": {
                    split: v7.v6.prompt_audit(rows) for split, rows in step_rows.items()
                },
                "sealed_test_commitment": {
                    "cohorts": list(SEALED_SPLITS),
                    "entities": 24,
                    "rows": len(sealed_rows),
                    "sha256": test_commitment,
                    "materialized": bool(sealed_paths),
                },
            },
        )

        step_manifests: dict[str, Path] = {}
        for split, path in step_paths.items():
            manifest_path = path.with_suffix(".manifest.json")
            v7.write_json(
                manifest_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "created_at": CREATED_AT,
                    "dataset_id": DATASET_ID,
                    "role": SPLIT_SPECS[split]["role"],
                    "model_name_or_path": str(model_path),
                    "rows": len(step_rows[split]),
                    "allowed_step_indices": sorted({int(row["step_index"]) for row in step_rows[split]}),
                    "distribution": summarize_rows(step_rows[split]),
                    "files": {
                        "cases": str(case_paths[split].relative_to(ROOT)),
                        "step_data": str(path.relative_to(ROOT)),
                    },
                    "sha256": {
                        "cases": v7.sha256_file(case_paths[split]),
                        "step_data": v7.sha256_file(path),
                        "config": v7.sha256_file(model_path / "config.json"),
                        "tokenizer_config": v7.sha256_file(model_path / "tokenizer_config.json"),
                        "chat_template": v7.sha256_file(model_path / "chat_template.jinja"),
                    },
                },
            )
            step_manifests[split] = manifest_path

        manifest_path = DATASET_DIR / "manifest.json"
        files = {
            "builder": str(Path(__file__).resolve().relative_to(ROOT)),
            "preregistration": str((STUDY_DIR / "preregistration.json").relative_to(ROOT)),
            "comparison_contract": str((STUDY_DIR / "comparison_contract.json").relative_to(ROOT)),
            "pilot_decision": str((STUDY_DIR / "v7_pilot_decision.json").relative_to(ROOT)),
            "audit_report": str(audit_path.relative_to(ROOT)),
            **{f"cases_{split}": str(path.relative_to(ROOT)) for split, path in case_paths.items()},
            **{f"step_data_{split}": str(path.relative_to(ROOT)) for split, path in step_paths.items()},
            **{f"step_manifest_{split}": str(path.relative_to(ROOT)) for split, path in step_manifests.items()},
            **{f"fixture_{index + 1:02d}": str(path.relative_to(ROOT)) for index, path in enumerate(fixture_paths)},
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at": CREATED_AT,
            "dataset_id": DATASET_ID,
            "study_id": STUDY_ID,
            "status": "frozen_train_support_selection_test_committed",
            "seed": SEED,
            "independent_unit": "entity_id",
            "blocked_bundle": "entity × detector family",
            "primary_scenarios": list(PRIMARY_SCENARIOS),
            "stability_controls": list(CONTROL_SCENARIOS),
            "case_rows": {split: len(rows) for split, rows in cases_by_split.items()},
            "step_rows": {split: summarize_rows(rows) for split, rows in step_rows.items()},
            "sealed_test_commitment": {
                "cohorts": list(SEALED_SPLITS),
                "entities": 24,
                "rows": len(sealed_rows),
                "sha256": test_commitment,
                "materialized": bool(sealed_paths),
            },
            "integrity": {
                "case_validation": validation["status"],
                "historical_protected_value_overlap": overlap,
                "repository_prompt_overlap": prompt_overlap["overlap"],
                "independent_human_review": "pending",
            },
            "files": files,
            "sha256": {name: v7.sha256_file(ROOT / path) for name, path in files.items()},
        }
        v7.write_json(manifest_path, manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
