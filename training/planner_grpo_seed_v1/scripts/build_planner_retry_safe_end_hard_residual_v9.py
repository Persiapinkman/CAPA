#!/usr/bin/env python3
"""Build the preregistered V9 hard retry/safe-end residual dataset."""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.planner_grpo_seed_v1.scripts import (
    build_planner_retry_migrate_residual_v7 as v7,
)


DATASET_ID = "planner_retry_safe_end_hard_residual_v9"
STUDY_ID = "planner_retry_safe_end_hard_residual_v9_qwen35_4b_v1"
SCHEMA_VERSION = "1.0"
SEED = 2026071609
CREATED_AT = "2026-07-16T18:23:21Z"
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
MATERIALIZED_CASE_SPLITS = (
    "grpo_train",
    "support_dev_a",
    "support_dev_b",
    "selection_dev",
)
SEALED_SPLITS = ("sealed_test_a", "sealed_test_b")

SPLIT_SPECS: dict[str, dict[str, Any]] = {
    "grpo_train": {
        "entities": 24,
        "code": "HT",
        "training_only": True,
        "evaluation_only": False,
        "exclude_from_training": False,
        "sealed": False,
        "role": "optimizer_only",
    },
    "support_dev_a": {
        "entities": 12,
        "code": "SA",
        "training_only": False,
        "evaluation_only": True,
        "exclude_from_training": True,
        "sealed": False,
        "role": "support_gate_block_a",
    },
    "support_dev_b": {
        "entities": 12,
        "code": "SB",
        "training_only": False,
        "evaluation_only": True,
        "exclude_from_training": True,
        "sealed": False,
        "role": "support_gate_block_b",
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
        "role": "sealed_confirmation_a",
    },
    "sealed_test_b": {
        "entities": 12,
        "code": "TB",
        "training_only": False,
        "evaluation_only": True,
        "exclude_from_training": True,
        "sealed": True,
        "role": "sealed_confirmation_b",
    },
}

LEXICONS: dict[str, dict[str, tuple[str, ...]]] = {
    "grpo_train": {
        "roots": (
            "海峡潮流测绘舱",
            "精密铸造冷却廊",
            "高压直流换流阀厅",
            "空间望远镜热控台",
            "医药冻干灌装岛",
            "地下储气井口站",
        ),
        "suffixes": ("异常预算会核", "失联回执终判", "迁移前审签", "探针恢复定级"),
        "styles": (
            "读取当前故障回执并完成处置终审",
            "请在失联告警背景下核销本轮探针预算",
            "迁移评审已经排队，先按最后一条结构化回执选路",
            "故障复盘混有旧预算记录，本轮仍需依据当前键值裁决",
        ),
    },
    "support_dev_a": {
        "roots": ("极低温氦循环舱", "林火瞭望中继塔", "自动码头岸桥台", "食品无菌配液间"),
        "suffixes": ("故障路径会审", "重试额度复签", "恢复边界核验"),
        "styles": (
            "读取这次异常回执后完成最终处置",
            "失联告警仍在标题中，请核对当前预算字段",
            "迁移候选已经登记，先按最新结构化状态继续",
            "旧故障摘要可能过期，只裁决本轮显式键值",
        ),
    },
    "support_dev_b": {
        "roots": ("深空天线伺服间", "海水淡化高压泵站", "柔性屏封装检测线", "山区索道驱动舱"),
        "suffixes": ("异常链路复议", "探针预算核销", "迁移前置终审"),
        "styles": (
            "完成当前故障状态的处置会签",
            "请在设备失联语境中复核本轮重试资格",
            "迁移工单已创建，仍先读取最新机器回执",
            "备注中的预算结论不是当前状态，请按结构化字段判定",
        ),
    },
    "selection_dev": {
        "roots": ("火箭推进剂加注坪", "智慧水库闸控间", "粒子束准直机房", "冷轧带钢测厚台"),
        "suffixes": ("失联处置复核", "预算边界终签", "探针回执定级"),
        "styles": (
            "读取最新异常回执并完成动作终审",
            "失联告警只作语境，请核验当前重试计数",
            "迁移复核已经排期，先依据最后一条状态选择动作",
            "历史预算记录与当前字段并存时，以结构化键值裁决",
        ),
    },
    "sealed_test_a": {
        "roots": ("月球通信导航站", "海底观测缆分接舱", "特高压套管试验厅", "细胞培养灌流平台"),
        "suffixes": ("异常路由终验", "重试资格复审", "迁移条件会核"),
        "styles": (
            "请完成当前异常探针的最终路由",
            "设备失联写在工单中，仍需核销本轮显式预算",
            "迁移队列已经建立，先按最新机器状态裁决",
            "旧故障结论不得覆盖当前结构化回执",
        ),
    },
    "sealed_test_b": {
        "roots": ("高原射电阵列站", "城市供热调峰舱", "复合材料铺放台", "农田无人机补给坪"),
        "suffixes": ("回执边界会签", "异常预算终核", "能力恢复复议"),
        "styles": (
            "读取本轮故障状态并给出唯一处置",
            "失联背景与最新预算同时出现时，请按显式字段继续",
            "迁移评估已候场，先核验当前结构化回执",
            "历史备注可能误导，本轮只依据机器键值裁决",
        ),
    },
}

HARD_POLICY = (
    "自然语言里的故障等级和预算描述没有路由优先级，只读取本 query 最新 observation 的结构化键值。",
    "错误状态仅在 retryable=true 与 retry_count=0 同时成立时复用原检测器一次，否则调用迁移顾问。",
    "正常状态需要候选数、置信度、跨提示一致性和域偏移四个门槛全部通过才结束，任一否决项都转迁移。",
    "缺省、非法、冲突状态采取保守分支；已经重试一次的结果不能再次触发检测器。",
)
POLICY_WORDING = {split: HARD_POLICY for split in SPLIT_SPECS}


def _aliases(prefix: str) -> dict[str, tuple[str, str]]:
    return {
        "timeout": (f"{prefix}_sensor_reply_epoch_late", f"{prefix}_probe_ack_window_closed"),
        "transport": (f"{prefix}_feature_return_path_lost", f"{prefix}_detector_control_route_broken"),
        "quota": (f"{prefix}_inspection_replay_credit_empty", f"{prefix}_vision_capacity_lease_absent"),
        "payload": (f"{prefix}_probe_state_packet_invalid", f"{prefix}_detector_contract_shape_unknown"),
    }


ERROR_ALIASES = {
    "grpo_train": _aliases("hardtrain"),
    "support_dev_a": _aliases("supportalpha"),
    "support_dev_b": _aliases("supportbravo"),
    "selection_dev": _aliases("selectiondelta"),
    "sealed_test_a": _aliases("sealedalpha"),
    "sealed_test_b": _aliases("sealedbravo"),
}

FIXTURES: dict[str, tuple[dict[str, Any], ...]] = {
    "grpo_train": (
        {"target": "孔雀蓝双环校准件", "slug": "peacock_double_ring_calibrator", "family": "prhv9_ht_peacock_rings", "shape": "rings", "fg": (24, 128, 154), "bg": (233, 226, 210)},
        {"target": "琥珀色五星锁定片", "slug": "amber_five_star_lock_tab", "family": "prhv9_ht_amber_star", "shape": "star", "fg": (206, 139, 28), "bg": (37, 46, 57)},
    ),
    "support_dev_a": (
        {"target": "石榴红十字限位块", "slug": "garnet_cross_limit_block", "family": "prhv9_sa_garnet_cross", "shape": "cross", "fg": (166, 45, 62), "bg": (231, 225, 212)},
        {"target": "天青三辐调节轮", "slug": "cerulean_three_spoke_wheel", "family": "prhv9_sa_cerulean_wheel", "shape": "wheel", "fg": (38, 143, 183), "bg": (40, 47, 58)},
    ),
    "support_dev_b": (
        {"target": "黄铜弧顶隔振座", "slug": "brass_arch_isolator", "family": "prhv9_sb_brass_arch", "shape": "arch", "fg": (183, 132, 58), "bg": (232, 227, 215)},
        {"target": "群青六角巡检牌", "slug": "ultramarine_hex_inspection_tag", "family": "prhv9_sb_ultramarine_hex", "shape": "hexagon", "fg": (50, 73, 170), "bg": (39, 46, 56)},
    ),
    "selection_dev": (
        {"target": "松石绿双环耦合件", "slug": "turquoise_double_ring_coupler", "family": "prhv9_vd_turquoise_rings", "shape": "rings", "fg": (43, 151, 132), "bg": (38, 46, 56)},
        {"target": "樱桃红五星复核片", "slug": "cherry_five_star_review_tab", "family": "prhv9_vd_cherry_star", "shape": "star", "fg": (194, 49, 73), "bg": (231, 225, 213)},
    ),
    "sealed_test_a": (
        {"target": "紫铜十字定位块", "slug": "copper_cross_locator", "family": "prhv9_ta_copper_cross", "shape": "cross", "fg": (173, 88, 54), "bg": (37, 45, 55)},
        {"target": "藤紫三辐手轮", "slug": "wisteria_three_spoke_handwheel", "family": "prhv9_ta_wisteria_wheel", "shape": "wheel", "fg": (136, 89, 172), "bg": (232, 227, 214)},
    ),
    "sealed_test_b": (
        {"target": "铂灰弧顶隔离座", "slug": "platinum_arch_isolator", "family": "prhv9_tb_platinum_arch", "shape": "arch", "fg": (173, 177, 181), "bg": (38, 46, 56)},
        {"target": "珊瑚橙六角校验片", "slug": "coral_orange_hex_check_tab", "family": "prhv9_tb_coral_hex", "shape": "hexagon", "fg": (217, 103, 67), "bg": (232, 226, 213)},
    ),
}


def v9_entity_id(split: str, entity_index: int) -> str:
    return f"prhv9_{SPLIT_SPECS[split]['code'].lower()}_entity_{entity_index + 1:03d}"


def factor_layout(split: str) -> list[dict[str, int]]:
    styles = range(len(LEXICONS[split]["styles"]))
    badges = range(len(v7.BADGES))
    if split == "grpo_train":
        combinations = list(itertools.product(styles, badges, range(2)))
    else:
        offset = int(split in {"support_dev_b", "sealed_test_b"})
        combinations = [
            (style, badge, (style + badge + offset) % 2)
            for style, badge in itertools.product(styles, badges)
        ]
    random.Random(SEED + 211 + sum(map(ord, split))).shuffle(combinations)
    return [
        {"style_index": style, "badge_index": badge, "fixture_index": fixture}
        for style, badge, fixture in combinations
    ]


@contextlib.contextmanager
def configured_v7() -> Iterator[None]:
    names = (
        "DATASET_ID", "SCHEMA_VERSION", "SEED", "CREATED_AT", "STUDY_ID",
        "DATASET_DIR", "FIXTURE_DIR", "STUDY_DIR", "PRIMARY_SCENARIOS",
        "CONTROL_SCENARIOS", "ALL_SCENARIOS", "SPLIT_SPECS", "LEXICONS",
        "POLICY_WORDING", "ERROR_ALIASES", "FIXTURES", "PROJECTS",
        "FACTOR_LAYOUTS", "ALIAS_LAYOUTS", "entity_id", "base_case",
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
        v7.entity_id = v9_entity_id
        v7.PROJECTS = {split: v7.build_projects(split) for split in SPLIT_SPECS}
        v7.FACTOR_LAYOUTS = {split: factor_layout(split) for split in SPLIT_SPECS}
        v7.ALIAS_LAYOUTS = {split: v7.alias_layout(split) for split in SPLIT_SPECS}

        def base_case_v9(**kwargs: Any) -> dict[str, Any]:
            row = original_base_case(**kwargs)
            row["case_id"] = str(row["case_id"]).replace("PRRV7-", "PRHV9-", 1)
            row["template_id"] = str(row["template_id"]).replace("prrv7_", "prhv9_", 1)
            row["grpo_eligible"] = (
                row["split"] == "grpo_train" and row["scenario_id"] in PRIMARY_SCENARIOS
            )
            row["difficulty_family"] = "v7_anchored_hard_nuisance"
            row["support_block"] = (
                "A" if row["split"] == "support_dev_a"
                else "B" if row["split"] == "support_dev_b"
                else ""
            )
            row["provenance_class"] = "independent_synthetic_hard_residual_factorial_v9"
            return row

        v7.base_case = base_case_v9
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
        historical_rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
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
        "support_blocks": dict(sorted(Counter(str(row.get("support_block") or "none") for row in rows).items())),
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
    if args.materialize_test and args.confirm_materialize_test != "OPEN_V9_TEST":
        raise PermissionError("test materialization requires --confirm-materialize-test OPEN_V9_TEST")
    model_path = args.model_name_or_path.resolve()
    with configured_v7():
        fixture_paths = v7.write_fixture_images()
        cases_by_split = v7.build_all_cases()
        validation = v7.validate_cases(cases_by_split)
        overlap = historical_overlap(cases_by_split)
        if any(overlap.values()):
            validation["errors"].append(f"V9 protected values overlap prior retry datasets: {overlap}")
            validation["status"] = "fail"
        if validation["status"] != "pass":
            raise ValueError("V9 case validation failed:\n" + "\n".join(validation["errors"][:100]))

        case_paths: dict[str, Path] = {}
        for split in MATERIALIZED_CASE_SPLITS:
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
            combined = STUDY_DIR / "sealed_test_data/combined_test_cases.jsonl"
            v7.write_jsonl(combined, sealed_rows)
            sealed_paths["combined"] = combined
            if v7.sha256_file(combined) != test_commitment:
                raise ValueError("materialized V9 test does not match the frozen commitment")

        tokenizer = v7.load_tokenizer(model_path)
        output_sources = {
            "grpo_train": ["grpo_train"],
            "support_dev": ["support_dev_a", "support_dev_b"],
            "selection_dev": ["selection_dev"],
        }
        step_rows: dict[str, list[dict[str, Any]]] = {}
        step_paths: dict[str, Path] = {}
        for output_name, source_splits in output_sources.items():
            source = [row for split in source_splits for row in cases_by_split[split]]
            if output_name in {"grpo_train", "support_dev"}:
                source = [row for row in source if row["scenario_id"] in PRIMARY_SCENARIOS]
            case_lookup = {str(row["case_id"]): row for row in source}
            rows = v7.build_step_rows(source, tokenizer)
            for row in rows:
                case = case_lookup[str(row["case_id"])]
                row["difficulty_family"] = str(case["difficulty_family"])
                row["support_block"] = str(case["support_block"])
            audit = v7.v6.prompt_audit(rows)
            if audit["status"] != "pass":
                raise ValueError(f"{output_name} prompt audit failed: {audit}")
            path = STEP_DIR / f"{DATASET_ID}_{output_name}_qwen35_4b_nothinking_mixed_steps.jsonl"
            v7.write_jsonl(path, rows)
            step_rows[output_name] = rows
            step_paths[output_name] = path

        all_steps = [row for rows in step_rows.values() for row in rows]
        if len({row["prompt_sha256"] for row in all_steps}) != len(all_steps):
            raise ValueError("V9 formatted train/dev prompts are not unique")
        prompt_overlap = v7.prompt_repository_overlap(
            all_steps, {path.resolve() for path in step_paths.values()}
        )
        if prompt_overlap["overlap"]:
            raise ValueError(f"V9 prompts overlap repository history: {prompt_overlap}")

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
                    name: v7.v6.prompt_audit(rows) for name, rows in step_rows.items()
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
        for name, path in step_paths.items():
            manifest_path = path.with_suffix(".manifest.json")
            source_splits = output_sources[name]
            v7.write_json(
                manifest_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "created_at": CREATED_AT,
                    "dataset_id": DATASET_ID,
                    "role": "support_gate_only" if name == "support_dev" else SPLIT_SPECS[source_splits[0]]["role"],
                    "source_splits": source_splits,
                    "model_name_or_path": str(model_path),
                    "rows": len(step_rows[name]),
                    "allowed_step_indices": sorted({int(row["step_index"]) for row in step_rows[name]}),
                    "distribution": summarize_rows(step_rows[name]),
                    "files": {
                        "cases": [str(case_paths[split].relative_to(ROOT)) for split in source_splits],
                        "step_data": str(path.relative_to(ROOT)),
                    },
                    "sha256": {
                        "cases": {split: v7.sha256_file(case_paths[split]) for split in source_splits},
                        "step_data": v7.sha256_file(path),
                        "config": v7.sha256_file(model_path / "config.json"),
                        "tokenizer_config": v7.sha256_file(model_path / "tokenizer_config.json"),
                        "chat_template": v7.sha256_file(model_path / "chat_template.jinja"),
                    },
                },
            )
            step_manifests[name] = manifest_path

        manifest_path = DATASET_DIR / "manifest.json"
        files = {
            "builder": str(Path(__file__).resolve().relative_to(ROOT)),
            "preregistration": str((STUDY_DIR / "preregistration.json").relative_to(ROOT)),
            "comparison_contract": str((STUDY_DIR / "comparison_contract.json").relative_to(ROOT)),
            "temperature_decision": str((STUDY_DIR / "temperature_calibration_decision.json").relative_to(ROOT)),
            "audit_report": str(audit_path.relative_to(ROOT)),
            **{f"cases_{split}": str(path.relative_to(ROOT)) for split, path in case_paths.items()},
            **{f"step_data_{name}": str(path.relative_to(ROOT)) for name, path in step_paths.items()},
            **{f"step_manifest_{name}": str(path.relative_to(ROOT)) for name, path in step_manifests.items()},
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
            "difficulty_family": "v7_anchored_hard_nuisance",
            "primary_scenarios": list(PRIMARY_SCENARIOS),
            "stability_controls": list(CONTROL_SCENARIOS),
            "case_rows": {split: len(rows) for split, rows in cases_by_split.items()},
            "step_rows": {name: summarize_rows(rows) for name, rows in step_rows.items()},
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
            },
            "files": files,
            "sha256": {name: v7.sha256_file(ROOT / path) for name, path in files.items()},
        }
        v7.write_json(manifest_path, manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
