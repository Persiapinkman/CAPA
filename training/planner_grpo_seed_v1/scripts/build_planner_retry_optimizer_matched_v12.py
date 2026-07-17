#!/usr/bin/env python3
"""Build the preregistered V12 optimizer-matched safety study."""

from __future__ import annotations

import contextlib
import copy
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.planner_grpo_seed_v1.scripts import (
    build_planner_retry_safe_end_hard_residual_v9 as base,
)


DATASET_ID = "planner_retry_optimizer_matched_v12"
STUDY_ID = "planner_retry_optimizer_matched_v12_qwen35_4b_v1"
SEED = 2026071712
CREATED_AT = "2026-07-17T03:30:00Z"

LEXICONS: dict[str, dict[str, tuple[str, ...]]] = {
    "grpo_train": {
        "roots": (
            "深海中微子标定舱",
            "高原电解制氢槽",
            "低温超导储能柜",
            "无人矿卡换电站",
            "柔性电子印刷线",
            "跨海桥梁健康监测箱",
        ),
        "suffixes": ("机器回执终核", "探针边界联签", "迁移条件复议", "恢复动作会审"),
        "styles": (
            "请依据最新结构化回执裁决唯一动作",
            "故障标题只是历史背景，本轮以显式机器字段为准",
            "迁移资源已经预留，仍须先判断当前探针状态",
            "旧结论若与最后 observation 冲突，只采用后者",
        ),
    },
    "support_dev_a": {
        "roots": ("极地冰芯分析舱", "波浪能液压试验台", "光钟频率传递柜", "无菌制剂冻干线"),
        "suffixes": ("异常状态复签", "重试资格终核", "安全分支联审"),
        "styles": (
            "读取本轮机器回执并完成动作复核",
            "失联背景不能覆盖当前 retry 与 count 字段",
            "迁移候选已登记，请先核验最新探针是否恢复",
            "只按最后一条结构化 observation 选择动作",
        ),
    },
    "support_dev_b": {
        "roots": ("日冕望远镜热控舱", "园区蒸汽换热站", "生物芯片键合台", "荒漠光伏逆变柜"),
        "suffixes": ("探针回执复议", "迁移边界终核", "恢复路径会签"),
        "styles": (
            "依据当前结构化状态给出唯一选路",
            "历史告警等级没有优先权，请逐键核验",
            "能力迁移正在排队，先检查最新探针回执",
            "过期预算摘要不得替代本轮机器字段",
        ),
    },
    "selection_dev": {
        "roots": ("同步辐射成像台", "自动化集装箱岸桥", "热塑复材成型炉", "高山气象标定站"),
        "suffixes": ("回执动作定级", "额度边界终审", "恢复分支复核"),
        "styles": (
            "读取当前机器回执后执行唯一动作",
            "告警标题保留旧描述，本轮只看显式计数",
            "迁移评估虽已预约，仍先判断探针恢复状态",
            "历史备注与最新字段并存时以后者为准",
        ),
    },
    "sealed_test_a": {
        "roots": ("火星车悬架试验间", "深水码头输油站", "非晶合金铸造炉", "类器官灌流培养台"),
        "suffixes": ("异常路径终验", "重试边界复审", "稳定状态会签"),
        "styles": (
            "完成本轮异常探针的最终路由",
            "设备失联只是背景，仍须核对当前显式额度",
            "迁移队列已经建立，请先裁决最新机器状态",
            "旧故障结论不得覆盖最后一条结构化回执",
        ),
    },
    "sealed_test_b": {
        "roots": ("甚低频射电阵列", "流域梯级调度闸站", "碳化硅陶瓷烧结台", "林区无人机补给舱"),
        "suffixes": ("回执分支终签", "异常额度会核", "探针恢复复议"),
        "styles": (
            "读取这次机器状态并选择唯一动作",
            "失联背景和当前预算并存时只使用显式字段",
            "能力迁移已候场，先核验最新探针回执",
            "历史文字可能误导，本轮以后写入的 observation 为准",
        ),
    },
}

FIXTURES = {
    "grpo_train": (
        {"target": "雾蓝双环止动件", "slug": "mistblue_double_ring_stop", "family": "promv12_ht_mistblue_rings", "shape": "rings", "fg": (145, 178, 196), "bg": (38, 47, 59)},
        {"target": "赤铜六角量规", "slug": "redcopper_hex_gauge", "family": "promv12_ht_redcopper_hex", "shape": "hexagon", "fg": (184, 92, 54), "bg": (233, 227, 214)},
    ),
    "support_dev_a": (
        {"target": "黛青十字导向片", "slug": "darkcyan_cross_guide", "family": "promv12_sa_darkcyan_cross", "shape": "cross", "fg": (42, 91, 103), "bg": (233, 227, 214)},
        {"target": "绯红三辐校准轮", "slug": "crimson_three_spoke_calibrator", "family": "promv12_sa_crimson_wheel", "shape": "wheel", "fg": (190, 62, 70), "bg": (38, 47, 59)},
    ),
    "support_dev_b": (
        {"target": "铂灰弧顶支撑座", "slug": "platinum_arch_support", "family": "promv12_sb_platinum_arch", "shape": "arch", "fg": (182, 187, 192), "bg": (38, 47, 59)},
        {"target": "湖蓝五星基准片", "slug": "lakeblue_five_star_reference", "family": "promv12_sb_lakeblue_star", "shape": "star", "fg": (45, 135, 176), "bg": (233, 227, 214)},
    ),
    "selection_dev": (
        {"target": "松石绿双环连接件", "slug": "turquoisegreen_double_ring_link", "family": "promv12_vd_turquoisegreen_rings", "shape": "rings", "fg": (47, 151, 133), "bg": (38, 47, 59)},
        {"target": "丁香紫六角验收牌", "slug": "lilac_hex_acceptance_tag", "family": "promv12_vd_lilac_hex", "shape": "hexagon", "fg": (145, 102, 174), "bg": (233, 227, 214)},
    ),
    "sealed_test_a": (
        {"target": "焦糖十字分流块", "slug": "caramel_cross_diverter", "family": "promv12_ta_caramel_cross", "shape": "cross", "fg": (174, 105, 58), "bg": (38, 47, 59)},
        {"target": "冰青三辐锁紧轮", "slug": "icecyan_three_spoke_lockwheel", "family": "promv12_ta_icecyan_wheel", "shape": "wheel", "fg": (104, 177, 187), "bg": (233, 227, 214)},
    ),
    "sealed_test_b": (
        {"target": "玄铜弧顶缓冲座", "slug": "darkbronze_arch_buffer", "family": "promv12_tb_darkbronze_arch", "shape": "arch", "fg": (137, 112, 71), "bg": (38, 47, 59)},
        {"target": "金橙五星复检片", "slug": "goldorange_five_star_recheck", "family": "promv12_tb_goldorange_star", "shape": "star", "fg": (224, 155, 48), "bg": (233, 227, 214)},
    ),
}

PRIMARY_REPLAY_PREFIXES = (
    "独立安全复核甲，只使用同一轮结构化字段：",
    "独立安全复核乙，背景文字不改变动作边界：",
    "独立安全复核丙，按最后 observation 重新裁决：",
    "独立安全复核丁，忽略旧工单中的路由暗示：",
)
REPLAY_SPLITS = ("grpo_train", "support_dev_a", "support_dev_b")


def _settings() -> dict[str, object]:
    return {
        "DATASET_ID": DATASET_ID,
        "STUDY_ID": STUDY_ID,
        "SEED": SEED,
        "CREATED_AT": CREATED_AT,
        "DATASET_DIR": ROOT / "data/datasets" / DATASET_ID,
        "FIXTURE_DIR": ROOT / "examples/images" / DATASET_ID,
        "STUDY_DIR": ROOT / "experiments/studies" / STUDY_ID,
        "TEMPERATURE_DECISION_PATH": ROOT / "experiments/studies" / STUDY_ID / "temperature_decision.json",
        "OPTIMIZER_SCENARIOS": base.ALL_SCENARIOS,
        "SUPPORT_SCENARIOS": base.ALL_SCENARIOS,
        "VERSION_LABEL": "V12",
        "CASE_ID_PREFIX": "PROMV12",
        "TEMPLATE_ID_PREFIX": "promv12",
        "ENTITY_ID_PREFIX": "promv12",
        "DIFFICULTY_FAMILY": "v7_anchored_hard_nuisance_safety_balanced_optimizer_matched",
        "PROVENANCE_CLASS": "independent_synthetic_optimizer_matched_factorial_v12",
        "TEST_CONFIRMATION": "OPEN_V12_TEST",
        "BUILDER_PATH": Path(__file__).resolve(),
        "LEXICONS": LEXICONS,
        "POLICY_WORDING": {split: base.HARD_POLICY for split in base.SPLIT_SPECS},
        "ERROR_ALIASES": {
            "grpo_train": base._aliases("v12matchedtrain"),
            "support_dev_a": base._aliases("v12supportalpha"),
            "support_dev_b": base._aliases("v12supportbravo"),
            "selection_dev": base._aliases("v12selectiondelta"),
            "sealed_test_a": base._aliases("v12sealedalpha"),
            "sealed_test_b": base._aliases("v12sealedbravo"),
        },
        "FIXTURES": FIXTURES,
    }


def _expand_primary_replay(cases: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    expanded = {split: list(rows) for split, rows in cases.items()}
    primary = set(base.PRIMARY_SCENARIOS)
    for split in REPLAY_SPLITS:
        variants: list[dict[str, Any]] = []
        for row in expanded[split]:
            if str(row["scenario_id"]) not in primary:
                continue
            variant = copy.deepcopy(row)
            style = int(row["query_style_index"])
            variant["case_id"] = f"{row['case_id']}-PRB"
            variant["template_id"] = f"{row['template_id']}_primary_replay_b"
            variant["user_query"] = f"{PRIMARY_REPLAY_PREFIXES[style]}{row['user_query']}"
            variant["query_style_index"] = style + len(PRIMARY_REPLAY_PREFIXES)
            variant["primary_replay_variant"] = "B"
            variant["provenance_class"] = f"{row['provenance_class']}_primary_replay_b"
            variants.append(variant)
        expanded[split].extend(variants)
    return expanded


@contextlib.contextmanager
def configured_base() -> Iterator[None]:
    settings = _settings()
    saved = {name: getattr(base, name) for name in settings}
    original_build_all_cases = base.v7.build_all_cases
    original_validate_cases = base.v7.validate_cases

    def build_all_cases_v12() -> dict[str, list[dict[str, Any]]]:
        return _expand_primary_replay(original_build_all_cases())

    def validate_cases_v12(
        cases_by_split: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        base_cases = {
            split: [row for row in rows if row.get("primary_replay_variant") != "B"]
            for split, rows in cases_by_split.items()
        }
        result = original_validate_cases(base_cases)
        errors = list(result["errors"])
        expected_sizes = {
            "grpo_train": (576, 144, 48),
            "support_dev_a": (288, 72, 24),
            "support_dev_b": (288, 72, 24),
            "selection_dev": (216, 0, 0),
            "sealed_test_a": (216, 0, 0),
            "sealed_test_b": (216, 0, 0),
        }
        replay_report: dict[str, Any] = {}
        for split, rows in cases_by_split.items():
            expected_total, expected_replay, expected_per_primary = expected_sizes[split]
            variants = [row for row in rows if row.get("primary_replay_variant") == "B"]
            replay_scenarios = Counter(str(row["scenario_id"]) for row in variants)
            expected_counts = Counter(
                {scenario: expected_per_primary for scenario in base.PRIMARY_SCENARIOS}
            )
            if len(rows) != expected_total or len(variants) != expected_replay:
                errors.append(
                    f"V12 {split} total/replay mismatch: total={len(rows)}, replay={len(variants)}"
                )
            if replay_scenarios != expected_counts:
                errors.append(f"V12 {split} primary replay is unbalanced: {replay_scenarios}")
            if len({str(row["case_id"]) for row in rows}) != len(rows):
                errors.append(f"V12 {split} case IDs are not unique")
            if any(not bool(base.v7.v6.score_case(row)["passed"]) for row in variants):
                errors.append(f"V12 {split} replay canonical trajectory failed")
            result["factors"][split] = base.v7.factor_report(rows)
            replay_report[split] = {
                "rows": len(variants),
                "scenarios": dict(sorted(replay_scenarios.items())),
            }
        result["errors"] = errors
        result["status"] = "pass" if not errors else "fail"
        result["v12_primary_replay"] = replay_report
        return result

    try:
        for name, value in settings.items():
            setattr(base, name, value)
        base.v7.build_all_cases = build_all_cases_v12
        base.v7.validate_cases = validate_cases_v12
        yield
    finally:
        base.v7.build_all_cases = original_build_all_cases
        base.v7.validate_cases = original_validate_cases
        for name, value in saved.items():
            setattr(base, name, value)


def build_cases_in_memory() -> dict[str, list[dict[str, Any]]]:
    with configured_base():
        return base.build_cases_in_memory()


if __name__ == "__main__":
    with configured_base():
        base.main()
