#!/usr/bin/env python3
"""Build the preregistered V11 safety-weighted, action-balanced study."""

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


DATASET_ID = "planner_retry_safety_balanced_v11"
STUDY_ID = "planner_retry_safety_balanced_v11_qwen35_4b_v1"
SEED = 2026071711
CREATED_AT = "2026-07-17T02:06:15Z"

LEXICONS: dict[str, dict[str, tuple[str, ...]]] = {
    "grpo_train": {
        "roots": (
            "月面样品密封舱",
            "聚变装置冷却环",
            "智能织造张力台",
            "地下储氢监测井",
            "海上碳封存注入站",
            "高速磁浮供电柜",
        ),
        "suffixes": ("回执边界复核", "探针状态终签", "迁移资格联审", "恢复路径会核"),
        "styles": (
            "请按本轮结构化回执完成唯一动作裁决",
            "旧告警标题仅作背景，当前机器键值决定路由",
            "迁移工单已候场，仍须先判断现有探针状态",
            "历史结论与最后 observation 冲突时只使用后者",
        ),
    },
    "support_dev_a": {
        "roots": ("火星土壤模拟舱", "潮汐能叶轮试验台", "量子通信中继柜", "食品冷冻干燥线"),
        "suffixes": ("异常回执复签", "重试边界终核", "安全路由会审"),
        "styles": (
            "读取最新机器回执并完成处置复核",
            "失联描述不能覆盖当前 retry 与 count 字段",
            "迁移候选已登记，先核验本轮探针是否恢复",
            "只按最后一条结构化 observation 选择动作",
        ),
    },
    "support_dev_b": {
        "roots": ("深空天线馈源舱", "城市热网换热站", "微流控芯片封装台", "山地风电变流柜"),
        "suffixes": ("探针状态复议", "迁移条件终核", "恢复分支联签"),
        "styles": (
            "依据本轮结构化状态给出唯一选路",
            "背景故障等级没有优先权，请逐键核验",
            "能力迁移正在排队，先检查最新探针回执",
            "过期预算摘要不得替代当前机器字段",
        ),
    },
    "selection_dev": {
        "roots": ("中子散射样品台", "智慧港口岸桥柜", "航空复材固化炉", "冰川遥感标定站"),
        "suffixes": ("回执动作定级", "额度边界终审", "恢复路径复核"),
        "styles": (
            "读取当前机器回执后执行唯一动作",
            "告警标题保留旧描述，本轮只看显式计数",
            "迁移评估虽已预约，仍先判断探针恢复状态",
            "历史备注与最新字段并存时以后者为准",
        ),
    },
    "sealed_test_a": {
        "roots": ("月球车热控试验间", "海底电缆登陆站", "高熵合金熔炼炉", "细胞培养灌流台"),
        "suffixes": ("异常路径终验", "重试资格复审", "稳定状态会签"),
        "styles": (
            "完成本轮异常探针的最终路由",
            "设备失联只是背景，仍须核对当前显式额度",
            "迁移队列已经建立，请先裁决最新机器状态",
            "旧故障结论不得覆盖最后一条结构化回执",
        ),
    },
    "sealed_test_b": {
        "roots": ("近地轨道测控阵列", "区域水库调度闸站", "陶瓷基复材烧结台", "森林火情补给舱"),
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
        {"target": "月白双环隔离件", "slug": "moonwhite_double_ring_isolator", "family": "prsbv11_ht_moonwhite_rings", "shape": "rings", "fg": (205, 214, 221), "bg": (39, 47, 58)},
        {"target": "琥珀六角校验块", "slug": "amber_hex_validation_block", "family": "prsbv11_ht_amber_hex", "shape": "hexagon", "fg": (210, 137, 45), "bg": (232, 226, 213)},
    ),
    "support_dev_a": (
        {"target": "青黛十字限位片", "slug": "indigo_cross_limit_tab", "family": "prsbv11_sa_indigo_cross", "shape": "cross", "fg": (49, 72, 128), "bg": (232, 226, 213)},
        {"target": "珊瑚红三辐调节轮", "slug": "coral_three_spoke_adjuster", "family": "prsbv11_sa_coral_wheel", "shape": "wheel", "fg": (214, 88, 76), "bg": (39, 47, 58)},
    ),
    "support_dev_b": (
        {"target": "银灰弧顶承压座", "slug": "silver_arch_pressure_seat", "family": "prsbv11_sb_silver_arch", "shape": "arch", "fg": (171, 180, 188), "bg": (39, 47, 58)},
        {"target": "孔雀蓝五星定位片", "slug": "peacock_five_star_locator", "family": "prsbv11_sb_peacock_star", "shape": "star", "fg": (35, 126, 154), "bg": (232, 226, 213)},
    ),
    "selection_dev": (
        {"target": "松石绿双环耦合件", "slug": "turquoise_double_ring_coupler", "family": "prsbv11_vd_turquoise_rings", "shape": "rings", "fg": (47, 155, 139), "bg": (39, 47, 58)},
        {"target": "胭脂紫六角签核牌", "slug": "rouge_purple_hex_signoff", "family": "prsbv11_vd_rouge_hex", "shape": "hexagon", "fg": (154, 72, 137), "bg": (232, 226, 213)},
    ),
    "sealed_test_a": (
        {"target": "赭石十字导流块", "slug": "ochre_cross_flow_guide", "family": "prsbv11_ta_ochre_cross", "shape": "cross", "fg": (176, 101, 55), "bg": (39, 47, 58)},
        {"target": "霜蓝三辐手轮", "slug": "frostblue_three_spoke_handwheel", "family": "prsbv11_ta_frostblue_wheel", "shape": "wheel", "fg": (104, 160, 190), "bg": (232, 226, 213)},
    ),
    "sealed_test_b": (
        {"target": "墨金弧顶阻尼座", "slug": "inkgold_arch_damper", "family": "prsbv11_tb_inkgold_arch", "shape": "arch", "fg": (172, 139, 61), "bg": (39, 47, 58)},
        {"target": "柠黄五星复核片", "slug": "lemon_five_star_recheck", "family": "prsbv11_tb_lemon_star", "shape": "star", "fg": (224, 202, 54), "bg": (232, 226, 213)},
    ),
}

PRIMARY_REPLAY_PREFIXES = (
    "独立复核措辞甲，只使用同一轮结构化字段：",
    "独立复核措辞乙，背景文字不改变动作边界：",
    "独立复核措辞丙，按最后 observation 重新裁决：",
    "独立复核措辞丁，忽略旧工单中的路由暗示：",
)


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
        "VERSION_LABEL": "V11",
        "CASE_ID_PREFIX": "PRSBV11",
        "TEMPLATE_ID_PREFIX": "prsbv11",
        "ENTITY_ID_PREFIX": "prsbv11",
        "DIFFICULTY_FAMILY": "v7_anchored_hard_nuisance_safety_balanced",
        "PROVENANCE_CLASS": "independent_synthetic_safety_balanced_factorial_v11",
        "TEST_CONFIRMATION": "OPEN_V11_TEST",
        "BUILDER_PATH": Path(__file__).resolve(),
        "LEXICONS": LEXICONS,
        "POLICY_WORDING": {split: base.HARD_POLICY for split in base.SPLIT_SPECS},
        "ERROR_ALIASES": {
            "grpo_train": base._aliases("v11safetytrain"),
            "support_dev_a": base._aliases("v11supportalpha"),
            "support_dev_b": base._aliases("v11supportbravo"),
            "selection_dev": base._aliases("v11selectiondelta"),
            "sealed_test_a": base._aliases("v11sealedalpha"),
            "sealed_test_b": base._aliases("v11sealedbravo"),
        },
        "FIXTURES": FIXTURES,
    }


def _expand_primary_replay(cases: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    expanded = {split: list(rows) for split, rows in cases.items()}
    variants: list[dict[str, Any]] = []
    for row in expanded["grpo_train"]:
        if str(row["scenario_id"]) not in set(base.PRIMARY_SCENARIOS):
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
    expanded["grpo_train"].extend(variants)
    return expanded


@contextlib.contextmanager
def configured_base() -> Iterator[None]:
    settings = _settings()
    saved = {name: getattr(base, name) for name in settings}
    original_build_all_cases = base.v7.build_all_cases
    original_validate_cases = base.v7.validate_cases

    def build_all_cases_v11() -> dict[str, list[dict[str, Any]]]:
        return _expand_primary_replay(original_build_all_cases())

    def validate_cases_v11(
        cases_by_split: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        base_cases = {
            split: [row for row in rows if row.get("primary_replay_variant") != "B"]
            for split, rows in cases_by_split.items()
        }
        result = original_validate_cases(base_cases)
        errors = list(result["errors"])
        train = cases_by_split["grpo_train"]
        variants = [row for row in train if row.get("primary_replay_variant") == "B"]
        if len(train) != 576 or len(variants) != 144:
            errors.append(
                f"V11 train/replay size mismatch: train={len(train)}, replay={len(variants)}"
            )
        replay_scenarios = Counter(str(row["scenario_id"]) for row in variants)
        if replay_scenarios != Counter({scenario: 48 for scenario in base.PRIMARY_SCENARIOS}):
            errors.append(f"V11 primary replay is unbalanced: {replay_scenarios}")
        if len({str(row["case_id"]) for row in train}) != len(train):
            errors.append("V11 train case IDs are not unique")
        if any(not bool(base.v7.v6.score_case(row)["passed"]) for row in variants):
            errors.append("V11 replay canonical trajectory failed")
        result["errors"] = errors
        result["status"] = "pass" if not errors else "fail"
        result["factors"]["grpo_train"] = base.v7.factor_report(train)
        result["v11_primary_replay"] = {
            "rows": len(variants),
            "scenarios": dict(sorted(replay_scenarios.items())),
        }
        return result

    try:
        for name, value in settings.items():
            setattr(base, name, value)
        base.v7.build_all_cases = build_all_cases_v11
        base.v7.validate_cases = validate_cases_v11
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
