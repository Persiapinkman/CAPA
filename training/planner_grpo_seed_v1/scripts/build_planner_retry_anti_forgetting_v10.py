#!/usr/bin/env python3
"""Build the preregistered V10 control-replay anti-forgetting study."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Iterator

from training.planner_grpo_seed_v1.scripts import (
    build_planner_retry_safe_end_hard_residual_v9 as base,
)


ROOT = Path(__file__).resolve().parents[3]
DATASET_ID = "planner_retry_anti_forgetting_v10"
STUDY_ID = "planner_retry_anti_forgetting_v10_qwen35_4b_v1"
SEED = 2026071610
CREATED_AT = "2026-07-16T22:20:00Z"

LEXICONS: dict[str, dict[str, tuple[str, ...]]] = {
    "grpo_train": {
        "roots": (
            "海上风机变桨试验舱",
            "量子磁强计屏蔽室",
            "高温合金粉末制备塔",
            "生物反应器补料平台",
            "隧道掘进姿态控制台",
            "卫星激光测距终端",
        ),
        "suffixes": ("控制回执联审", "探针预算复核", "异常分支终签", "恢复状态会核"),
        "styles": (
            "请读取本轮机器回执并完成唯一动作裁决",
            "标题保留旧故障描述，仍以最新结构化字段核销预算",
            "迁移工单正在候场，请先判断当前探针是否允许继续",
            "历史备注与当前键值冲突时，只按最后一条 observation 选路",
        ),
    },
    "support_dev_a": {
        "roots": ("极地冰芯扫描舱", "港口氢能加注岛", "精密光刻对准台", "山洪预警遥测站"),
        "suffixes": ("回执路径复审", "重试资格会签", "稳定边界核验"),
        "styles": (
            "依据最新异常回执完成处置复审",
            "工单仍写着设备失联，请核验当前显式计数",
            "迁移候选已经登记，先裁决本轮检测器状态",
            "旧预算摘要不具优先级，只读取机器键值",
        ),
    },
    "support_dev_b": {
        "roots": ("深海采样绞车间", "城市轨交信号柜", "柔性电池封装线", "高原气象雷达站"),
        "suffixes": ("异常预算终核", "探针路径复议", "迁移条件联签"),
        "styles": (
            "完成本轮结构化状态的最终选路",
            "失联语境不能代替当前 retry 字段，请逐键裁决",
            "能力迁移已排队，仍先检查最后一次机器回执",
            "备注可能来自上一轮，本轮仅使用显式状态",
        ),
    },
    "selection_dev": {
        "roots": ("聚变装置真空泵厅", "智慧农场灌溉枢纽", "航空发动机叶片台", "冷链仓储分拣岛"),
        "suffixes": ("故障回执定级", "额度边界终审", "恢复路径复签"),
        "styles": (
            "读取当前机器回执后给出唯一处置动作",
            "告警标题仅提供背景，请按本轮重试键值继续",
            "迁移评估已预约，先判断现有探针是否恢复",
            "历史结论与最新字段并存时，以结构化状态为准",
        ),
    },
    "sealed_test_a": {
        "roots": ("太阳能热发电塔", "海洋牧场监测浮台", "超导电缆试验廊", "药物晶型筛选站"),
        "suffixes": ("异常路由终验", "预算资格复核", "恢复状态会审"),
        "styles": (
            "完成这次异常探针的最终路由",
            "设备失联写在背景中，仍需核对本轮显式额度",
            "迁移队列已经创建，请先依据最新机器状态裁决",
            "过期故障结论不得覆盖当前结构化回执",
        ),
    },
    "sealed_test_b": {
        "roots": ("空间碎片监测阵列", "区域供水调蓄泵站", "碳纤维预浸料台", "林业无人机补给舱"),
        "suffixes": ("回执分支会签", "异常额度终核", "探针恢复复议"),
        "styles": (
            "读取本轮机器状态并选择唯一动作",
            "失联背景与当前预算同时出现时，只按显式字段处理",
            "能力迁移已候场，先核验最新探针回执",
            "历史备注可能误导，本轮以最后一条 observation 为准",
        ),
    },
}

FIXTURES = {
    "grpo_train": (
        {"target": "靛蓝双环稳相件", "slug": "indigo_double_ring_phase_tab", "family": "prafv10_ht_indigo_rings", "shape": "rings", "fg": (43, 77, 161), "bg": (232, 226, 212)},
        {"target": "金橙五星校准片", "slug": "gold_orange_five_star_calibrator", "family": "prafv10_ht_gold_star", "shape": "star", "fg": (221, 142, 37), "bg": (38, 46, 57)},
    ),
    "support_dev_a": (
        {"target": "莓红十字锁止块", "slug": "berry_cross_lock_block", "family": "prafv10_sa_berry_cross", "shape": "cross", "fg": (177, 49, 87), "bg": (233, 227, 214)},
        {"target": "湖蓝三辐微调轮", "slug": "lake_blue_three_spoke_wheel", "family": "prafv10_sa_lake_wheel", "shape": "wheel", "fg": (40, 137, 181), "bg": (39, 47, 58)},
    ),
    "support_dev_b": (
        {"target": "古铜弧顶缓冲座", "slug": "bronze_arch_buffer", "family": "prafv10_sb_bronze_arch", "shape": "arch", "fg": (167, 111, 63), "bg": (232, 226, 213)},
        {"target": "钴蓝六角复验牌", "slug": "cobalt_hex_recheck_tag", "family": "prafv10_sb_cobalt_hex", "shape": "hexagon", "fg": (49, 79, 176), "bg": (38, 46, 57)},
    ),
    "selection_dev": (
        {"target": "翡翠绿双环联接件", "slug": "emerald_double_ring_link", "family": "prafv10_vd_emerald_rings", "shape": "rings", "fg": (42, 149, 110), "bg": (38, 46, 57)},
        {"target": "朱砂红五星审签片", "slug": "vermilion_five_star_signoff", "family": "prafv10_vd_vermilion_star", "shape": "star", "fg": (196, 61, 55), "bg": (232, 226, 213)},
    ),
    "sealed_test_a": (
        {"target": "赤铜十字导向块", "slug": "red_copper_cross_guide", "family": "prafv10_ta_copper_cross", "shape": "cross", "fg": (181, 82, 49), "bg": (38, 46, 57)},
        {"target": "鸢尾紫三辐手轮", "slug": "iris_three_spoke_handwheel", "family": "prafv10_ta_iris_wheel", "shape": "wheel", "fg": (121, 91, 178), "bg": (233, 227, 214)},
    ),
    "sealed_test_b": (
        {"target": "钛灰弧顶隔振座", "slug": "titanium_arch_isolator", "family": "prafv10_tb_titanium_arch", "shape": "arch", "fg": (158, 166, 174), "bg": (38, 46, 57)},
        {"target": "蜜橘六角核验片", "slug": "tangerine_hex_verification_tab", "family": "prafv10_tb_tangerine_hex", "shape": "hexagon", "fg": (224, 118, 55), "bg": (232, 226, 213)},
    ),
}


def _settings() -> dict[str, object]:
    return {
        "DATASET_ID": DATASET_ID,
        "STUDY_ID": STUDY_ID,
        "SEED": SEED,
        "CREATED_AT": CREATED_AT,
        "DATASET_DIR": ROOT / "data/datasets" / DATASET_ID,
        "FIXTURE_DIR": ROOT / "examples/images" / DATASET_ID,
        "STUDY_DIR": ROOT / "experiments/studies" / STUDY_ID,
        "OPTIMIZER_SCENARIOS": base.ALL_SCENARIOS,
        "SUPPORT_SCENARIOS": base.ALL_SCENARIOS,
        "VERSION_LABEL": "V10",
        "CASE_ID_PREFIX": "PRAFV10",
        "TEMPLATE_ID_PREFIX": "prafv10",
        "ENTITY_ID_PREFIX": "prafv10",
        "DIFFICULTY_FAMILY": "v7_anchored_hard_nuisance_control_replay",
        "PROVENANCE_CLASS": "independent_synthetic_anti_forgetting_factorial_v10",
        "TEST_CONFIRMATION": "OPEN_V10_TEST",
        "BUILDER_PATH": Path(__file__).resolve(),
        "LEXICONS": LEXICONS,
        "POLICY_WORDING": {split: base.HARD_POLICY for split in base.SPLIT_SPECS},
        "ERROR_ALIASES": {
            "grpo_train": base._aliases("v10hardtrain"),
            "support_dev_a": base._aliases("v10supportalpha"),
            "support_dev_b": base._aliases("v10supportbravo"),
            "selection_dev": base._aliases("v10selectiondelta"),
            "sealed_test_a": base._aliases("v10sealedalpha"),
            "sealed_test_b": base._aliases("v10sealedbravo"),
        },
        "FIXTURES": FIXTURES,
    }


@contextlib.contextmanager
def configured_base() -> Iterator[None]:
    settings = _settings()
    saved = {name: getattr(base, name) for name in settings}
    try:
        for name, value in settings.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(base, name, value)


def build_cases_in_memory() -> dict[str, list[dict[str, object]]]:
    with configured_base():
        return base.build_cases_in_memory()


if __name__ == "__main__":
    with configured_base():
        base.main()
