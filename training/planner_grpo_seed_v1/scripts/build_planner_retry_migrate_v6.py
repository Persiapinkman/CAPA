#!/usr/bin/env python3
"""Build entity-disjoint SFT, GRPO, dev, and sealed-test data for retry routing.

The experimental unit is an ``(entity, detector, error_alias)`` bundle.  Every
core bundle contains three matched states with the same query and nuisance
fields: fresh retryable error, non-retryable error, and exhausted retry budget.
This makes the decision depend on ``retryable`` and ``retry_count`` rather than
on entity identity, wording, error spelling, badge, fixture, or detector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[3]
for import_root in (ROOT / "src", ROOT, ROOT / "demo"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from training.planner_grpo_seed_v1.scripts import (  # noqa: E402
    build_planner_multistep_grpo_hard_v1 as v1,
)
from training.planner_grpo_seed_v1.scripts.reward_planner_grpo import (  # noqa: E402
    score_case,
)


DATASET_ID = "planner_retry_migrate_v6"
SCHEMA_VERSION = "2.0"
SEED = 2026071506
CREATED_AT = "2026-07-15T00:00:00+00:00"
CASE_DIR = ROOT / "training/planner_grpo_seed_v1/cases"
DATASET_DIR = ROOT / "data/datasets" / DATASET_ID
FIXTURE_DIR = ROOT / "examples/images" / DATASET_ID
SFT_DIR = ROOT / "training/planner_grpo_seed_v1/sft_data_planner_retry_migrate_v6_qwen35_nothinking"
STEP_DIR = ROOT / "training/planner_grpo_seed_v1/step_data"
DEFAULT_MODEL = Path("/raid/zkq/models/Qwen3.5-4B")
NONTHINKING_SUFFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
EXPECTED_EOS_ID = 248046
EXPECTED_PAD_ID = 248044
MAX_PROMPT_TOKENS = 4608

CORE_STATES = ("retryable_fresh", "nonretryable", "budget_exhausted")
GUARD_TYPES = (
    "initial_success_end",
    "initial_metric_veto_migrate",
    "missing_required_state_migrate",
    "conflicting_state_migrate",
    "stale_history_current_success_end",
    "stale_history_current_error_migrate",
)
DETECTORS = (
    ("qwen", "qwen_detection", "Qwen"),
    ("rex", "rexomni_detection", "Rex-Omni"),
)
BADGE_MODES = ("red", "amber", "missing")

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

ROLE_SPECS: dict[str, dict[str, Any]] = {
    "sft_train": {
        "entities": 80,
        "stage": "sft",
        "selection_role": "optimization_only_sft_train",
        "training_only": True,
        "evaluation_only": False,
        "exclude_from_training": False,
        "sealed": False,
        "code": "ST",
    },
    "sft_dev": {
        "entities": 20,
        "stage": "sft",
        "selection_role": "sft_validation_only",
        "training_only": False,
        "evaluation_only": True,
        "exclude_from_training": True,
        "sealed": False,
        "code": "SD",
    },
    "grpo_train": {
        "entities": 60,
        "stage": "grpo",
        "selection_role": "optimization_only_grpo_train",
        "training_only": True,
        "evaluation_only": False,
        "exclude_from_training": False,
        "sealed": False,
        "code": "GT",
    },
    "grpo_dev": {
        "entities": 30,
        "stage": "grpo",
        "selection_role": "grpo_validation_only",
        "training_only": False,
        "evaluation_only": True,
        "exclude_from_training": True,
        "sealed": False,
        "code": "GD",
    },
    "test": {
        "entities": 60,
        "stage": "evaluation",
        "selection_role": "sealed_final_evaluation",
        "training_only": False,
        "evaluation_only": True,
        "exclude_from_training": True,
        "sealed": True,
        "code": "TE",
    },
}

LEXICONS: dict[str, dict[str, tuple[str, ...]]] = {
    "sft_train": {
        "roots": (
            "潮间带储能舱",
            "地下管廊泵站",
            "制冷压缩机房",
            "港口堆取料区",
            "柔性装配工位",
            "山地输电巡线站",
            "化工园区卸料台",
            "航空货运分拨仓",
            "城市水务加压站",
            "轨交供电检修间",
        ),
        "suffixes": (
            "异常恢复核验",
            "边缘探针复核",
            "接口韧性审查",
            "灰度准入判定",
            "状态机验收",
            "能力边界盘点",
            "容错路径评审",
            "现场回执校核",
        ),
        "styles": (
            "请处理本轮现场恢复单",
            "依据当前探针回执完成判定",
            "按这次接口验收顺序推进",
            "请核验当前能力边界工单",
        ),
    },
    "sft_dev": {
        "roots": (
            "海上风机升压舱",
            "玻璃熔窑控制室",
            "智慧农机停放库",
            "山区索道驱动站",
            "电子洁净物料间",
        ),
        "suffixes": (
            "服务降级复盘",
            "探测链路验收",
            "现场恢复确认",
            "模型接入校验",
        ),
        "styles": (
            "请完成这次服务降级检查",
            "只按本次结构化回执继续",
            "依据当前接入单执行后续步骤",
            "请审阅这一轮探测链路状态",
        ),
    },
    "grpo_train": {
        "roots": (
            "深水码头引桥区",
            "锂盐结晶车间",
            "高寒公路养护站",
            "机场行李分拣岛",
            "水下机器人保障舱",
            "算力园区冷却廊",
            "粮油散装装卸区",
            "城市消防训练塔",
            "精密铸造清理间",
            "跨江隧道监控室",
        ),
        "suffixes": (
            "重试预算决策",
            "故障分支核查",
            "上线恢复评估",
            "探针状态审计",
            "迁移边界会签",
            "异常回路验收",
        ),
        "styles": (
            "处理当前故障分支会签单",
            "根据最新执行状态选择下一步",
            "请按本轮恢复预算完成路由",
            "本次只依据当前工具状态决策",
        ),
    },
    "grpo_dev": {
        "roots": (
            "沙漠光热镜场",
            "海关智能查验台",
            "低温种质保存舱",
            "船厂分段涂装房",
            "城市桥梁顶升区",
            "山区水库启闭室",
        ),
        "suffixes": (
            "通道恢复判断",
            "故障等级复核",
            "视觉回执验收",
            "现场能力会审",
            "异常预算核销",
        ),
        "styles": (
            "请判定本次通道恢复路径",
            "按最新视觉回执继续办理",
            "核对当前异常预算后作出选择",
            "请完成这一轮现场能力会审",
        ),
    },
    "test": {
        "roots": (
            "极地科考补给舱",
            "深井提升机硐室",
            "卫星总装测试厅",
            "沿海防潮闸站",
            "危废焚烧配伍间",
            "高铁道岔监测房",
            "海洋牧场控制台",
            "古建消防值守点",
            "无人矿卡调度区",
            "生鲜冷链预冷库",
        ),
        "suffixes": (
            "状态转移终验",
            "探针恢复定级",
            "边界条件复查",
            "故障处置裁决",
            "视觉链路终审",
            "异常状态确认",
        ),
        "styles": (
            "请完成当前终验状态裁决",
            "读取这次探针结果后决定后续",
            "本轮按边界条件清单逐项处理",
            "请对最新故障回执作最终判定",
        ),
    },
}

POLICY_WORDING: dict[str, tuple[str, ...]] = {
    "sft_train": (
        "只采用当前 query 最后一个 observation 的结构化值；归档记录、看板颜色与备注不参与路由。",
        "出现技术错误时，仅在 retryable=true 且 retry_count=0 时允许同一检测器再执行一次；不可重试或预算已用完则进入迁移顾问。",
        "没有技术错误时，candidate_count>=1、min_confidence>=0.88、cross_prompt_iou>=0.72 与 domain_shift=low 必须同时成立才能结束，否则进入迁移顾问。",
        "字段缺失、类型非法或彼此冲突均按不安全状态进入迁移顾问；重试后的回执仍按相同门槛判断且禁止第二次重试。",
    ),
    "sft_dev": (
        "判断时以这个 query 的最新结构化 observation 为准，旧批次结论、颜色标签和自由文本不能覆盖字段。",
        "若 gateway_error 非 none，只有 retryable=true 并且 retry_count=0 才能原工具再试一次；其余错误状态交给迁移顾问。",
        "若 gateway_error=none，只有候选数不少于1、最低置信度不少于0.88、跨提示 IoU 不少于0.72 且域偏移为 low 才收口，任一失败都交给迁移顾问。",
        "不完整、非法或矛盾的状态采取保守分支；一次重试之后不再重试，并重新检查同一组指标。",
    ),
    "grpo_train": (
        "路由只能读取当前问题最新一步的字段，历史轨迹、overall_badge 和注释都只是干扰项。",
        "gateway_error 不为 none 时，原检测器的唯一重试许可是 retryable=true 且 retry_count=0；否则调用迁移顾问。",
        "gateway_error 为 none 时需要 candidate_count>=1、min_confidence>=0.88、cross_prompt_iou>=0.72、domain_shift=low 四项全过才结束，有否决项就调用迁移顾问。",
        "缺字段、错误类型或逻辑冲突视为不满足安全门槛；重试只允许一次，第二份回执继续执行上述判断。",
    ),
    "grpo_dev": (
        "请忽略过往 query 的结论与展示性 badge，只让本 query 最后一次 observation 的显式字段驱动动作。",
        "技术错误下，仅 fresh retryable 状态，也就是 retryable=true 且 retry_count=0，可以复用原检测器一次；其他情况转迁移顾问。",
        "正常回执必须同时达到候选数1、置信度0.88、跨提示 IoU 0.72 和低域偏移四个下限才可以结束，否则转迁移顾问。",
        "任何缺失、非法、互相矛盾的字段都走保守路径；重试后只能结束或转迁移，不能形成循环。",
    ),
    "test": (
        "当前 query 的最后一条结构化 observation 是唯一裁决依据，历史结果、颜色标记及自然语言说明均无优先权。",
        "当 gateway_error 非 none，必须同时满足 retryable=true 与 retry_count=0 才能让相同检测器重做一次；否则请求迁移顾问。",
        "当 gateway_error=none，candidate_count 至少1、min_confidence 至少0.88、cross_prompt_iou 至少0.72、domain_shift 等于 low 四项缺一不可，未全过则请求迁移顾问。",
        "缺省值、类型异常或字段冲突按安全失败处理；已经重做一次后只评估新回执，不允许再次重做。",
    ),
}

ERROR_ALIASES: dict[str, dict[str, tuple[str, str]]] = {
    "sft_train": {
        "timeout": ("vision_frame_lease_elapsed", "detector_probe_clock_expired"),
        "transport": ("inference_socket_half_closed", "feature_stream_channel_reset"),
        "quota": ("probe_slot_limit_reached", "detector_credit_window_empty"),
        "payload": ("detector_envelope_schema_rejected", "vision_request_digest_invalid"),
    },
    "sft_dev": {
        "timeout": ("render_wait_budget_spent", "feature_probe_deadline_over"),
        "transport": ("vision_peer_pipe_detached", "detector_route_carrier_lost"),
        "quota": ("inspection_token_pool_depleted", "probe_lane_capacity_exceeded"),
        "payload": ("visual_packet_contract_broken", "detector_input_stamp_malformed"),
    },
    "grpo_train": {
        "timeout": ("inference_watchdog_interval_closed", "visual_task_clock_consumed"),
        "transport": ("detector_mesh_edge_dropped", "feature_bus_session_aborted"),
        "quota": ("vision_worker_permit_unavailable", "probe_concurrency_ceiling_hit"),
        "payload": ("image_request_header_corrupt", "detector_payload_shape_unsupported"),
    },
    "grpo_dev": {
        "timeout": ("probe_response_horizon_crossed", "detector_wait_epoch_finished"),
        "transport": ("vision_relay_tunnel_closed", "inference_link_epoch_reset"),
        "quota": ("detector_admission_window_full", "visual_probe_lease_denied"),
        "payload": ("frame_manifest_field_invalid", "probe_request_signature_bad"),
    },
    "test": {
        "timeout": ("visual_ack_window_missed", "detector_turnaround_limit_reached"),
        "transport": ("inference_backplane_route_lost", "vision_exchange_channel_broken"),
        "quota": ("probe_execution_ticket_exhausted", "detector_queue_budget_unavailable"),
        "payload": ("visual_job_descriptor_invalid", "detector_frame_contract_mismatch"),
    },
}

FIXTURES: dict[str, tuple[dict[str, Any], ...]] = {
    "sft_train": (
        {"target": "黄色六边巡检牌", "slug": "yellow_hex_patrol_marker", "family": "prmv6_st_hex_marker", "shape": "hexagon", "fg": (235, 190, 45), "bg": (38, 49, 61)},
        {"target": "青色三辐手轮", "slug": "cyan_three_spoke_wheel", "family": "prmv6_st_spoke_wheel", "shape": "wheel", "fg": (49, 173, 178), "bg": (220, 226, 229)},
    ),
    "sft_dev": (
        {"target": "紫色梯形校准块", "slug": "violet_trapezoid_calibrator", "family": "prmv6_sd_trapezoid", "shape": "trapezoid", "fg": (135, 77, 183), "bg": (232, 228, 220)},
        {"target": "银色双环接头", "slug": "silver_double_ring_joint", "family": "prmv6_sd_double_ring", "shape": "rings", "fg": (177, 188, 197), "bg": (46, 54, 63)},
    ),
    "grpo_train": (
        {"target": "橙色叉形定位架", "slug": "orange_cross_locator", "family": "prmv6_gt_cross_locator", "shape": "cross", "fg": (230, 121, 38), "bg": (43, 47, 54)},
        {"target": "靛蓝四叶旋钮", "slug": "indigo_four_lobe_knob", "family": "prmv6_gt_lobe_knob", "shape": "clover", "fg": (63, 83, 164), "bg": (224, 220, 209)},
    ),
    "grpo_dev": (
        {"target": "绿色弧顶隔离柱", "slug": "green_arch_isolator", "family": "prmv6_gd_arch_isolator", "shape": "arch", "fg": (65, 152, 91), "bg": (215, 221, 216)},
        {"target": "铜色双槽法兰", "slug": "copper_twin_slot_flange", "family": "prmv6_gd_slot_flange", "shape": "flange", "fg": (177, 111, 61), "bg": (47, 51, 56)},
    ),
    "test": (
        {"target": "玫红五角检修标", "slug": "magenta_star_service_mark", "family": "prmv6_te_star_mark", "shape": "star", "fg": (194, 54, 119), "bg": (231, 226, 218)},
        {"target": "湖蓝折线警示框", "slug": "lake_blue_zigzag_frame", "family": "prmv6_te_zigzag_frame", "shape": "zigzag", "fg": (44, 145, 184), "bg": (39, 47, 57)},
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def deterministic_id(prefix: str, *parts: Any) -> str:
    payload = "|".join(str(part) for part in (SEED, *parts))
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:20]}"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in materialized),
        encoding="utf-8",
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: row must be an object")
        rows.append(value)
    return rows


def _regular_polygon(cx: float, cy: float, radius: float, sides: int, rotation: float = 0.0) -> list[tuple[float, float]]:
    return [
        (
            cx + radius * math.cos(rotation + 2 * math.pi * index / sides),
            cy + radius * math.sin(rotation + 2 * math.pi * index / sides),
        )
        for index in range(sides)
    ]


def _star(cx: float, cy: float, outer: float, inner: float) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for index in range(10):
        radius = outer if index % 2 == 0 else inner
        angle = -math.pi / 2 + index * math.pi / 5
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return points


def draw_fixture(path: Path, spec: dict[str, Any], *, index: int) -> None:
    image = Image.new("RGB", (320, 240), tuple(spec["bg"]))
    draw = ImageDraw.Draw(image)
    fg = tuple(spec["fg"])
    outline = tuple(max(0, channel - 70) for channel in fg)
    draw.rectangle((0, 190, 320, 240), fill=tuple(min(255, channel + 25) for channel in spec["bg"]))
    shape = str(spec["shape"])
    if shape == "hexagon":
        draw.polygon(_regular_polygon(160, 112, 82, 6, math.pi / 6), fill=fg, outline=outline, width=9)
        draw.ellipse((145, 96, 175, 126), fill=outline)
    elif shape == "wheel":
        draw.ellipse((72, 24, 248, 200), fill=fg, outline=outline, width=12)
        draw.ellipse((110, 62, 210, 162), fill=spec["bg"], outline=outline, width=8)
        for angle in (-math.pi / 2, math.pi / 6, 5 * math.pi / 6):
            draw.line((160, 112, 160 + 80 * math.cos(angle), 112 + 80 * math.sin(angle)), fill=outline, width=10)
        draw.ellipse((144, 96, 176, 128), fill=fg, outline=outline, width=4)
    elif shape == "trapezoid":
        draw.polygon(((95, 45), (225, 45), (274, 184), (46, 184)), fill=fg, outline=outline, width=10)
        draw.rectangle((133, 80, 187, 150), fill=spec["bg"], outline=outline, width=6)
    elif shape == "rings":
        for x in (80, 150):
            draw.ellipse((x, 46, x + 92, 174), fill=fg, outline=outline, width=9)
            draw.ellipse((x + 22, 73, x + 70, 147), fill=spec["bg"], outline=outline, width=5)
    elif shape == "cross":
        draw.polygon(((132, 28), (188, 28), (188, 82), (252, 82), (252, 142), (188, 142), (188, 198), (132, 198), (132, 142), (68, 142), (68, 82), (132, 82)), fill=fg, outline=outline)
        draw.ellipse((143, 98, 177, 132), fill=outline)
    elif shape == "clover":
        for box in ((126, 24, 194, 104), (126, 120, 194, 200), (64, 82, 144, 150), (176, 82, 256, 150)):
            draw.ellipse(box, fill=fg, outline=outline, width=7)
        draw.ellipse((126, 78, 194, 146), fill=fg, outline=outline, width=8)
    elif shape == "arch":
        draw.rounded_rectangle((104, 25, 216, 204), radius=52, fill=fg, outline=outline, width=9)
        draw.rectangle((132, 92, 188, 204), fill=spec["bg"], outline=outline, width=7)
    elif shape == "flange":
        draw.ellipse((62, 18, 258, 210), fill=fg, outline=outline, width=11)
        draw.ellipse((114, 70, 206, 158), fill=spec["bg"], outline=outline, width=7)
        draw.rounded_rectangle((82, 83, 112, 145), radius=12, fill=spec["bg"], outline=outline, width=5)
        draw.rounded_rectangle((208, 83, 238, 145), radius=12, fill=spec["bg"], outline=outline, width=5)
    elif shape == "star":
        draw.polygon(_star(160, 112, 94, 42), fill=fg, outline=outline)
        draw.ellipse((144, 96, 176, 128), fill=outline)
    elif shape == "zigzag":
        points = ((58, 52), (118, 52), (145, 92), (178, 52), (262, 52), (231, 113), (262, 174), (190, 174), (160, 137), (128, 174), (58, 174), (88, 113))
        draw.line((*points, points[0]), fill=fg, width=18, joint="curve")
        draw.line((*points, points[0]), fill=outline, width=4, joint="curve")
    else:
        raise ValueError(f"unknown fixture shape: {shape}")
    draw.text((12, 214), f"V{index + 1:02d}", fill=(245, 245, 245))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False, compress_level=9)


def fixture_path(spec: dict[str, Any]) -> Path:
    return FIXTURE_DIR / f"{spec['slug']}.png"


def fixture_relative_path(spec: dict[str, Any]) -> str:
    return str(fixture_path(spec).relative_to(ROOT))


def write_fixture_images() -> list[Path]:
    paths: list[Path] = []
    index = 0
    for split in ROLE_SPECS:
        for spec in FIXTURES[split]:
            path = fixture_path(spec)
            draw_fixture(path, spec, index=index)
            paths.append(path)
            index += 1
    hashes = [sha256_file(path) for path in paths]
    if len(hashes) != len(set(hashes)):
        raise ValueError("fixture images must have unique content hashes")
    return paths


def build_projects(split: str) -> list[str]:
    lexicon = LEXICONS[split]
    pairs = [(root, suffix) for root in lexicon["roots"] for suffix in lexicon["suffixes"]]
    expected = int(ROLE_SPECS[split]["entities"])
    if len(pairs) != expected:
        raise AssertionError(f"{split}: lexicon produces {len(pairs)} entities, expected {expected}")
    rng = random.Random(SEED + sum(ord(char) for char in split))
    rng.shuffle(pairs)
    projects: list[str] = []
    for index, (root, suffix) in enumerate(pairs):
        code = 1000 + int(hashlib.sha256(f"{split}|{index}|{root}|{suffix}".encode()).hexdigest()[:6], 16) % 9000
        projects.append(f"{root}{suffix}{code:04d}号")
    return projects


PROJECTS = {split: build_projects(split) for split in ROLE_SPECS}


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


def policy_query(
    *,
    split: str,
    style: str,
    project: str,
    target: str,
    model_text: str,
) -> str:
    policy = POLICY_WORDING[split]
    return (
        f"{style}：先使用{model_text}检测附件中的{target}，该检测只提供中间状态。"
        f"最终目标是判断{project}能否直接沿用现有视觉能力。"
        + "".join(policy)
        + "检测工具设 finish_after_tool=false；迁移顾问使用当前图片与视觉探针，"
        "并设 finish_after_tool=true。"
    )


def _field_text(key: str, value: Any) -> str:
    if isinstance(value, bool):
        rendered = "true" if value else "false"
    elif value is None:
        rendered = "NA"
    else:
        rendered = str(value)
    return f"{key}={rendered}"


def structured_observation(
    after_step: int,
    *,
    success: bool,
    status: str,
    candidate_count: int | None,
    min_confidence: float | None,
    cross_prompt_iou: float | None,
    domain_shift: str,
    gateway_error: str,
    retryable: bool | None,
    retry_count: int | str | None,
    badge_mode: str,
    omit_fields: tuple[str, ...] = (),
) -> dict[str, Any]:
    values: list[tuple[str, Any]] = [
        ("candidate_count", candidate_count),
        ("min_confidence", min_confidence),
        ("cross_prompt_iou", cross_prompt_iou),
        ("domain_shift", domain_shift),
        ("gateway_error", gateway_error),
        ("retryable", retryable),
        ("retry_count", retry_count),
    ]
    fields = [_field_text(key, value) for key, value in values if key not in set(omit_fields)]
    if badge_mode != "missing":
        fields.append(_field_text("overall_badge", badge_mode))
    summary = "当前结构化状态：" + "；".join(fields) + "。说明：展示字段不覆盖结构化值。"
    return v1.observation(
        after_step,
        success=success,
        status=status,
        summary=summary,
    )


def error_observation(
    after_step: int,
    *,
    alias: str,
    retryable: bool,
    retry_count: int,
    badge_mode: str,
) -> dict[str, Any]:
    return structured_observation(
        after_step,
        success=False,
        status="gateway_error",
        candidate_count=None,
        min_confidence=None,
        cross_prompt_iou=None,
        domain_shift="unknown",
        gateway_error=alias,
        retryable=retryable,
        retry_count=retry_count,
        badge_mode=badge_mode,
    )


def success_observation(
    after_step: int,
    *,
    badge_mode: str,
    retry_count: int,
) -> dict[str, Any]:
    return structured_observation(
        after_step,
        success=True,
        status="completed",
        candidate_count=2,
        min_confidence=0.93,
        cross_prompt_iou=0.81,
        domain_shift="low",
        gateway_error="none",
        retryable=False,
        retry_count=retry_count,
        badge_mode=badge_mode,
    )


def veto_observation(
    after_step: int,
    *,
    badge_mode: str,
    retry_count: int,
    veto_index: int,
) -> dict[str, Any]:
    variants = (
        {"candidate_count": 0, "min_confidence": 0.94, "cross_prompt_iou": 0.82, "domain_shift": "low"},
        {"candidate_count": 2, "min_confidence": 0.61, "cross_prompt_iou": 0.80, "domain_shift": "low"},
        {"candidate_count": 2, "min_confidence": 0.92, "cross_prompt_iou": 0.48, "domain_shift": "low"},
        {"candidate_count": 3, "min_confidence": 0.94, "cross_prompt_iou": 0.84, "domain_shift": "high"},
    )
    selected = variants[veto_index % len(variants)]
    return structured_observation(
        after_step,
        success=True,
        status="completed",
        candidate_count=int(selected["candidate_count"]),
        min_confidence=float(selected["min_confidence"]),
        cross_prompt_iou=float(selected["cross_prompt_iou"]),
        domain_shift=str(selected["domain_shift"]),
        gateway_error="none",
        retryable=False,
        retry_count=retry_count,
        badge_mode=badge_mode,
    )


def entity_id(split: str, entity_index: int) -> str:
    return f"prmv6_{ROLE_SPECS[split]['code'].lower()}_entity_{entity_index + 1:03d}"


def fixture_for(split: str, entity_index: int) -> dict[str, Any]:
    return FIXTURES[split][entity_index % len(FIXTURES[split])]


def alias_for(split: str, entity_index: int, detector_index: int) -> tuple[str, str]:
    families = tuple(ERROR_ALIASES[split])
    family = families[(entity_index + 2 * detector_index) % len(families)]
    aliases = ERROR_ALIASES[split][family]
    alias = aliases[(entity_index // len(families) + detector_index) % len(aliases)]
    return family, alias


def _role_fields(split: str) -> dict[str, Any]:
    spec = ROLE_SPECS[split]
    return {
        "split": split,
        "data_stage": spec["stage"],
        "selection_role": spec["selection_role"],
        "training_only": spec["training_only"],
        "evaluation_only": spec["evaluation_only"],
        "exclude_from_training": spec["exclude_from_training"],
        "sealed": spec["sealed"],
    }


def _base_case(
    *,
    split: str,
    entity_index: int,
    detector_index: int,
    scenario_id: str,
    case_suffix: str,
    query: str,
    expected: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    target_action_class: str,
    error_family: str,
    error_alias: str,
    badge_mode: str,
    fixture: dict[str, Any],
    template_id: str,
    counterfactual_bundle_id: str,
    guardrail: bool,
    query_trajectories: list[dict[str, Any]] | None = None,
    post_retry_outcome: str = "",
) -> dict[str, Any]:
    detector, _, _ = DETECTORS[detector_index]
    eid = entity_id(split, entity_index)
    code = str(ROLE_SPECS[split]["code"])
    case_id = f"PRMV6-{code}-{entity_index + 1:03d}-{detector.upper()}-{case_suffix}"
    allowed = v1.allowed_actions(expected)
    project = PROJECTS[split][entity_index]
    return {
        "case_id": case_id,
        "dataset_id": DATASET_ID,
        "schema_version": SCHEMA_VERSION,
        **_role_fields(split),
        "entity_id": eid,
        "group_id": counterfactual_bundle_id,
        "project_entity": project,
        "target_entity": str(fixture["target"]),
        "counterfactual_bundle_id": counterfactual_bundle_id,
        "template_id": template_id,
        "scenario_id": scenario_id,
        "category": scenario_id,
        "detector_family": detector,
        "target_action_class": target_action_class,
        "decision_under_test_step": 2,
        "grpo_target_step": 2,
        "grpo_eligible": not guardrail,
        "sft_eligible": split.startswith("sft_"),
        "guardrail": guardrail,
        "error_family": error_family,
        "error_alias": error_alias,
        "badge_condition": badge_mode,
        "post_retry_outcome": post_retry_outcome,
        "user_query": query,
        "image_fixture_family": str(fixture["family"]),
        "setup": {
            **v1.setup(fixture=fixture_relative_path(fixture)),
            "max_steps": 3,
            "query_trajectories": query_trajectories or [],
        },
        "expected_decisions": expected,
        "mock_observations": observations,
        "forbidden_actions": [tool for tool in v1.ACTIVE_TOOLS if tool not in allowed],
        "reward_spec": dict(VALUE_REWARD),
        "provenance_class": "independent_synthetic_counterfactual_v6",
    }


def make_core_cases(split: str, entity_index: int, detector_index: int) -> list[dict[str, Any]]:
    detector, action, model_text = DETECTORS[detector_index]
    fixture = fixture_for(split, entity_index)
    project = PROJECTS[split][entity_index]
    styles = LEXICONS[split]["styles"]
    style_index = entity_index % len(styles)
    query = policy_query(
        split=split,
        style=styles[style_index],
        project=project,
        target=str(fixture["target"]),
        model_text=model_text,
    )
    family, alias = alias_for(split, entity_index, detector_index)
    badge_mode = BADGE_MODES[(entity_index + detector_index) % len(BADGE_MODES)]
    bundle = f"{entity_id(split, entity_index)}_{detector}_{alias}"
    template_id = f"prmv6_{split}_{detector}_policy_{style_index + 1}"
    first = detection_step(action, str(fixture["target"]))
    post_outcome = ("success", "metric_veto", "error")[(2 * entity_index + detector_index) % 3]

    fresh_first = error_observation(
        1,
        alias=alias,
        retryable=True,
        retry_count=0,
        badge_mode=badge_mode,
    )
    if post_outcome == "success":
        fresh_second = success_observation(2, badge_mode=badge_mode, retry_count=1)
        final = v1.end_step()
    elif post_outcome == "metric_veto":
        fresh_second = veto_observation(
            2,
            badge_mode=badge_mode,
            retry_count=1,
            veto_index=entity_index + detector_index,
        )
        final = migration_step(project)
    else:
        fresh_second = error_observation(
            2,
            alias=alias,
            retryable=True,
            retry_count=1,
            badge_mode=badge_mode,
        )
        final = migration_step(project)

    exhausted_count = 1 + ((entity_index + detector_index) % 2)
    cases = [
        _base_case(
            split=split,
            entity_index=entity_index,
            detector_index=detector_index,
            scenario_id="core_retryable_fresh",
            case_suffix="RF",
            query=query,
            expected=[first, detection_step(action, str(fixture["target"])), final],
            observations=[fresh_first, fresh_second],
            target_action_class="retry",
            error_family=family,
            error_alias=alias,
            badge_mode=badge_mode,
            fixture=fixture,
            template_id=template_id,
            counterfactual_bundle_id=bundle,
            guardrail=False,
            post_retry_outcome=post_outcome,
        ),
        _base_case(
            split=split,
            entity_index=entity_index,
            detector_index=detector_index,
            scenario_id="core_nonretryable",
            case_suffix="NR",
            query=query,
            expected=[first, migration_step(project)],
            observations=[error_observation(1, alias=alias, retryable=False, retry_count=0, badge_mode=badge_mode)],
            target_action_class="migrate",
            error_family=family,
            error_alias=alias,
            badge_mode=badge_mode,
            fixture=fixture,
            template_id=template_id,
            counterfactual_bundle_id=bundle,
            guardrail=False,
        ),
        _base_case(
            split=split,
            entity_index=entity_index,
            detector_index=detector_index,
            scenario_id="core_budget_exhausted",
            case_suffix="BE",
            query=query,
            expected=[first, migration_step(project)],
            observations=[error_observation(1, alias=alias, retryable=True, retry_count=exhausted_count, badge_mode=badge_mode)],
            target_action_class="migrate",
            error_family=family,
            error_alias=alias,
            badge_mode=badge_mode,
            fixture=fixture,
            template_id=template_id,
            counterfactual_bundle_id=bundle,
            guardrail=False,
        ),
    ]
    return cases


def stale_history(
    *,
    project: str,
    target: str,
    action: str,
    current_success: bool,
) -> list[dict[str, Any]]:
    if current_success:
        summary = (
            "归档状态：candidate_count=NA；min_confidence=NA；cross_prompt_iou=NA；"
            "domain_shift=unknown；gateway_error=archived_link_fault；retryable=false；retry_count=0。"
        )
    else:
        summary = (
            "归档状态：candidate_count=2；min_confidence=0.96；cross_prompt_iou=0.86；"
            "domain_shift=low；gateway_error=none；retryable=false；retry_count=0。"
        )
    return [
        {
            "query": f"归档复核：检查{project}中的{target}",
            "result_summary": "这是已关闭批次，只用于验证当前状态优先级。",
            "steps": [
                {
                    "step_index": 1,
                    "action": action,
                    "action_input": {"label": target, "finish_after_tool": False},
                    "observation": {
                        "success": not current_success,
                        "status": "completed" if not current_success else "gateway_error",
                        "summary": summary,
                    },
                }
            ],
        }
    ]


def make_guard_case(split: str, guard_index: int) -> dict[str, Any]:
    entity_count = int(ROLE_SPECS[split]["entities"])
    entity_index = guard_index % entity_count
    occurrence = guard_index // entity_count
    detector_index = (entity_index + occurrence + guard_index // 6) % len(DETECTORS)
    detector, action, model_text = DETECTORS[detector_index]
    fixture = fixture_for(split, entity_index)
    project = PROJECTS[split][entity_index]
    styles = LEXICONS[split]["styles"]
    style_index = (entity_index + occurrence + 1) % len(styles)
    query = policy_query(
        split=split,
        style=styles[style_index],
        project=project,
        target=str(fixture["target"]),
        model_text=model_text,
    )
    family, alias = alias_for(split, entity_index + guard_index, detector_index)
    badge_mode = BADGE_MODES[guard_index % len(BADGE_MODES)]
    guard_type = GUARD_TYPES[guard_index % len(GUARD_TYPES)]
    first = detection_step(action, str(fixture["target"]))
    query_trajectories: list[dict[str, Any]] = []
    target_action = "migrate"

    if guard_type == "initial_success_end":
        expected = [first, v1.end_step()]
        observations = [success_observation(1, badge_mode=badge_mode, retry_count=0)]
        target_action = "end"
        error_alias = "none"
    elif guard_type == "initial_metric_veto_migrate":
        expected = [first, migration_step(project)]
        observations = [veto_observation(1, badge_mode=badge_mode, retry_count=0, veto_index=guard_index)]
        error_alias = "none"
    elif guard_type == "missing_required_state_migrate":
        expected = [first, migration_step(project)]
        observations = [
            structured_observation(
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
                badge_mode=badge_mode,
                omit_fields=("retryable",),
            )
        ]
        error_alias = alias
    elif guard_type == "conflicting_state_migrate":
        expected = [first, migration_step(project)]
        observations = [
            structured_observation(
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
                badge_mode=badge_mode,
            )
        ]
        error_alias = "none"
    elif guard_type == "stale_history_current_success_end":
        expected = [first, v1.end_step()]
        observations = [success_observation(1, badge_mode=badge_mode, retry_count=0)]
        target_action = "end"
        error_alias = "none"
        query_trajectories = stale_history(
            project=project,
            target=str(fixture["target"]),
            action=action,
            current_success=True,
        )
    elif guard_type == "stale_history_current_error_migrate":
        expected = [first, migration_step(project)]
        observations = [error_observation(1, alias=alias, retryable=False, retry_count=0, badge_mode=badge_mode)]
        error_alias = alias
        query_trajectories = stale_history(
            project=project,
            target=str(fixture["target"]),
            action=action,
            current_success=False,
        )
    else:
        raise ValueError(f"unknown guard type: {guard_type}")

    return _base_case(
        split=split,
        entity_index=entity_index,
        detector_index=detector_index,
        scenario_id=f"guard_{guard_type}",
        case_suffix=f"G{guard_index + 1:03d}",
        query=query,
        expected=expected,
        observations=observations,
        target_action_class=target_action,
        error_family=family if error_alias != "none" else "none",
        error_alias=error_alias,
        badge_mode=badge_mode,
        fixture=fixture,
        template_id=f"prmv6_{split}_{detector}_guard_{style_index + 1}",
        counterfactual_bundle_id=f"{entity_id(split, entity_index)}_{detector}_guard_{guard_index + 1:03d}",
        guardrail=True,
        query_trajectories=query_trajectories,
    )


def build_split_cases(split: str) -> list[dict[str, Any]]:
    entity_count = int(ROLE_SPECS[split]["entities"])
    core = [
        case
        for entity_index in range(entity_count)
        for detector_index in range(len(DETECTORS))
        for case in make_core_cases(split, entity_index, detector_index)
    ]
    guard_count = len(core) // 4
    guards = [make_guard_case(split, guard_index) for guard_index in range(guard_count)]
    rows = core + guards
    random.Random(SEED + 1000 + sum(ord(char) for char in split)).shuffle(rows)
    return rows


def build_all_cases() -> dict[str, list[dict[str, Any]]]:
    return {split: build_split_cases(split) for split in ROLE_SPECS}


def _counter(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _observation_summary(row: dict[str, Any], after_step: int = 1) -> str:
    for item in row.get("mock_observations") if isinstance(row.get("mock_observations"), list) else []:
        if isinstance(item, dict) and int(item.get("after_step") or 0) == after_step:
            observation = item.get("observation") if isinstance(item.get("observation"), dict) else {}
            return str(observation.get("summary") or "")
    return ""


def mutual_information(rows: list[dict[str, Any]], left_key: str, right_key: str) -> float:
    joint = Counter((str(row.get(left_key) or ""), str(row.get(right_key) or "")) for row in rows)
    left = Counter(key[0] for key in joint.elements())
    right = Counter(key[1] for key in joint.elements())
    total = sum(joint.values())
    if total <= 0:
        return 0.0
    value = 0.0
    for (left_value, right_value), count in joint.items():
        probability = count / total
        value += probability * math.log(
            probability / ((left[left_value] / total) * (right[right_value] / total)),
            2,
        )
    return value


def _normalized_query(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).lower()


def parse_structured_summary(summary: str) -> dict[str, str]:
    return {
        key: value.strip()
        for key, value in re.findall(r"([A-Za-z_]+)=([^；。]+)", str(summary or ""))
    }


def independent_oracle_action(summary: str) -> str:
    """Derive the next action class from observation fields without scenario metadata."""

    fields = parse_structured_summary(summary)
    required = {"gateway_error", "retryable", "retry_count"}
    if not required.issubset(fields):
        return "migrate"
    retryable_text = fields["retryable"]
    if retryable_text not in {"true", "false"}:
        return "migrate"
    try:
        retry_count = int(fields["retry_count"])
    except (TypeError, ValueError):
        return "migrate"
    gateway_error = fields["gateway_error"]
    if gateway_error != "none":
        return "retry" if retryable_text == "true" and retry_count == 0 else "migrate"

    metric_keys = {"candidate_count", "min_confidence", "cross_prompt_iou", "domain_shift"}
    if not metric_keys.issubset(fields) or retryable_text != "false" or retry_count < 0:
        return "migrate"
    try:
        candidate_count = int(fields["candidate_count"])
        min_confidence = float(fields["min_confidence"])
        cross_prompt_iou = float(fields["cross_prompt_iou"])
    except (TypeError, ValueError):
        return "migrate"
    passes = (
        candidate_count >= 1
        and min_confidence >= 0.88
        and cross_prompt_iou >= 0.72
        and fields["domain_shift"] == "low"
    )
    return "end" if passes else "migrate"


def expected_action_class(step: dict[str, Any], *, detector_action: str) -> str:
    action = expected_action_name(step)
    if action == detector_action:
        return "retry"
    if action == "migration_advisor":
        return "migrate"
    if action == "end":
        return "end"
    return action


def _protected_values(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    values: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        setup = row.get("setup") if isinstance(row.get("setup"), dict) else {}
        fixture = str(setup.get("image_fixture") or "")
        candidates = {
            "case_id": str(row.get("case_id") or ""),
            "entity_id": str(row.get("entity_id") or ""),
            "project_entity": str(row.get("project_entity") or ""),
            "target_entity": str(row.get("target_entity") or ""),
            "normalized_query": _normalized_query(str(row.get("user_query") or "")),
            "template_id": str(row.get("template_id") or ""),
            "error_alias": str(row.get("error_alias") or ""),
            "fixture_family": str(row.get("image_fixture_family") or ""),
            "fixture_path": fixture,
        }
        path = ROOT / fixture
        candidates["fixture_sha256"] = sha256_file(path) if fixture and path.is_file() else ""
        for key, value in candidates.items():
            if value and not (key == "error_alias" and value == "none"):
                values[key].add(value)
    return dict(values)


PROTECTED_KEYS = (
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


def _overlap(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    left_values = _protected_values(left)
    right_values = _protected_values(right)
    overlaps = {
        key: sorted(left_values.get(key, set()) & right_values.get(key, set()))
        for key in PROTECTED_KEYS
    }
    return {
        "status": "pass" if all(not values for values in overlaps.values()) else "fail",
        "counts": {key: len(values) for key, values in overlaps.items()},
        "examples": {key: values[:8] for key, values in overlaps.items() if values},
    }


def repository_isolation_report(all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    output_names = {f"{DATASET_ID}_{split}_cases.jsonl" for split in ROLE_SPECS}
    paths = [path for path in sorted(CASE_DIR.glob("*_cases.jsonl")) if path.name not in output_names]
    existing_rows = [row for path in paths for row in load_jsonl(path)]
    report = _overlap(all_rows, existing_rows)
    return {
        **report,
        "files_scanned": [str(path.relative_to(ROOT)) for path in paths],
        "protected_fields": list(PROTECTED_KEYS),
    }


def split_isolation_report(cases_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    names = list(cases_by_split)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            reports[f"{left_name}__{right_name}"] = _overlap(
                cases_by_split[left_name], cases_by_split[right_name]
            )
    return {
        "status": "pass" if all(report["status"] == "pass" for report in reports.values()) else "fail",
        "pairs": reports,
    }


def validate_split(split: str, rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    entity_count = int(ROLE_SPECS[split]["entities"])
    expected_core = entity_count * len(DETECTORS) * len(CORE_STATES)
    expected_guards = expected_core // 4
    expected_total = expected_core + expected_guards
    if len(rows) != expected_total:
        errors.append(f"{split}: expected {expected_total} cases, got {len(rows)}")
    ids = [str(row.get("case_id") or "") for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        errors.append(f"{split}: case IDs must be non-empty and unique")
    entities = {str(row.get("entity_id") or "") for row in rows}
    if len(entities) != entity_count:
        errors.append(f"{split}: expected {entity_count} entities, got {len(entities)}")

    core = [row for row in rows if row.get("guardrail") is False]
    guards = [row for row in rows if row.get("guardrail") is True]
    if len(core) != expected_core or len(guards) != expected_guards:
        errors.append(f"{split}: core/guard counts are {len(core)}/{len(guards)}")
    bundles: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in core:
        bundles[str(row.get("counterfactual_bundle_id") or "")].append(row)
    if len(bundles) != entity_count * len(DETECTORS):
        errors.append(f"{split}: expected {entity_count * 2} counterfactual bundles, got {len(bundles)}")
    for bundle_id, bundle in bundles.items():
        scenarios = {str(row.get("scenario_id") or "") for row in bundle}
        expected_scenarios = {f"core_{state}" for state in CORE_STATES}
        if len(bundle) != 3 or scenarios != expected_scenarios:
            errors.append(f"{split}/{bundle_id}: incomplete matched triple: {sorted(scenarios)}")
        if len({str(row.get("user_query") or "") for row in bundle}) != 1:
            errors.append(f"{split}/{bundle_id}: matched triple query changed")
        if len({str(row.get("error_alias") or "") for row in bundle}) != 1:
            errors.append(f"{split}/{bundle_id}: matched triple alias changed")
        if len({str(row.get("badge_condition") or "") for row in bundle}) != 1:
            errors.append(f"{split}/{bundle_id}: matched triple badge nuisance changed")
        if Counter(str(row.get("target_action_class") or "") for row in bundle) != Counter({"migrate": 2, "retry": 1}):
            errors.append(f"{split}/{bundle_id}: expected retry/migrate=1/2")
        first_observations = [
            (row.get("mock_observations") or [{}])[0].get("observation") or {}
            for row in bundle
        ]
        if len({str(observation.get("status") or "") for observation in first_observations}) != 1:
            errors.append(f"{split}/{bundle_id}: first-observation status changed")
        normalized_summaries = {
            re.sub(
                r"retryable=(?:true|false)；retry_count=[^；。]+",
                "retryable=<blocked>；retry_count=<blocked>",
                str(observation.get("summary") or ""),
            )
            for observation in first_observations
        }
        if len(normalized_summaries) != 1:
            errors.append(
                f"{split}/{bundle_id}: first observations differ outside retryable/retry_count"
            )
        for row in bundle:
            expected = row.get("expected_decisions") or []
            target_action = expected_action_name(expected[1]) if len(expected) > 1 else ""
            expected_target = (
                "migration_advisor"
                if row.get("target_action_class") == "migrate"
                else "qwen_detection"
                if row.get("detector_family") == "qwen"
                else "rexomni_detection"
            )
            if target_action != expected_target:
                errors.append(
                    f"{split}/{row.get('case_id')}: target label/action mismatch "
                    f"{row.get('target_action_class')}/{target_action}"
                )

    alias_actions: dict[str, set[str]] = defaultdict(set)
    badge_actions: dict[str, set[str]] = defaultdict(set)
    for row in core:
        alias_actions[str(row["error_alias"])].add(str(row["target_action_class"]))
        badge_actions[str(row["badge_condition"])].add(str(row["target_action_class"]))
    for alias, actions in alias_actions.items():
        if actions != {"retry", "migrate"}:
            errors.append(f"{split}/{alias}: alias does not support both actions")
    for badge, actions in badge_actions.items():
        if actions != {"retry", "migrate"}:
            errors.append(f"{split}/{badge}: badge does not support both actions")
    if abs(mutual_information(core, "badge_condition", "target_action_class")) > 1e-12:
        errors.append(f"{split}: badge/action mutual information is not zero")
    if abs(mutual_information(core, "error_alias", "target_action_class")) > 1e-12:
        errors.append(f"{split}: alias/action mutual information is not zero")

    forbidden_hints = ("migration_advisor", "应迁移", "应该重试", "再次调用", "直接结束")
    spec = ROLE_SPECS[split]
    for row in rows:
        case_id = str(row.get("case_id") or "")
        if row.get("split") != split or row.get("dataset_id") != DATASET_ID:
            errors.append(f"{case_id}: dataset/split identity mismatch")
        for key in ("training_only", "evaluation_only", "exclude_from_training", "sealed"):
            if row.get(key) is not spec[key]:
                errors.append(f"{case_id}: role flag {key} mismatch")
        if int(row.get("decision_under_test_step") or 0) != 2 or int(row.get("grpo_target_step") or 0) != 2:
            errors.append(f"{case_id}: target transition must be step 2")
        expected = row.get("expected_decisions") if isinstance(row.get("expected_decisions"), list) else []
        observations = row.get("mock_observations") if isinstance(row.get("mock_observations"), list) else []
        if len(expected) not in {2, 3} or len(observations) != len(expected) - 1:
            errors.append(f"{case_id}: trajectory decision/observation lengths are inconsistent")
        if len(expected) == 3 and len(observations) != 2:
            errors.append(f"{case_id}: retry trajectory lacks its second observation")
        for item in observations:
            summary = str((item.get("observation") or {}).get("summary") or "") if isinstance(item, dict) else ""
            for hint in forbidden_hints:
                if hint in summary:
                    errors.append(f"{case_id}: observation leaks action hint {hint!r}")
            after_step = int(item.get("after_step") or 0) if isinstance(item, dict) else 0
            if not (1 <= after_step < len(expected)):
                errors.append(f"{case_id}: observation after_step={after_step} has no next gold step")
            else:
                detector_action = expected_action_name(expected[0])
                oracle = independent_oracle_action(summary)
                gold = expected_action_class(expected[after_step], detector_action=detector_action)
                if oracle != gold:
                    errors.append(
                        f"{case_id}: independent oracle/gold mismatch after step {after_step}: "
                        f"{oracle}/{gold}"
                    )
        if case_id and case_id in str(row.get("user_query") or ""):
            errors.append(f"{case_id}: case ID leaked into user query")
        setup = row.get("setup") if isinstance(row.get("setup"), dict) else {}
        if int(setup.get("max_steps") or 0) != 3:
            errors.append(f"{case_id}: max_steps must be 3")
        fixture = ROOT / str(setup.get("image_fixture") or "")
        if not fixture.is_file():
            errors.append(f"{case_id}: fixture missing: {fixture}")
        if not score_case(row)["passed"]:
            errors.append(f"{case_id}: canonical full-trajectory reward failed")

    guard_counts = Counter(str(row.get("scenario_id") or "") for row in guards)
    if max(guard_counts.values(), default=0) - min(guard_counts.values(), default=0) > 1:
        errors.append(f"{split}: guard scenarios are not near-balanced: {dict(guard_counts)}")
    return errors


def validate_all_cases(cases_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    errors = [error for split, rows in cases_by_split.items() for error in validate_split(split, rows)]
    all_rows = [row for rows in cases_by_split.values() for row in rows]
    all_ids = [str(row.get("case_id") or "") for row in all_rows]
    if len(all_ids) != len(set(all_ids)):
        errors.append("case IDs overlap across splits")
    split_isolation = split_isolation_report(cases_by_split)
    if split_isolation["status"] != "pass":
        errors.append("protected entities/content overlap across V6 splits")
    repository_isolation = repository_isolation_report(all_rows)
    if repository_isolation["status"] != "pass":
        errors.append("protected entities/content overlap with an existing repository dataset")
    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "split_isolation": split_isolation,
        "repository_isolation": repository_isolation,
    }


def expected_action_name(expected: dict[str, Any]) -> str:
    decision_type = str(expected.get("decision_type") or "tool").strip()
    if decision_type in {"clarify", "end"}:
        return decision_type
    return str(expected.get("action") or "").strip()


def canonical_decision(
    case: dict[str, Any],
    expected: dict[str, Any],
    *,
    step_index: int,
) -> dict[str, Any]:
    decision_type = str(expected.get("decision_type") or "tool").strip()
    if decision_type == "end":
        required = expected.get("required_args") if isinstance(expected.get("required_args"), dict) else {}
        return {
            "thought": "最新结构化状态已满足全部硬门槛，可以结束本轮规划。",
            "decision_type": "end",
            "end_reason": str(required.get("end_reason") or "memory_hit"),
            "final_answer": "",
        }
    if decision_type == "clarify":
        return {
            "thought": "当前意图存在关键歧义，需要先补充信息。",
            "decision_type": "clarify",
            "clarification_question": "请补充希望执行的视觉任务及最终目标。",
        }
    action = str(expected.get("action") or "").strip()
    action_input: dict[str, Any] = {}
    required = expected.get("required_args") if isinstance(expected.get("required_args"), dict) else {}
    action_input.update(required)
    contains = expected.get("arg_contains") if isinstance(expected.get("arg_contains"), dict) else {}
    for key, tokens in contains.items():
        if isinstance(tokens, list):
            action_input.setdefault(key, " ".join(str(token) for token in tokens))
        else:
            action_input.setdefault(key, str(tokens))
    if action in {"qwen_detection", "rexomni_detection"} and step_index > 1:
        thought = "最新状态满足一次重试条件，应继续使用同一检测器。"
    elif action in {"qwen_detection", "rexomni_detection"}:
        thought = "先执行用户指定的检测器以取得当前结构化状态。"
    elif action == "migration_advisor":
        thought = "最新状态未满足结束或重试条件，需要进入能力迁移评估。"
    else:
        thought = "根据最新结构化状态执行下一项工具。"
    return {
        "thought": thought,
        "decision_type": "tool",
        "action": action,
        "action_input": action_input,
        "final_answer": "",
    }


def _mock_observation(case: dict[str, Any], after_step: int) -> dict[str, Any]:
    for item in case.get("mock_observations") if isinstance(case.get("mock_observations"), list) else []:
        if not isinstance(item, dict):
            continue
        if int(item.get("after_step") or 0) == after_step and isinstance(item.get("observation"), dict):
            return json.loads(json.dumps(item["observation"], ensure_ascii=False))
    raise ValueError(f"{case.get('case_id')}: missing observation after step {after_step}")


def _clean_context_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _clean_context_value(item)
            for key, item in value.items()
            if str(key) not in {"external_ref", "_thought", "thought"}
        }
    if isinstance(value, list):
        return [_clean_context_value(item) for item in value]
    if isinstance(value, str) and (value.startswith("/raid/") or value.startswith("/tmp/")):
        return Path(value).name
    return value


def _history_trajectories(case: dict[str, Any], *, session_id: str, thread_id: str) -> list[dict[str, Any]]:
    setup = case.get("setup") if isinstance(case.get("setup"), dict) else {}
    raw_history = setup.get("query_trajectories") if isinstance(setup.get("query_trajectories"), list) else []
    trajectories: list[dict[str, Any]] = []
    identity = str(case.get("counterfactual_bundle_id") or case.get("entity_id") or "bundle")
    for trajectory_index, item in enumerate(raw_history, start=1):
        if not isinstance(item, dict):
            continue
        steps: list[dict[str, Any]] = []
        for step_index, raw_step in enumerate(item.get("steps") if isinstance(item.get("steps"), list) else [], start=1):
            if not isinstance(raw_step, dict):
                continue
            observation = _clean_context_value(raw_step.get("observation") if isinstance(raw_step.get("observation"), dict) else {})
            action = str(raw_step.get("action") or "")
            action_input = _clean_context_value(raw_step.get("action_input") if isinstance(raw_step.get("action_input"), dict) else {})
            if isinstance(observation, dict):
                observation["_action"] = action
                observation["_action_input"] = action_input
            steps.append(
                {
                    "step_id": f"step_{step_index}",
                    "action": action,
                    "observation_event_id": deterministic_id("evt", identity, "history", trajectory_index, step_index),
                    "observation": observation,
                }
            )
        trajectories.append(
            {
                "session_id": session_id,
                "thread_id": thread_id,
                "query_id": deterministic_id("qry", identity, "history", trajectory_index),
                "query": str(item.get("query") or ""),
                "result_summary": str(item.get("result_summary") or ""),
                "steps": steps,
            }
        )
    return trajectories


def sanitized_query_trajectories(case: dict[str, Any], *, step_index: int) -> list[dict[str, Any]]:
    identity = str(case.get("counterfactual_bundle_id") or case.get("entity_id") or "bundle")
    entity_identity = str(case.get("entity_id") or identity)
    session_id = deterministic_id("sess", entity_identity)
    thread_id = deterministic_id("thr", entity_identity)
    trajectories = _history_trajectories(case, session_id=session_id, thread_id=thread_id)
    expected = case.get("expected_decisions") if isinstance(case.get("expected_decisions"), list) else []
    steps: list[dict[str, Any]] = []
    for previous_index in range(1, step_index):
        expected_step = expected[previous_index - 1]
        if not isinstance(expected_step, dict):
            continue
        decision = canonical_decision(case, expected_step, step_index=previous_index)
        observation = _clean_context_value(_mock_observation(case, previous_index))
        action = str(decision.get("action") or "")
        action_input = _clean_context_value(decision.get("action_input") if isinstance(decision.get("action_input"), dict) else {})
        if isinstance(observation, dict):
            observation["_action"] = action
            observation["_action_input"] = action_input
        steps.append(
            {
                "step_id": f"step_{previous_index}",
                "action": action,
                "observation_event_id": deterministic_id("evt", identity, "current", previous_index),
                "observation": observation,
            }
        )
    trajectories.append(
        {
            "session_id": session_id,
            "thread_id": thread_id,
            "query_id": deterministic_id("qry", identity, "current"),
            "query": str(case.get("user_query") or ""),
            "result_summary": "",
            "steps": steps,
        }
    )
    return trajectories


def build_sanitized_pseudo_prompt(case: dict[str, Any], step_index: int) -> str:
    from capa import agent

    setup = case.get("setup") if isinstance(case.get("setup"), dict) else {}
    max_steps = int(setup.get("max_steps") or 3)
    if max_steps != 3:
        raise ValueError(f"{case.get('case_id')}: prompt max_steps must be 3")
    fixture = str(setup.get("image_fixture") or "")
    planner_context = {
        "schema_version": "sanitized-v1",
        "query_trajectories": sanitized_query_trajectories(case, step_index=step_index),
    }
    system_prompt = agent.build_agent_system_prompt(max_steps=max_steps)
    user_prompt = agent.build_agent_user_prompt(
        str(case.get("user_query") or ""),
        fixture or None,
        planner_context=planner_context,
        step_index=step_index,
        max_steps=max_steps,
    )
    return (
        "<|system|>\n"
        f"{system_prompt}\n"
        "<|user|>\n"
        f"{user_prompt}\n"
        "<|assistant|>\n"
    )


def pseudo_prompt_to_messages(prompt: str) -> list[dict[str, str]]:
    prefix = "<|system|>\n"
    separator = "\n<|user|>\n"
    suffix = "\n<|assistant|>\n"
    if not prompt.startswith(prefix) or separator not in prompt or not prompt.endswith(suffix):
        raise ValueError("prompt does not match the CAPA pseudo-chat contract")
    body = prompt[len(prefix) : -len(suffix)]
    system, user = body.split(separator, 1)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def render_nonthinking_prompt(tokenizer: Any, pseudo: str) -> str:
    rendered = tokenizer.apply_chat_template(
        pseudo_prompt_to_messages(pseudo),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not rendered.endswith(NONTHINKING_SUFFIX):
        raise ValueError(f"unexpected Qwen3.5 non-thinking prompt tail: {rendered[-96:]!r}")
    return rendered


def _row_metadata(case: dict[str, Any], *, step_index: int) -> dict[str, Any]:
    expected = case.get("expected_decisions") if isinstance(case.get("expected_decisions"), list) else []
    expected_step = expected[step_index - 1]
    previous_action = expected_action_name(expected[step_index - 2]) if step_index > 1 else ""
    return {
        "case_id": str(case.get("case_id") or ""),
        "dataset_id": DATASET_ID,
        "split": str(case.get("split") or ""),
        "category": str(case.get("category") or ""),
        "step_index": step_index,
        "expected_step": json.dumps(expected_step, ensure_ascii=False, sort_keys=True),
        "forbidden_actions": json.dumps(case.get("forbidden_actions") or [], ensure_ascii=False, sort_keys=True),
        "reward_spec": json.dumps(case.get("reward_spec") or {}, ensure_ascii=False, sort_keys=True),
        "previous_action": previous_action,
        "entity_id": str(case.get("entity_id") or ""),
        "group_id": str(case.get("group_id") or ""),
        "counterfactual_bundle_id": str(case.get("counterfactual_bundle_id") or ""),
        "template_id": str(case.get("template_id") or ""),
        "scenario_id": str(case.get("scenario_id") or ""),
        "target_action_class": str(case.get("target_action_class") or ""),
        "full_expected_actions": json.dumps(
            [expected_action_name(step) for step in expected if isinstance(step, dict)],
            ensure_ascii=False,
            sort_keys=True,
        ),
    }


def _render_row_prompt(case: dict[str, Any], step_index: int, tokenizer: Any) -> tuple[str, int]:
    pseudo = build_sanitized_pseudo_prompt(case, step_index)
    prompt = render_nonthinking_prompt(tokenizer, pseudo)
    token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if len(token_ids) > MAX_PROMPT_TOKENS:
        raise ValueError(
            f"{case.get('case_id')} step {step_index}: {len(token_ids)} prompt tokens exceeds {MAX_PROMPT_TOKENS}"
        )
    return prompt, len(token_ids)


def build_sft_rows(cases: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    raw_rows: list[dict[str, Any]] = []
    for case in sorted(cases, key=lambda row: str(row.get("case_id") or "")):
        expected = case.get("expected_decisions") if isinstance(case.get("expected_decisions"), list) else []
        for step_index, expected_step in enumerate(expected, start=1):
            if not isinstance(expected_step, dict):
                continue
            prompt, prompt_tokens = _render_row_prompt(case, step_index, tokenizer)
            completion_object = canonical_decision(case, expected_step, step_index=step_index)
            completion_json = json.dumps(completion_object, ensure_ascii=False, separators=(",", ":"))
            completion = completion_json + "<|im_end|>"
            raw_rows.append(
                {
                    "prompt": prompt,
                    "completion": completion,
                    **_row_metadata(case, step_index=step_index),
                    "data_stage": "sft",
                    "prompt_token_count": prompt_tokens,
                    "completion_token_count": len(tokenizer(completion, add_special_tokens=False)["input_ids"]),
                    "prompt_sha256": sha256_text(prompt),
                    "completion_sha256": sha256_text(completion),
                    "source_case_ids": [str(case.get("case_id") or "")],
                }
            )

    deduplicated: dict[tuple[str, str], dict[str, Any]] = {}
    for row in raw_rows:
        key = (str(row["prompt_sha256"]), str(row["completion_sha256"]))
        if key not in deduplicated:
            deduplicated[key] = row
        else:
            deduplicated[key]["source_case_ids"].extend(row["source_case_ids"])
    rows = list(deduplicated.values())
    for row in rows:
        row["source_case_ids"] = sorted(set(str(value) for value in row["source_case_ids"]))
        row["source_case_count"] = len(row["source_case_ids"])
    return rows


def build_grpo_rows(cases: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected = [case for case in cases if case.get("grpo_eligible") is True and case.get("guardrail") is False]
    for case in sorted(selected, key=lambda row: str(row.get("case_id") or "")):
        step_index = int(case.get("grpo_target_step") or 0)
        if step_index != 2:
            raise ValueError(f"{case.get('case_id')}: GRPO transition must be step 2")
        prompt, prompt_tokens = _render_row_prompt(case, step_index, tokenizer)
        rows.append(
            {
                "prompt": prompt,
                **_row_metadata(case, step_index=step_index),
                "data_stage": "grpo",
                "prompt_token_count": prompt_tokens,
                "prompt_sha256": sha256_text(prompt),
            }
        )
    return rows


def _canonical_score_for_stage_row(row: dict[str, Any]) -> float:
    from training.planner_grpo_seed_v1.scripts.train_planner_grpo import score_step_completion

    expected = json.loads(str(row["expected_step"]))
    case_stub = {"target_action_class": row.get("target_action_class")}
    completion = canonical_decision(case_stub, expected, step_index=int(row["step_index"]))
    return score_step_completion(
        completion=json.dumps(completion, ensure_ascii=False),
        expected_step=str(row["expected_step"]),
        forbidden_actions=str(row["forbidden_actions"]),
        reward_spec=str(row["reward_spec"]),
        previous_action=str(row.get("previous_action") or ""),
        full_expected_actions=str(row["full_expected_actions"]),
        step_index=int(row["step_index"]),
    )


def prompt_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    forbidden_fragments = ("/raid/", "/tmp/", "按训练样本期望", '"external_ref"', '"_thought"')
    fragment_hits = {
        fragment: sum(fragment in str(row.get("prompt") or "") for row in rows)
        for fragment in forbidden_fragments
    }
    case_id_hits = sum(
        str(row.get("case_id") or "") in str(row.get("prompt") or "")
        for row in rows
        if str(row.get("case_id") or "")
    )
    entity_id_hits = sum(
        str(row.get("entity_id") or "") in str(row.get("prompt") or "")
        for row in rows
        if str(row.get("entity_id") or "")
    )
    max_steps_mismatch = sum('"max_steps": 3' not in str(row.get("prompt") or "") for row in rows)
    duplicate_prompt_hashes = len(rows) - len({str(row.get("prompt_sha256") or "") for row in rows})
    reward_scores = [_canonical_score_for_stage_row(row) for row in rows]
    return {
        "status": "pass"
        if not any(fragment_hits.values())
        and case_id_hits == 0
        and entity_id_hits == 0
        and max_steps_mismatch == 0
        and duplicate_prompt_hashes == 0
        and all(abs(score - 1.0) <= 1e-12 for score in reward_scores)
        else "fail",
        "rows": len(rows),
        "forbidden_fragment_hits": fragment_hits,
        "case_id_hits": case_id_hits,
        "entity_id_hits": entity_id_hits,
        "max_steps_mismatch": max_steps_mismatch,
        "duplicate_prompt_hashes": duplicate_prompt_hashes,
        "canonical_step_reward": {
            "min": min(reward_scores) if reward_scores else None,
            "max": max(reward_scores) if reward_scores else None,
            "non_unit": sum(abs(score - 1.0) > 1e-12 for score in reward_scores),
        },
        "prompt_tokens": summarize_numbers([int(row["prompt_token_count"]) for row in rows]),
    }


def summarize_numbers(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "mean": None, "p50": None, "p95": None, "max": None}
    ordered = sorted(values)
    percentile = lambda q: ordered[min(len(ordered) - 1, int((len(ordered) - 1) * q))]
    return {
        "min": min(ordered),
        "mean": statistics.fmean(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": max(ordered),
    }


def summarize_case_split(split: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    core = [row for row in rows if row.get("guardrail") is False]
    guards = [row for row in rows if row.get("guardrail") is True]
    all_steps = sum(len(row.get("expected_decisions") or []) for row in rows)
    target_step_actions = []
    terminal_actions = []
    for row in rows:
        expected = row.get("expected_decisions") if isinstance(row.get("expected_decisions"), list) else []
        if len(expected) >= 2 and isinstance(expected[1], dict):
            target_step_actions.append(expected_action_name(expected[1]))
        if expected and isinstance(expected[-1], dict):
            terminal_actions.append(expected_action_name(expected[-1]))
    return {
        "split": split,
        "role": ROLE_SPECS[split]["selection_role"],
        "cases": len(rows),
        "entities": len({str(row.get("entity_id") or "") for row in rows}),
        "counterfactual_bundles": len({str(row.get("counterfactual_bundle_id") or "") for row in core}),
        "core_cases": len(core),
        "guardrail_cases": len(guards),
        "guardrail_fraction": len(guards) / len(core) if core else 0.0,
        "expected_decision_rows_before_sft_dedup": all_steps,
        "three_step_cases": sum(len(row.get("expected_decisions") or []) == 3 for row in rows),
        "second_observation_cases": sum(len(row.get("mock_observations") or []) == 2 for row in rows),
        "target_action_classes": _counter(row.get("target_action_class") for row in rows),
        "core_target_action_classes": _counter(row.get("target_action_class") for row in core),
        "target_step_actions": _counter(target_step_actions),
        "terminal_actions": _counter(terminal_actions),
        "detector_families": _counter(row.get("detector_family") for row in rows),
        "badge_conditions_core": _counter(row.get("badge_condition") for row in core),
        "error_aliases_core": _counter(row.get("error_alias") for row in core),
        "guard_scenarios": _counter(row.get("scenario_id") for row in guards),
        "post_retry_outcomes": _counter(
            row.get("post_retry_outcome")
            for row in core
            if row.get("scenario_id") == "core_retryable_fresh"
        ),
        "templates": len({str(row.get("template_id") or "") for row in rows}),
        "fixture_families": _counter(row.get("image_fixture_family") for row in rows),
        "badge_action_mutual_information_bits_core": mutual_information(
            core, "badge_condition", "target_action_class"
        ),
        "alias_action_mutual_information_bits_core": mutual_information(
            core, "error_alias", "target_action_class"
        ),
        "canonical_full_trajectory_pass": sum(bool(score_case(row)["passed"]) for row in rows),
        "independent_oracle_transitions_pass": sum(
            independent_oracle_action(str((item.get("observation") or {}).get("summary") or ""))
            == expected_action_class(
                row["expected_decisions"][int(item["after_step"])],
                detector_action=expected_action_name(row["expected_decisions"][0]),
            )
            for row in rows
            for item in row["mock_observations"]
        ),
        "independent_oracle_transitions_total": sum(len(row["mock_observations"]) for row in rows),
    }


def build_human_review_sample(cases_by_split: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    sample: list[dict[str, Any]] = []
    for split, rows in cases_by_split.items():
        by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_scenario[str(row.get("scenario_id") or "")].append(row)
        for scenario, candidates in sorted(by_scenario.items()):
            selected = sorted(candidates, key=lambda row: str(row.get("case_id") or ""))[0]
            sample.append(
                {
                    "case_id": selected["case_id"],
                    "split": split,
                    "scenario_id": scenario,
                    "entity_id": selected["entity_id"],
                    "detector_family": selected["detector_family"],
                    "target_action_class": selected["target_action_class"],
                    "user_query": selected["user_query"],
                    "mock_observations": selected["mock_observations"],
                    "expected_actions": [
                        expected_action_name(step)
                        for step in selected["expected_decisions"]
                        if isinstance(step, dict)
                    ],
                    "review_status": "pending_independent_human_review",
                    "review_checks": {
                        "policy_label_correct": None,
                        "observation_has_no_action_hint": None,
                        "counterfactual_isolation_valid": None,
                        "language_natural_enough": None,
                    },
                }
            )
    return sample


def _load_tokenizer(model_path: Path) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=False,
        use_fast=True,
        padding_side="left",
    )
    if tokenizer.eos_token_id != EXPECTED_EOS_ID or tokenizer.pad_token_id != EXPECTED_PAD_ID:
        raise ValueError(
            "Qwen3.5 tokenizer stop contract mismatch: "
            f"eos={tokenizer.eos_token_id}, pad={tokenizer.pad_token_id}"
        )
    return tokenizer


def write_stage_data(
    cases_by_split: dict[str, list[dict[str, Any]]],
    *,
    model_path: Path,
    sft_dir: Path,
    step_dir: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    tokenizer = _load_tokenizer(model_path)
    outputs: dict[str, Path] = {}
    sft_train_rows = build_sft_rows(cases_by_split["sft_train"], tokenizer)
    sft_dev_rows = build_sft_rows(cases_by_split["sft_dev"], tokenizer)
    sft_train_path = sft_dir / "train.jsonl"
    sft_dev_path = sft_dir / "dev.jsonl"
    write_jsonl(sft_train_path, sft_train_rows)
    write_jsonl(sft_dev_path, sft_dev_rows)
    outputs["sft_train"] = sft_train_path
    outputs["sft_dev"] = sft_dev_path

    grpo_train_rows = build_grpo_rows(cases_by_split["grpo_train"], tokenizer)
    grpo_dev_rows = build_grpo_rows(cases_by_split["grpo_dev"], tokenizer)
    grpo_train_path = step_dir / f"{DATASET_ID}_grpo_train_qwen35_4b_nothinking_step2.jsonl"
    grpo_dev_path = step_dir / f"{DATASET_ID}_grpo_dev_qwen35_4b_nothinking_step2.jsonl"
    write_jsonl(grpo_train_path, grpo_train_rows)
    write_jsonl(grpo_dev_path, grpo_dev_rows)
    outputs["grpo_train"] = grpo_train_path
    outputs["grpo_dev"] = grpo_dev_path

    row_sets = {
        "sft_train": sft_train_rows,
        "sft_dev": sft_dev_rows,
        "grpo_train": grpo_train_rows,
        "grpo_dev": grpo_dev_rows,
    }
    audits = {name: prompt_audit(rows) for name, rows in row_sets.items()}
    failed = {name: report for name, report in audits.items() if report["status"] != "pass"}
    if failed:
        raise ValueError(f"stage prompt/reward audit failed: {json.dumps(failed, ensure_ascii=False)[:4000]}")

    metadata = {
        "schema_version": "1.0",
        "created_at": CREATED_AT,
        "dataset_id": DATASET_ID,
        "model_name_or_path": str(model_path),
        "prompt_contract": {
            "chat_template": "native_qwen35",
            "enable_thinking": False,
            "assistant_suffix": NONTHINKING_SUFFIX,
            "completion_suffix_sft": "<|im_end|>",
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "max_steps": 3,
            "sanitized_context": True,
        },
        "rows": {name: len(rows) for name, rows in row_sets.items()},
        "deduplication": {
            "sft_train_source_decisions": sum(
                len(case["expected_decisions"]) for case in cases_by_split["sft_train"]
            ),
            "sft_train_rows": len(sft_train_rows),
            "sft_dev_source_decisions": sum(
                len(case["expected_decisions"]) for case in cases_by_split["sft_dev"]
            ),
            "sft_dev_rows": len(sft_dev_rows),
            "rule": "deduplicate identical prompt+completion pairs created by matched core step-1 probes",
        },
        "audits": audits,
        "files": {name: str(path.relative_to(ROOT)) for name, path in outputs.items()},
        "sha256": {name: sha256_file(path) for name, path in outputs.items()},
        "model_contract_sha256": {
            name: sha256_file(model_path / filename)
            for name, filename in {
                "config": "config.json",
                "tokenizer_config": "tokenizer_config.json",
                "chat_template": "chat_template.jinja",
            }.items()
        },
    }
    metadata_path = sft_dir / "metadata.json"
    write_json(metadata_path, metadata)
    outputs["stage_metadata"] = metadata_path
    return metadata, outputs


def build_eda_summary(
    cases_by_split: dict[str, list[dict[str, Any]]],
    *,
    validation: dict[str, Any],
    stage_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    split_summaries = {
        split: summarize_case_split(split, rows) for split, rows in cases_by_split.items()
    }
    all_rows = [row for rows in cases_by_split.values() for row in rows]
    return {
        "schema_version": "1.0",
        "generated_at": CREATED_AT,
        "dataset_id": DATASET_ID,
        "format": "JSONL with one case object per line; nested expected decisions and observations",
        "dimensions": {
            "splits": len(cases_by_split),
            "cases": len(all_rows),
            "entities": len({str(row["entity_id"]) for row in all_rows}),
            "core_cases": sum(row.get("guardrail") is False for row in all_rows),
            "guardrail_cases": sum(row.get("guardrail") is True for row in all_rows),
        },
        "splits": split_summaries,
        "quality": {
            "validation_status": validation["status"],
            "validation_errors": validation["errors"],
            "split_isolation_status": validation["split_isolation"]["status"],
            "repository_isolation_status": validation["repository_isolation"]["status"],
            "missing_required_case_keys": 0,
            "duplicate_case_ids": len(all_rows) - len({str(row["case_id"]) for row in all_rows}),
            "canonical_full_trajectory_failures": sum(
                not bool(score_case(row)["passed"]) for row in all_rows
            ),
        },
        "stage_data": stage_metadata or {"generated": False},
        "analysis_parameters": {
            "seed": SEED,
            "core_states": list(CORE_STATES),
            "guardrail_ratio_to_core": 0.25,
            "experimental_unit": "entity × detector × error_alias counterfactual bundle",
            "nuisance_blocking": ["query", "entity", "detector", "error_alias", "badge", "fixture"],
        },
    }


def build_manifest(
    cases_by_split: dict[str, list[dict[str, Any]]],
    *,
    case_paths: dict[str, Path],
    fixture_paths: list[Path],
    validation: dict[str, Any],
    stage_metadata: dict[str, Any] | None,
    stage_paths: dict[str, Path],
    review_path: Path,
    eda_summary_path: Path,
    audit_path: Path,
) -> dict[str, Any]:
    files: dict[str, Path] = {
        "builder": Path(__file__).resolve(),
        "case_reward_scorer": ROOT / "training/planner_grpo_seed_v1/scripts/reward_planner_grpo.py",
        "step_reward_scorer": ROOT / "training/planner_grpo_seed_v1/scripts/train_planner_grpo.py",
        **{f"cases_{split}": path for split, path in case_paths.items()},
        **{f"fixture_{index + 1:02d}": path for index, path in enumerate(fixture_paths)},
        **stage_paths,
        "human_review_sample": review_path,
        "eda_summary": eda_summary_path,
        "audit_report": audit_path,
    }
    for name, filename in {
        "readme": "README.md",
        "dataset_card": "DATASET_CARD.md",
        "eda_report": "EDA_REPORT.md",
        "human_review_protocol": "HUMAN_REVIEW.md",
    }.items():
        path = review_path.parent / filename
        if path.is_file():
            files[name] = path
    return {
        "schema_version": "1.0",
        "created_at": CREATED_AT,
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "status": "frozen" if validation["status"] == "pass" else "invalid",
        "purpose": "SFT warm-up, GRPO transition optimization, development selection, and entity-disjoint final evaluation",
        "experimental_unit": "entity × detector × error_alias matched counterfactual bundle",
        "construction": {
            "copied_v5_rows": 0,
            "core_states_per_bundle": list(CORE_STATES),
            "guardrail_cases_per_core_case": 0.25,
            "fixed_max_steps": 3,
            "grpo_target_step": 2,
            "split_entity_counts": {
                split: int(spec["entities"]) for split, spec in ROLE_SPECS.items()
            },
        },
        "stats": {
            split: summarize_case_split(split, rows) for split, rows in cases_by_split.items()
        },
        "integrity": {
            "validation_status": validation["status"],
            "split_isolation": validation["split_isolation"],
            "repository_isolation": validation["repository_isolation"],
            "stage_prompt_and_reward_audits": (stage_metadata or {}).get("audits", {}),
            "independent_human_review_status": "pending",
        },
        "files": {name: str(path.relative_to(ROOT)) for name, path in files.items()},
        "sha256": {name: sha256_file(path) for name, path in files.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, default=CASE_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--sft-dir", type=Path, default=SFT_DIR)
    parser.add_argument("--step-dir", type=Path, default=STEP_DIR)
    parser.add_argument("--model-name-or-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--skip-stage-data", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_dir = args.case_dir if args.case_dir.is_absolute() else ROOT / args.case_dir
    dataset_dir = args.dataset_dir if args.dataset_dir.is_absolute() else ROOT / args.dataset_dir
    sft_dir = args.sft_dir if args.sft_dir.is_absolute() else ROOT / args.sft_dir
    step_dir = args.step_dir if args.step_dir.is_absolute() else ROOT / args.step_dir
    model_path = args.model_name_or_path.resolve()

    fixture_paths = write_fixture_images()
    cases_by_split = build_all_cases()
    validation = validate_all_cases(cases_by_split)
    if validation["status"] != "pass":
        raise ValueError("case validation failed:\n" + "\n".join(validation["errors"][:100]))

    case_paths: dict[str, Path] = {}
    for split, rows in cases_by_split.items():
        path = case_dir / f"{DATASET_ID}_{split}_cases.jsonl"
        write_jsonl(path, rows)
        if len(load_jsonl(path)) != len(rows):
            raise ValueError(f"round-trip row count mismatch: {path}")
        case_paths[split] = path

    stage_metadata: dict[str, Any] | None = None
    stage_paths: dict[str, Path] = {}
    if not args.skip_stage_data:
        stage_metadata, stage_paths = write_stage_data(
            cases_by_split,
            model_path=model_path,
            sft_dir=sft_dir,
            step_dir=step_dir,
        )

    dataset_dir.mkdir(parents=True, exist_ok=True)
    review_path = dataset_dir / "human_review_sample.jsonl"
    write_jsonl(review_path, build_human_review_sample(cases_by_split))

    audit_path = dataset_dir / "audit_report.json"
    audit_payload = {
        "schema_version": "1.0",
        "created_at": CREATED_AT,
        "dataset_id": DATASET_ID,
        "validation": validation,
        "split_summaries": {
            split: summarize_case_split(split, rows) for split, rows in cases_by_split.items()
        },
        "stage_audits": (stage_metadata or {}).get("audits", {}),
    }
    write_json(audit_path, audit_payload)

    eda_summary_path = dataset_dir / "eda_summary.json"
    write_json(
        eda_summary_path,
        build_eda_summary(
            cases_by_split,
            validation=validation,
            stage_metadata=stage_metadata,
        ),
    )

    manifest = build_manifest(
        cases_by_split,
        case_paths=case_paths,
        fixture_paths=fixture_paths,
        validation=validation,
        stage_metadata=stage_metadata,
        stage_paths=stage_paths,
        review_path=review_path,
        eda_summary_path=eda_summary_path,
        audit_path=audit_path,
    )
    manifest_path = dataset_dir / "manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
