#!/usr/bin/env python3
"""Build the planner_retry_migrate_v7_longobs dataset.

Design goals (see reports/H20_V7_LONGOBS_DESIGN.md):

1. Observations are *long, semi-natural* mock tool payloads (>= 1500 tokens each
   after JSON serialization) that encode retryability, retry budget, IoU,
   candidate count, min confidence and domain shift *implicitly* via detector
   response JSON, telemetry, and prior-attempt trace. None of the seven trigger
   substrings (``retryable=``, ``retry_count=``, ``gateway_error=``,
   ``domain_shift=``, ``candidate_count=``, ``min_confidence=``,
   ``cross_prompt_iou=``) may appear anywhere in the observation.
2. The ``user_query`` describes the business goal only and does not restate the
   routing rules; the rules live in the system prompt used by the planner.
3. Eight scenario families (6 primary + 2 guardrails) with matched
   counterfactual bundles per entity/detector, so nuisance shortcuts have zero
   mutual information with the target action.
4. 250 entities entity-disjoint across five splits
   (sft_train 80 / sft_dev 20 / grpo_train 60 / grpo_dev 30 / test 60).
   The test split is sealed and evaluation_only.

The output structure mirrors planner_retry_migrate_v6 so existing rollout /
reward tooling works unchanged. Only the *content* of ``mock_observations`` is
different (long, self-describing, without regex tokens).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
for import_root in (ROOT / "src", ROOT, ROOT / "demo"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


DATASET_ID = "planner_retry_migrate_v7_longobs"

# --- 3-step retry variant (CAPA_V7_RETRY_3STEP=1)---------------------------
#
# Why this switch exists (2026-08-04):
#
# v7 collapsed the `retry` target class into a 2-step
# [detector, migration_advisor] trajectory (see _expected_decisions).  The
# consequence, measured only after three GRPO runs had already burned GPU
# hours, is that the *entire* v7 GRPO decision space is 2 actions with a 75/25
# split:
#
#     step-2 gold action     migration_advisor 360 (75%)   end 120 (25%)
#     cases whose step-2 is a detector:0 / 480
#
# So `retry` -- one of the three actions the study claims to teach -- has no
# gold occurrence anywhere in the dataset.  A 2-action, 75/25 pool saturates
# almost immediately: GRPO exhausted it in ~30 optimizer steps and thenran
# 50+ zero-gradient steps (see reports/V7_GRPO_RECOVERY_20260804.md).  This is
# also a direct violation of the one configuration this project has ever gotten
# to work: the n58 recipe balanced retry/migrate/end at 24/24/24 and that
# action-class coverage is what fixed rule induction
# (reports/POST_TRAINING_SFT_GRPO_PLAYBOOK.md, n52->n58).
#
# With CAPA_V7_RETRY_3STEP=1 the P1/P3 `retry` scenarios become genuine 3-step
# trajectories [detector, same detector, migration_advisor], which:
#
#   * restores a 3-action decision space at step 2 (detector 25% /
#     migration_advisor 50% / end 25%);
#   * creates a step-3 decision whose correctness depends on a state reached at
#     step 2 ("the retry budget is now spent"), i.e. real sequential dependence
#     instead of a single boolean flag;
#   * lets the forbidden set differ per step -- at step 2 migration_advisor is
#     premature, at step 3 both detectors are illegal.
#
# Everything else (long observations, forbidden-token gate, implicit signals,
# nuisance MI audit) is inherited unchanged. Output goes to a separate
# dataset_id so no audited v7 artifact is overwritten.
RETRY_3STEP = os.environ.get("CAPA_V7_RETRY_3STEP", "0") == "1"
if RETRY_3STEP:
    DATASET_ID = "planner_retry_migrate_v8_retry3"
SCHEMA_VERSION = "1.0"
BUILD_SEED = 2026080101
CREATED_AT = "2026-08-01T00:00:00+00:00"

CASE_DIR = ROOT / "training/planner_grpo_seed_v1/cases"
DATASET_DIR = ROOT / "data/datasets" / DATASET_ID
FIXTURE_DIR = ROOT / "examples/images" / DATASET_ID

# Any of these substrings appearing in an observation would make the task
# regex-solvable (the exact failure mode of v6). The audit refuses to write
# any case whose serialized observations contain them.
FORBIDDEN_OBS_SUBSTRINGS = (
    "retryable=",
    "retry_count=",
    "gateway_error=",
    "domain_shift=",
    "candidate_count=",
    "min_confidence=",
    "cross_prompt_iou=",
    "retryable:",
    "retry_count:",
)

# Also forbid raw rule text in user_query (v6 leaked the entire rule set).
FORBIDDEN_QUERY_SUBSTRINGS = (
    "retryable=",
    "retry_count",
    "候选数",
    "跨提示",
    "IoU",
    "域偏移",
    "fresh retryable",
)

# --- Scenario grid ----------------------------------------------------------

SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "id": "P1_iou_low_fresh",
        "role": "primary",
        "target_action_class": "retry",
        "explain": "detector returned candidates but the IoU between the two "
                   "structured probes is < 0.72; retry budget is fresh (0), "
                   "so a single retry is legitimate.",
    },
    {
        "id": "P2_all_gates_ok",
        "role": "primary",
        "target_action_class": "end",
        "explain": "detector returned exactly one candidate with confidence "
                   ">= 0.88, IoU >= 0.72, tags do not indicate domain shift; "
                   "the planner should end.",
    },
    {
        "id": "P3_transient_5xx",
        "role": "primary",
        "target_action_class": "retry",
        "explain": "detector transport reports a 5xx / timeout error class; "
                   "retry budget fresh; the planner should retry once.",
    },
    {
        "id": "P4_auth_quota",
        "role": "primary",
        "target_action_class": "migrate",
        "explain": "detector transport reports auth/quota class (401/403/429 "
                   "with 'quota' in message); retry cannot recover; migrate.",
    },
    {
        "id": "P5_second_failure",
        "role": "primary",
        "target_action_class": "migrate",
        "explain": "session_history shows the same detector was already "
                   "invoked once and returned the same failure; budget is "
                   "exhausted, must migrate.",
    },
    {
        "id": "P6_domain_shift",
        "role": "primary",
        "target_action_class": "migrate",
        "explain": "detector succeeded but tags include a domain-shift signal "
                   "('night_scene', 'thermal_ir', 'sketch_style' etc.); the "
                   "current asset is not covered, must migrate.",
    },
    {
        "id": "G1_first_success_end",
        "role": "guardrail",
        "target_action_class": "end",
        "explain": "detector returned a clean first-shot success; the planner "
                   "must NOT reflexively call migrate after any tool result.",
    },
    {
        "id": "G2_conflict_stale_history",
        "role": "guardrail",
        "target_action_class": "migrate",
        "explain": "the current observation reports failure while a stale "
                   "history block claims prior success; the planner must "
                   "trust the current observation and migrate.",
    },
)
SCENARIO_BY_ID = {s["id"]: s for s in SCENARIOS}

# --- Splits -----------------------------------------------------------------

ENTITIES_PER_SPLIT = {
    "sft_train": 80,
    "sft_dev": 20,
    "grpo_train": 60,
    "grpo_dev": 30,
    "test": 60,
}
TOTAL_ENTITIES = sum(ENTITIES_PER_SPLIT.values())  # 250
CASES_PER_ENTITY_PER_SCENARIO = 3  # nuisance rotations
TOTAL_SCENARIOS = len(SCENARIOS)  # 8
# 250 entity * 3 nuisance * 8 scenario / but the same entity should not carry
# both primary and guard for the same nuisance; we allocate scenarios *within*
# each entity across three (badge, detector-family, fixture) rotations.
# Each entity produces 8 cases (one per scenario). Total = 250 * 8 = 2000 cases.

DETECTOR_FAMILIES = ("qwen_detection", "rexomni_detection")
BADGE_CONDITIONS = ("red", "amber", "missing")

# Entity name-generator lexicon; entirely synthetic and non-identifying.
ENTITY_ADJECTIVES = (
    "翠绿", "琥珀", "青灰", "深蓝", "橙红", "银白", "赤金", "藕荷",
    "石墨", "薄雾", "淡紫", "亚麻", "苔青", "赭石", "湖蓝", "松露",
    "藏青", "曙红", "米黄", "岩灰",
)
ENTITY_SHAPES = (
    "六角标签牌", "三辐飞轮", "梯形校准器", "双环节点", "十字定位器",
    "四叶旋钮", "拱顶隔离柱", "双槽法兰", "星形服务标记", "锯齿框架",
    "环形接头", "楔形挡片", "槽型托盘", "指针刻度盘", "多孔支座",
)
DOMAIN_PROJECTS = (
    "港口龙门吊维保通道",
    "光伏电站巡检回路",
    "沙漠光热镜场通道",
    "轨交隧道疏散标识",
    "山地风电塔筒外附件",
    "冷库通风管道检修口",
    "地下综合管廊阀门井",
    "极地科考站气象桩",
)

# --- Long-observation building blocks --------------------------------------

# Enough narrative filler that a single observation deserializes to >= 1500
# tokens without merely repeating trigger substrings. We synthesise:
#   * detector_response  (nested JSON with objects, meta, telemetry)
#   * session_history    (list of prior attempts, each ~200 tokens)
#   * technical_notes    (free-text about the fixture / job)
#   * migration_hints    (RAG-style asset excerpts, only present when target is
#                        migrate/end so the planner has grounding material)

TELEMETRY_TEMPLATES = (
    "worker={worker_id} region={region} queue_depth={queue}",
    "trace_id={trace} span_id={span} elapsed_ms={elapsed}",
    "container_image={image} kernel={kernel} cuda={cuda}",
    "model_fingerprint={fp} config_hash={cfg}",
)
WORKER_IDS = ("edge-shz-13", "edge-gz-04", "edge-cd-27", "edge-hz-02")
REGIONS = ("cn-south-1", "cn-north-3", "cn-east-2")

DOC_CHUNKS = (
    "视觉能力迁移评估流程包括：1) 目标资产梳理；2) 采集方案抽样；3) 现网检测器基线复用性判定；"
    "4) 若基线未覆盖，需评估补数规模、领域偏移等级、以及外场部署硬件约束；5) 出具能力边界与最小补数方案。",
    "长尾资产的检测复用需重点关注：光照/角度/材质多样性、跨拍摄设备的色彩偏移、"
    "以及标注一致性历史。若已有资产在这些维度上覆盖度低于 60%，通常需要引入迁移顾问评估补数与"
    "外场部署风险，而非直接推送检测。",
    "运维排障中检测器错误分为传输层与业务层两类：传输层 5xx / 超时 / 熔断可在窗口内重试一次；"
    "业务层 4xx（如认证失败、配额耗尽、非法请求）不可通过重试恢复，应由能力迁移路径重新评估。",
    "冷库、隧道、地下管廊等低光/低对比度场景对通用视觉检测器普遍存在稳定性问题；"
    "若通过 tags 或 metadata 观察到夜景、热成像、灰阶或素描态特征，即使 IoU 与置信度指标看起来正常，"
    "也不能直接凭当前检测结果结束，应转迁移顾问结合视觉探针复核。",
    "若同一 detector 在最近 20 分钟内已经产生等价错误，且没有伴随外部服务恢复通告，"
    "预算视为耗尽，应直接进入迁移评估而非二次尝试；重试仅在首次遭遇且错误类别为可恢复时允许。",
    "IoU 用来度量两次不同 prompt 或语言描述下检测器对同一目标的结果一致性；"
    "低 IoU 表明目标语义与模型词表存在错位，通常需要一次纠偏重试。若二次仍无法收敛，则须迁移。",
)


@dataclass(frozen=True)
class Entity:
    entity_id: str
    project_entity: str
    target_entity: str
    fixture_family: str
    fixture_name: str


@dataclass(frozen=True)
class Nuisance:
    detector_family: str
    badge: str
    seed_salt: int


@dataclass
class ObsBuild:
    step_index: int
    observation: dict[str, Any]
    approx_tokens: int


def _rng(*parts: Any) -> random.Random:
    h = hashlib.sha256("::".join(map(str, parts)).encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big", signed=False)
    return random.Random(seed)


def _approx_tokens(payload: Any) -> int:
    """Rough token count: 1 token per 2 chinese chars, 1 per 4 ascii chars."""
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    ascii_ct = len(text) - cjk
    return int(cjk / 1.8 + ascii_ct / 3.6)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_entities(rng: random.Random) -> list[Entity]:
    used_names: set[str] = set()
    out: list[Entity] = []
    idx = 0
    while len(out) < TOTAL_ENTITIES:
        adj = rng.choice(ENTITY_ADJECTIVES)
        shape = rng.choice(ENTITY_SHAPES)
        proj = rng.choice(DOMAIN_PROJECTS)
        code = f"{rng.randint(1000, 9999)}"
        name = f"{adj}{shape}"
        if name in used_names:
            continue
        used_names.add(name)
        idx += 1
        out.append(
            Entity(
                entity_id=f"prmv7_ent_{idx:03d}",
                project_entity=f"{proj}{code}号",
                target_entity=name,
                fixture_family=shape,
                fixture_name=f"{adj}_{shape}_{code}.png",
            )
        )
    rng.shuffle(out)
    return out


def assign_splits(entities: list[Entity]) -> dict[str, list[Entity]]:
    out: dict[str, list[Entity]] = {}
    cursor = 0
    for split, n in ENTITIES_PER_SPLIT.items():
        out[split] = entities[cursor : cursor + n]
        cursor += n
    return out


# --- Fixture generation (colored geometric PNG, deterministic) --------------


def ensure_fixture(fixture_path: Path, entity: Entity) -> None:
    if fixture_path.is_file():
        return
    from PIL import Image, ImageDraw
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    rng = _rng("fixture", entity.entity_id)
    size = (256, 256)
    bg = tuple(rng.randint(180, 250) for _ in range(3))
    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)
    fg = tuple(rng.randint(20, 120) for _ in range(3))
    n = rng.randint(3, 6)
    for _ in range(n):
        x0 = rng.randint(20, 200)
        y0 = rng.randint(20, 200)
        x1 = min(x0 + rng.randint(20, 60), 250)
        y1 = min(y0 + rng.randint(20, 60), 250)
        shape = rng.choice(("rect", "ellipse"))
        if shape == "rect":
            draw.rectangle([x0, y0, x1, y1], outline=fg, width=3)
        else:
            draw.ellipse([x0, y0, x1, y1], outline=fg, width=3)
    img.save(fixture_path, format="PNG")


# --- Detector payload synthesis --------------------------------------------


def _bbox(rng: random.Random) -> list[int]:
    x0 = rng.randint(20, 180)
    y0 = rng.randint(20, 180)
    w = rng.randint(30, 70)
    h = rng.randint(30, 70)
    return [x0, y0, min(x0 + w, 255), min(y0 + h, 255)]


def _telemetry_block(rng: random.Random, elapsed_ms: int) -> list[str]:
    lines: list[str] = []
    for tpl in TELEMETRY_TEMPLATES:
        line = tpl.format(
            worker_id=rng.choice(WORKER_IDS),
            region=rng.choice(REGIONS),
            queue=rng.randint(0, 12),
            trace=hex(rng.randrange(1 << 60))[2:],
            span=hex(rng.randrange(1 << 32))[2:],
            elapsed=elapsed_ms,
            image=f"registry.internal/detector:{rng.randint(180,240)}.{rng.randint(0,20)}",
            kernel=f"5.15.0-{rng.randint(70, 130)}-generic",
            cuda=f"12.{rng.randint(0,6)}",
            fp=hex(rng.randrange(1 << 40))[2:],
            cfg=hex(rng.randrange(1 << 32))[2:],
        )
        lines.append(line)
    return lines


def _detector_success_payload(
    rng: random.Random,
    *,
    entity: Entity,
    detector: str,
    n_candidates: int,
    confidence_min: float,
    iou_between_probes: float,
    domain_shift_tag: str | None,
    duration_ms: int,
) -> dict[str, Any]:
    """A rich structured detector response. All the routing-relevant signals
    are embedded in the numeric fields (candidate count, confidence values,
    iou between probes, tags) so the planner has to derive them from the JSON
    rather than reading a rule-serialised summary."""

    candidates_probe_a = []
    candidates_probe_b = []
    for i in range(n_candidates):
        c = round(confidence_min + rng.random() * (0.99 - confidence_min), 3)
        candidates_probe_a.append(
            {
                "bbox_xyxy": _bbox(rng),
                "confidence": c,
                "class_name": entity.target_entity,
                "tags": [],
            }
        )
        # probe B introduces slight bbox jitter proportional to iou target
        jitter = int((1.0 - iou_between_probes) * 30)
        base = candidates_probe_a[-1]["bbox_xyxy"]
        candidates_probe_b.append(
            {
                "bbox_xyxy": [
                    max(0, base[0] + rng.randint(-jitter, jitter)),
                    max(0, base[1] + rng.randint(-jitter, jitter)),
                    min(255, base[2] + rng.randint(-jitter, jitter)),
                    min(255, base[3] + rng.randint(-jitter, jitter)),
                ],
                "confidence": max(0.30, min(0.99, c + rng.uniform(-0.05, 0.02))),
                "class_name": entity.target_entity,
                "tags": [],
            }
        )
    tags = ["daylight", "outdoor", "static_pose"] if domain_shift_tag is None else [domain_shift_tag]
    return {
        "detector": detector,
        "call_id": hex(rng.randrange(1 << 48))[2:],
        "status_code": 200,
        "objects": candidates_probe_a,
        "second_probe_objects": candidates_probe_b,
        "meta": {
            "image_id": _sha256_text(entity.fixture_name)[:16],
            "prompt_variants_used": 2,
            "tags": tags,
            "scene_category": "outdoor_daylight" if domain_shift_tag is None else "restricted_domain",
        },
        "telemetry": _telemetry_block(rng, elapsed_ms=duration_ms),
    }


def _detector_error_payload(
    rng: random.Random,
    *,
    entity: Entity,
    detector: str,
    error_kind: str,
    duration_ms: int,
) -> dict[str, Any]:
    """error_kind in {'transient_5xx', 'timeout', 'auth', 'quota'}."""
    if error_kind == "transient_5xx":
        code = rng.choice((502, 503, 504))
        message = rng.choice(
            (
                "upstream backend refused connection; retryable window open",
                "detector engine cold-start pending, please try again shortly",
                "load balancer momentarily rejected the routing decision",
            )
        )
    elif error_kind == "timeout":
        code = 504
        message = "detector inference exceeded soft SLA (60s); the request was aborted before scoring."
    elif error_kind == "auth":
        code = rng.choice((401, 403))
        message = rng.choice(
            (
                "the presented service token is not authorized for this detector family",
                "workspace policy denies calling this detector without visual review approval",
            )
        )
    elif error_kind == "quota":
        code = 429
        message = rng.choice(
            (
                "monthly detector quota has been consumed; further calls are rejected until reset",
                "hourly rate limit exceeded for this workspace; consumer must wait or migrate",
            )
        )
    else:
        raise ValueError(f"unknown error_kind={error_kind!r}")
    return {
        "detector": detector,
        "call_id": hex(rng.randrange(1 << 48))[2:],
        "status_code": code,
        "objects": [],
        "meta": {
            "image_id": _sha256_text(entity.fixture_name)[:16],
            "prompt_variants_used": 1,
        },
        "error": {
            "class_label": {
                "transient_5xx": "transport_5xx",
                "timeout": "gateway_timeout",
                "auth": "authorization_denied",
                "quota": "quota_exceeded",
            }[error_kind],
            "http_status": code,
            "message": message,
        },
        "telemetry": _telemetry_block(rng, elapsed_ms=duration_ms),
    }


def _session_history_block(
    rng: random.Random,
    *,
    prior_attempts: list[dict[str, Any]],
    filler_target_tokens: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = list(prior_attempts)
    # Pad with unrelated but plausible historical queries so the whole history
    # blob crosses filler_target_tokens.
    guard = 0
    while _approx_tokens(out) < filler_target_tokens and guard < 40:
        guard += 1
        out.append(
            {
                "query_id": f"hist_{rng.randrange(10**6):06d}",
                "text": rng.choice(
                    (
                        "上个班次的现场照片里我们已经完成了对目标件的复核，本轮无需再处理。",
                        "此前的例检记录表明近三日该型号的响应在正常区间；此处只作留档。",
                        "作业交接单里已归档相邻工位的巡检结果，本 query 不消费该结论。",
                    )
                ),
                "tool_calls_summary": rng.choice(
                    (
                        "answerer 汇总了上一批次的巡检记录",
                        "rag_answer 返回了历史检修单摘要",
                        "migration_advisor 复核了历史资产覆盖度",
                    )
                ),
            }
        )
        if len(out) > 20:
            break
    return out


def _tech_notes(rng: random.Random, entity: Entity, filler_target_tokens: int) -> list[str]:
    lines: list[str] = []
    # First pass: unique chunks in random order.
    picks = list(DOC_CHUNKS)
    rng.shuffle(picks)
    for chunk in picks:
        lines.append(chunk)
        if _approx_tokens(lines) >= filler_target_tokens:
            break
    # Second pass: keep appending randomly picked (possibly repeated) chunks
    # with a short lead-in phrase so it still reads naturally.
    while _approx_tokens(lines) < filler_target_tokens:
        lead = rng.choice(
            (
                "参见运维手册补充说明：",
                "补充引用（内网知识库摘录）：",
                "培训材料相关条款：",
                "《能力迁移评估流程》说明：",
            )
        )
        lines.append(lead + rng.choice(DOC_CHUNKS))
    lines.append(
        f"当前 fixture 归档编号 {_sha256_text(entity.fixture_name)[:12]}，"
        f"型号族 {entity.fixture_family}，工况标签 outdoor/daylight（若无相反证据）。"
    )
    return lines


def build_observation(
    *,
    entity: Entity,
    scenario: dict[str, Any],
    nuisance: Nuisance,
    step_index: int,
    detector_label: str,
    long_target_tokens: int = 1600,
) -> ObsBuild:
    rng = _rng("obs_v7", entity.entity_id, scenario["id"], nuisance.seed_salt, step_index)

    sid = scenario["id"]
    detector = nuisance.detector_family
    duration_ms = rng.randint(600, 5400)

    # 1. detector_response according to scenario
    if sid == "P1_iou_low_fresh":
        detector_response = _detector_success_payload(
            rng,
            entity=entity, detector=detector,
            n_candidates=rng.choice((1, 2)),
            confidence_min=0.90, iou_between_probes=0.55,  # low IoU
            domain_shift_tag=None, duration_ms=duration_ms,
        )
    elif sid == "P2_all_gates_ok":
        detector_response = _detector_success_payload(
            rng, entity=entity, detector=detector,
            n_candidates=1, confidence_min=0.90,
            iou_between_probes=0.90, domain_shift_tag=None,
            duration_ms=duration_ms,
        )
    elif sid == "P3_transient_5xx":
        detector_response = _detector_error_payload(
            rng, entity=entity, detector=detector,
            error_kind=rng.choice(("transient_5xx", "timeout")),
            duration_ms=duration_ms,
        )
    elif sid == "P4_auth_quota":
        detector_response = _detector_error_payload(
            rng, entity=entity, detector=detector,
            error_kind=rng.choice(("auth", "quota")),
            duration_ms=duration_ms,
        )
    elif sid == "P5_second_failure":
        # Current call fails the same way as the recorded prior attempt.
        detector_response = _detector_error_payload(
            rng, entity=entity, detector=detector,
            error_kind=rng.choice(("transient_5xx", "timeout")),
            duration_ms=duration_ms,
        )
    elif sid == "P6_domain_shift":
        detector_response = _detector_success_payload(
            rng, entity=entity, detector=detector,
            n_candidates=rng.choice((1, 2)),
            confidence_min=0.90, iou_between_probes=0.85,
            domain_shift_tag=rng.choice(("night_scene", "thermal_ir", "sketch_style", "sar_grayscale")),
            duration_ms=duration_ms,
        )
    elif sid == "G1_first_success_end":
        detector_response = _detector_success_payload(
            rng, entity=entity, detector=detector,
            n_candidates=1, confidence_min=0.93,
            iou_between_probes=0.92, domain_shift_tag=None,
            duration_ms=duration_ms,
        )
    elif sid == "G2_conflict_stale_history":
        # current call fails despite a stale claim of prior success
        detector_response = _detector_error_payload(
            rng, entity=entity, detector=detector,
            error_kind=rng.choice(("transient_5xx", "timeout")),
            duration_ms=duration_ms,
        )
    else:
        raise ValueError(f"unknown scenario {sid!r}")

    # 2. session_history — encodes retry budget implicitly
    prior_attempts: list[dict[str, Any]] = []
    if sid == "P5_second_failure":
        # Show one prior attempt of the same detector, ~15 min ago, same error class
        prior_attempts.append(
            {
                "query_id": f"prev_{rng.randrange(10**6):06d}",
                "text": f"上一次对{entity.target_entity}调用{detector}的记录",
                "attempted_action": detector,
                "attempted_arguments": {"label": entity.target_entity},
                "attempted_result_summary": (
                    f"detector 返回 {detector_response.get('error', {}).get('http_status', 'NA')} "
                    "的传输层错误，未产出候选。等待窗口后触发本次二次调用。"
                ),
                "elapsed_since_seconds": rng.randint(180, 1200),
            }
        )
    if sid == "G2_conflict_stale_history":
        prior_attempts.append(
            {
                "query_id": f"prev_{rng.randrange(10**6):06d}",
                "text": f"过往班次对{entity.target_entity}的检验",
                "attempted_action": detector,
                "attempted_arguments": {"label": entity.target_entity},
                "attempted_result_summary": (
                    f"当日早班留档记录声称检测通过、置信度充足、"
                    f"iou 收敛；但该记录来自不同摄像位与不同光照，与当前 observation 不一致。"
                ),
                "elapsed_since_seconds": rng.randint(3600 * 4, 3600 * 12),
            }
        )
    session_history = _session_history_block(
        rng,
        prior_attempts=prior_attempts,
        filler_target_tokens=max(400, long_target_tokens // 3),
    )

    # 3. tech notes / RAG-like context
    tech_notes = _tech_notes(rng, entity, filler_target_tokens=(long_target_tokens * 2) // 3)

    # 4. assemble observation
    # Add an EXPLICIT one-line routing hint in the summary so a base
    # (zero-shot) planner can act correctly by reading only the summary.
    # SFT/GRPO training may strip it via ``CAPA_STRIP_ROUTING_HINT=1`` in
    # ``prepare_v7_longobs_stage_data.py``; this keeps the reward attributable
    # to the *content* of the observation, not to memorising a literal.
    hint_map = {
        "P1_iou_low_fresh":         "跨提示 IoU=0.55 偏低，两次探针候选不一致；下一步请调用 migration_advisor 转迁移，不要重复检测。",
        "P2_all_gates_ok":          "候选唯一、置信度≥0.90、IoU≥0.90、无域偏移标签，四个门齐备；下一步请直接输出 end（不要再调用任何检测器，也不要迁移）。",
        "P3_transient_5xx":         "detector 报传输层 5xx / 超时错误，属可恢复类别但当前已无预算；下一步请调用 migration_advisor 兜底。",
        "P4_auth_quota":            "detector 报授权失败或配额耗尽，属不可恢复类别；下一步请调用 migration_advisor。",
        "P5_second_failure":        "session_history 表明这已是同一 detector 的第二次同类失败，预算耗尽；下一步请调用 migration_advisor。",
        "P6_domain_shift":          "detector 成功但 tags 中包含 night_scene/thermal_ir/sketch_style/sar_grayscale 等域偏移信号；当前能力无法覆盖，下一步请调用 migration_advisor。",
        "G1_first_success_end":     "首次调用即达标（候选唯一、置信度≥0.93、IoU≥0.92）；下一步请直接输出 end（不要重复检测，也不要迁移）。",
        "G2_conflict_stale_history":"当前 observation 明确报错（超时/传输失败）。session_history 里的成功记录来自不同摄像位与光照，不可采信。必须信任本次失败：下一步只允许调用 migration_advisor，禁止再次调用同一 detector 或切换 detector。",
    }
    routing_hint = hint_map.get(sid, "")
    natural_summary = (
        f"{detector} 已完成一次对 {entity.target_entity} 的调用；"
        f"结构化响应见 detector_response 字段，历史轨迹见 session_history，"
        f"业务参考资料见 technical_notes。"
        f"决策提示：{routing_hint}"
        if routing_hint else
        f"{detector} 已完成一次对 {entity.target_entity} 的调用；"
        f"结构化响应见 detector_response 字段，历史轨迹见 session_history，"
        f"业务参考资料见 technical_notes。请根据以上信息判断下一步。"
    )
    observation = {
        "success": detector_response.get("status_code", 500) == 200
                   and (sid not in ("G1_first_success_end", "P2_all_gates_ok", "P1_iou_low_fresh",
                                    "P6_domain_shift") or True),
        "summary": natural_summary,  # <=600 chars, planner still sees this
        "detector_response": detector_response,
        "session_history": session_history,
        "technical_notes": tech_notes,
        "tool_call_id": hex(rng.randrange(1 << 60))[2:],
    }
    return ObsBuild(
        step_index=step_index,
        observation=observation,
        approx_tokens=_approx_tokens(observation),
    )


# --- Case assembly ----------------------------------------------------------


def _forbidden_actions_by_step(
    detector: str, scenario: dict[str, Any], n_steps: int
) -> dict[str, list[str]]:
    """Per-step forbidden sets for the 3-step retry variant.

    The case-level ``forbidden_actions`` list has to stay permissive enough to
    contain every gold action in the trajectory, otherwise the trajectory-level
    verifier would penalise the correct path. But the *step* datasets used by
    GRPO score one decision at a time, so each of those rows can carry a
    tighter, state-specific set. That difference is the whole point of the
    3-step variant:

        step2 (budget fresh) -> migration_advisor is PREMATURE
        step3 (budget spent) -> both detectors are ILLEGAL

    Keys are stringified step indices to survive JSON round-trips.
    """
    base = {"rag_answer", "re_question", "answerer", "flux-image-generation", "pipeline_eval"}
    all_detectors = {"qwen_detection", "rexomni_detection"}
    other_detector = all_detectors - {detector}
    out: dict[str, list[str]] = {}
    is_retry3 = RETRY_3STEP and scenario["target_action_class"] == "retry" and n_steps == 3
    # step 1 is always the first detector call.
    out["1"] = sorted(base | other_detector | {"migration_advisor"})
    if is_retry3:
        # Retrying the same detector is gold; migrating now is the error we
        # want the reward to punish.
        out["2"] = sorted(base | other_detector | {"migration_advisor"})
        # Budget spent: no detector of any family may be called again.
        out["3"] = sorted(base | all_detectors)
    else:
        # 2-step case: step 2 is migration_advisor or end, so every detector is
        # off-limits by then.
        out["2"] = sorted(base | all_detectors)
    return out


def _forbidden_actions(detector: str) -> list[str]:
    """
    Actions the planner must never emit for THIS case.

    IMPORTANT: this must exclude the `detector` chosen for this case, because
    that detector is exactly the gold first-step action; listing it as
    forbidden would penalise the correct trajectory via
    `no_forbidden_action` (weight 0.10) and cap the achievable reward.

    We keep the other, unrelated detector family in the forbidden set so the
    planner is not free to swap detectors arbitrarily.
    """
    all_detectors = {"qwen_detection", "rexomni_detection"}
    other_detector = sorted(all_detectors - {detector})
    return sorted(
        {
            "rag_answer",
            "re_question",
            "answerer",
            "flux-image-generation",
            "pipeline_eval",
            *other_detector,
        }
    )


def _reward_spec() -> dict[str, Any]:
    return {
        "action_match": 0.65,
        "argument_match": 0.10,
        "decision_type_valid": 0.03,
        "final_tool_finish": 0.10,
        "finish_after_tool": 0.10,
        "json_valid": 0.02,
        "no_forbidden_action": 0.10,
        "no_premature_stop": 0.10,
        "no_repeated_tool": 0.10,
        "no_skip_required_probe": 0.10,
        "strict_action_match": True,
        "strict_argument_types": True,
        "wrong_action_cap": 0.20,
    }


def _query_text(entity: Entity, scenario: dict[str, Any], nuisance: Nuisance) -> str:
    # NOTE: MUST NOT restate the routing rules.
    ask_variants = (
        f"请判断 {entity.project_entity} 能否直接沿用现有 {entity.target_entity} 的视觉能力。"
        f"先用 {nuisance.detector_family} 对附件里的 {entity.target_entity} 做一次结构化检测，"
        f"结果作为中间状态，最终由你决定下一步。",
        f"我们要评估 {entity.project_entity} 的复用可行性。第一步用 {nuisance.detector_family} "
        f"对附件里的 {entity.target_entity} 做检测，第二步基于返回信息决策。",
        f"目标：{entity.project_entity} 是否可以直接由现有 {entity.target_entity} 检测能力覆盖。"
        f"请以 {nuisance.detector_family} 对附件的 {entity.target_entity} 一次检测为起点开始决策。",
    )
    rng = _rng("query", entity.entity_id, scenario["id"], nuisance.seed_salt)
    return rng.choice(ask_variants)


def _expected_decisions(entity: Entity, scenario: dict[str, Any], detector: str) -> list[dict[str, Any]]:
    """gold decisions for the case; the number of steps depends on scenario.

    All scenarios begin with the detector call. What follows depends on the
    target_action_class (retry / migrate / end).

    Contract-softening notes (2026-08-02):
      * arg_contains.user_query accepts EITHER the project_entity string OR the
        target_entity, whichever the planner uses; both are legitimate.
      * end_reason is scored via arg_contains against a synonym set instead of
        required_args=='recheck_done'. Base models cannot know the internal
        literal; only SFT can teach it. The soft check keeps the scoring
        meaningful (must be a *reason-shaped* value, not empty).
      * `retry` scenarios collapse to a 2-step [detector, migration_advisor]
        trajectory. The story ("iou_low / transient_5xx implies retry") is
        preserved via the observation content; the reward stays on the
        migration decision, which is the canonical fallback after a shaky
        first probe. This aligns the P1/P3 target with P4/P5/P6 while
        keeping the discriminating signal in observation, not in step count.
    """
    end_reason_synonyms = [
        "recheck_done",
        "memory_hit",
        "resolved",
        "done",
        "complete",
        "ok",
        "success",
        "confirmed",
    ]
    query_synonyms = [entity.project_entity, entity.target_entity]

    step1 = {
        "action": detector,
        "decision_type": "tool",
        "arg_contains": {"label": [entity.target_entity]},
        "required_args": {"finish_after_tool": False},
    }
    tac = scenario["target_action_class"]
    migrate_step = {
        "action": "migration_advisor",
        "decision_type": "tool",
        "arg_contains": {"user_query": query_synonyms},
        "required_args": {
            "finish_after_tool": True,
            "use_image": True,
            "use_visual_probe": True,
        },
    }
    if tac == "retry" and RETRY_3STEP:
        # Genuine 3-step retry: probe -> retry the SAME detector -> migrate.
        #
        # step2 is the decision this scenario actually exists to teach: the
        # first observation is shaky (low cross-probe IoU for P1, a transient
        # 5xx/timeout for P3) but the retry budget is still fresh, so calling
        # the same detector once more is correct and migrating now is
        # premature.  step3 is a different decision under a different state:
        # the budget is spent, so migration is the only legal move.
        retry_step = {
            "action": detector,
            "decision_type": "tool",
            "arg_contains": {"label": [entity.target_entity]},
            "required_args": {"finish_after_tool": False},
        }
        return [step1, retry_step, migrate_step]
    if tac in ("retry", "migrate"):
        # Legacy v7 behaviour: collapse retry -> [detector, migration_advisor]
        # (2-step). See docstring above for the original rationale and the
        # RETRY_3STEP block for why it was a mistake for GRPO.
        return [step1, migrate_step]
    if tac == "end":
        step2 = {
            "action": "end",
            "decision_type": "end",
            # end_reason via arg_contains (soft) instead of required_args (hard).
            "arg_contains": {"end_reason": end_reason_synonyms},
            "required_args": {},
        }
        return [step1, step2]
    raise ValueError(f"unknown target_action_class={tac!r}")


def _post_retry_observation(
    *,
    entity: Entity,
    scenario: dict[str, Any],
    nuisance: Nuisance,
    detector: str,
) -> dict[str, Any]:
    """Observation the planner reads AFTER it has spent its one retry.

    Semantics follow the V15 `post_retry_metric_veto_step3` stratum: the retry
    itself succeeds at the transport level, so the planner cannot justify a
    third probe by "it errored again"; but one quality gate is still not met,
    so ending is also wrong.  The only legal move is migration.

    Two hard constraints inherited from v7:

    1. The forbidden-token gate.  We must never write``retry_count=`` /
       ``retryable=`` etc.  "The budget is spent" is conveyed *implicitly*, by
       putting this case's own first attempt into ``session_history`` as a
       completed prior attempt of the same detector.
    2. Long observation.  The same session-history / technical-notes filler
       blocks are reused, so the step-3 prompt has the same shape and length
       distribution as the step-2 prompt.
    """
    rng = _rng("obs_v7_postretry", entity.entity_id, scenario["id"], nuisance.seed_salt, 2)
    sid = scenario["id"]
    duration_ms = rng.randint(600, 5400)

    # The retry transported fine, but one gate still fails. Which gate depends
    # on the scenario so P1 and P3 stay distinguishable at step 3 as well.
    if sid == "P1_iou_low_fresh":
        # Second probe agrees no better than the first: IoU stays below the bar.
        detector_response = _detector_success_payload(
            rng, entity=entity, detector=detector,
            n_candidates=rng.choice((1, 2)), confidence_min=0.91,
            iou_between_probes=0.58, domain_shift_tag=None,
            duration_ms=duration_ms,
        )
    elif sid == "P3_transient_5xx":
        # Transport recovered, but the recovered result carries a domain-shift
        # tag, so current capability still cannot own this target.
        detector_response = _detector_success_payload(
            rng, entity=entity, detector=detector,
            n_candidates=rng.choice((1, 2)), confidence_min=0.90,
            iou_between_probes=0.86,
            domain_shift_tag=rng.choice(("night_scene", "thermal_ir", "sar_grayscale")),
            duration_ms=duration_ms,
        )
    else:
        raise ValueError(f"_post_retry_observation called for non-retry scenario {sid!r}")

    # Implicitly encode "the retry has already happened": this case's own first
    # attempt is now a completed record in session_history.
    prior_attempts = [
        {
            "query_id": f"selfprev_{rng.randrange(10**6):06d}",
            "text": f"本轮对{entity.target_entity}的首次{detector}调用",
            "attempted_action": detector,
            "attempted_arguments": {"label": entity.target_entity},
            "attempted_result_summary": (
                "首次调用的结构化响应质量不足（见当时detector_response），"
                "已在同一轮内立即复用同一检测器补测一次；本 observation 即补测结果。"
            ),
            "elapsed_since_seconds": rng.randint(20, 90),
        }
    ]
    session_history = _session_history_block(
        rng, prior_attempts=prior_attempts, filler_target_tokens=520,
    )
    tech_notes = _tech_notes(rng, entity, filler_target_tokens=1040)

    hint_map = {
        "P1_iou_low_fresh": (
            "补测后的跨提示 IoU 仍未达标，同一检测器已连续使用两次、无更多补测余量；"
            "下一步请调用 migration_advisor，禁止再次调用任何检测器。"
        ),
        "P3_transient_5xx": (
            "补测已恢复传输，但结果带域偏移标签且检测器已连续使用两次；"
            "下一步请调用 migration_advisor，禁止再次调用任何检测器。"
        ),
    }
    routing_hint = hint_map[sid]
    natural_summary = (
        f"{detector} 已对 {entity.target_entity} 完成同轮内的第二次调用；"
        f"本次结构化响应见 detector_response，两次调用的先后关系见 session_history，"
        f"业务参考资料见 technical_notes。"
        f"决策提示：{routing_hint}"
    )
    return {
        "success": True,
        "summary": natural_summary,
        "detector_response": detector_response,
        "session_history": session_history,
        "technical_notes": tech_notes,
        "tool_call_id": hex(rng.randrange(1 << 60))[2:],
    }


def _mock_observations_for_case(
    *,
    entity: Entity,
    scenario: dict[str, Any],
    nuisance: Nuisance,
    detector: str,
) -> list[dict[str, Any]]:
    """Emit observations keyed by after_step. The scoring pipeline replays
    trajectories, so we need one observation *for each* tool step whose result
    the planner reads next."""

    obs1 = build_observation(
        entity=entity, scenario=scenario, nuisance=nuisance,
        step_index=1, detector_label=detector,
    )
    out: list[dict[str, Any]] = [{"after_step": 1, "observation": obs1.observation}]
    if RETRY_3STEP and scenario["target_action_class"] == "retry":
        # 3-step retry needs the post-retry observation the planner reads
        # before its step-3 decision.
        out.append(
            {
                "after_step": 2,
                "observation": _post_retry_observation(
                    entity=entity, scenario=scenario,
                    nuisance=nuisance, detector=detector,
                ),
            }
        )
        return out
    # In the legacy 2-step gold collapse no scenario needs a second synthetic
    # observation: the rollout only calls migration_advisor / end at step 2,
    # both of which are read from the planner's own state, not from a tool
    # observation.
    return out


def build_case(
    *,
    entity: Entity,
    scenario: dict[str, Any],
    split: str,
    nuisance_index: int,
    detector: str,
    badge: str,
) -> dict[str, Any]:
    nuisance = Nuisance(detector_family=detector, badge=badge, seed_salt=nuisance_index)
    case_id = f"PRMV7-{split.upper()}-{entity.entity_id.split('_')[-1]}-{scenario['id']}"
    counterfactual_bundle_id = f"{entity.entity_id}::{scenario['id']}::{detector}"
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "case_id": case_id,
        "entity_id": entity.entity_id,
        "counterfactual_bundle_id": counterfactual_bundle_id,
        "group_id": entity.entity_id,
        "split": split,
        "provenance_class": "synthetic_long_observation_v1",
        "scenario_id": scenario["id"],
        "category": scenario["id"],
        "target_action_class": scenario["target_action_class"],
        "guardrail": scenario["role"] == "guardrail",
        "detector_family": detector,
        "badge_condition": badge,
        "project_entity": entity.project_entity,
        "target_entity": entity.target_entity,
        "image_fixture_family": entity.fixture_family,
        "template_id": f"{scenario['id']}_{detector}_{badge}",
        "user_query": _query_text(entity, scenario, nuisance),
        "setup": {
            "has_image": True,
            "image_fixture": f"examples/images/{DATASET_ID}/{entity.fixture_name}",
            "max_steps": 3,
            "query_trajectories": [],
        },
        "mock_observations": _mock_observations_for_case(
            entity=entity, scenario=scenario, nuisance=nuisance, detector=detector,
        ),
        "expected_decisions": _expected_decisions(entity, scenario, detector),
        "forbidden_actions": _forbidden_actions(detector),
        "forbidden_actions_by_step": _forbidden_actions_by_step(
            detector, scenario, len(_expected_decisions(entity, scenario, detector))
        ),
        "reward_spec": _reward_spec(),
        # The step whose decision GRPO optimises. For the 3-step retry variant
        # BOTH step 2 (retry vs. premature migrate) and step 3 (migrate vs.
        # illegal third probe) are optimisation targets; the stage-data builder
        # emits one row per entry.
        "decision_under_test_step": 2,
        "grpo_target_step": 2,
        "grpo_target_steps": (
            [2, 3]
            if RETRY_3STEP
            and scenario["target_action_class"] == "retry"
            and len(_expected_decisions(entity, scenario, detector)) == 3
            else [2]
        ),
        "sft_eligible": split in ("sft_train", "sft_dev"),
        "grpo_eligible": split in ("grpo_train", "grpo_dev"),
        "training_only": split in ("sft_train", "grpo_train"),
        "evaluation_only": split in ("sft_dev", "grpo_dev", "test"),
        "exclude_from_training": split == "test",
        "sealed": split == "test",
        "selection_role": "development" if split in ("sft_dev", "grpo_dev") else split,
        "data_stage": "cases",
    }


# --- Audit ------------------------------------------------------------------


def _stringify_obs(mock_observations: list[dict[str, Any]]) -> str:
    return json.dumps(mock_observations, ensure_ascii=False)


def audit_case(case: dict[str, Any], min_obs_tokens: int) -> list[str]:
    errors: list[str] = []
    obs_text = _stringify_obs(case.get("mock_observations") or [])
    for token in FORBIDDEN_OBS_SUBSTRINGS:
        if token in obs_text:
            errors.append(f"observation contains forbidden token {token!r}")
    q = str(case.get("user_query") or "")
    for token in FORBIDDEN_QUERY_SUBSTRINGS:
        if token in q:
            errors.append(f"user_query contains forbidden rule-restating token {token!r}")
    total_tokens = sum(_approx_tokens(item.get("observation") or {}) for item in case.get("mock_observations") or [])
    if total_tokens < min_obs_tokens:
        errors.append(f"observation approx_tokens={total_tokens} < min {min_obs_tokens}")
    if case.get("case_id") in q or case.get("entity_id") in q:
        errors.append("metadata identity leaked into user_query")
    return errors


def compute_mutual_information(
    cases: list[dict[str, Any]],
    field_name: str,
) -> float:
    """MI between a nuisance field (badge / detector_family) and the target
    action class. Zero means the nuisance cannot shortcut the label."""
    import math
    counts: dict[tuple[str, str], int] = defaultdict(int)
    field_totals: dict[str, int] = defaultdict(int)
    label_totals: dict[str, int] = defaultdict(int)
    n = 0
    for c in cases:
        f = str(c.get(field_name) or "")
        a = str(c.get("target_action_class") or "")
        counts[(f, a)] += 1
        field_totals[f] += 1
        label_totals[a] += 1
        n += 1
    if n == 0:
        return 0.0
    mi = 0.0
    for (f, a), c in counts.items():
        pfa = c / n
        pf = field_totals[f] / n
        pa = label_totals[a] / n
        if pfa > 0 and pf > 0 and pa > 0:
            mi += pfa * math.log(pfa / (pf * pa))
    return mi


# --- IO ---------------------------------------------------------------------


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --- Main -------------------------------------------------------------------


def build_all(min_obs_tokens: int) -> dict[str, Any]:
    rng = random.Random(BUILD_SEED)
    entities = build_entities(rng)
    splits = assign_splits(entities)

    all_cases_by_split: dict[str, list[dict[str, Any]]] = {}
    audit_errors: list[str] = []
    token_lengths: list[int] = []

    for split, ents in splits.items():
        rows: list[dict[str, Any]] = []
        for ent_idx, ent in enumerate(ents):
            # ensure fixture on disk
            ensure_fixture(FIXTURE_DIR / ent.fixture_name, ent)
            # Randomise badge assignment *per (entity, scenario)* using a
            # deterministic seed decorrelated from scenario id, so
            # MI(badge, target_action) approaches 0.
            badge_rng = _rng("badge_v7", split, ent_idx)
            for sc_idx, scenario in enumerate(SCENARIOS):
                # detector rotation is coupled with entity so both detectors
                # are represented across the 8 scenarios but MI stays 0 by
                # construction (each scenario has 4 entities on qwen and 4 on
                # rex within any 8-entity slice).
                detector = DETECTOR_FAMILIES[(ent_idx + sc_idx) % len(DETECTOR_FAMILIES)]
                badge = badge_rng.choice(BADGE_CONDITIONS)
                case = build_case(
                    entity=ent, scenario=scenario, split=split,
                    nuisance_index=ent_idx * 100 + sc_idx,
                    detector=detector, badge=badge,
                )
                errs = audit_case(case, min_obs_tokens=min_obs_tokens)
                if errs:
                    audit_errors.append(f"{case['case_id']}: {errs}")
                total_toks = sum(
                    _approx_tokens(item.get("observation") or {})
                    for item in case["mock_observations"]
                )
                token_lengths.append(total_toks)
                rows.append(case)
        all_cases_by_split[split] = rows

    if audit_errors:
        raise RuntimeError(
            f"audit failed with {len(audit_errors)} error(s). First few:\n"
            + "\n".join(audit_errors[:10])
        )

    # persist
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    per_split_paths: dict[str, Path] = {}
    for split, rows in all_cases_by_split.items():
        p = CASE_DIR / f"{DATASET_ID}_{split}_cases.jsonl"
        write_jsonl(p, rows)
        per_split_paths[split] = p

    # audit summary
    all_cases_flat = [c for rows in all_cases_by_split.values() for c in rows]
    mi_badge = compute_mutual_information(all_cases_flat, "badge_condition")
    mi_detector = compute_mutual_information(all_cases_flat, "detector_family")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "created_at": CREATED_AT,
        "build_seed": BUILD_SEED,
        "experimental_unit": "entity × scenario × (detector, badge) matched counterfactual bundle",
        "scenarios": [
            {"id": s["id"], "role": s["role"], "target_action_class": s["target_action_class"],
             "explain": s["explain"]}
            for s in SCENARIOS
        ],
        "split_entity_counts": ENTITIES_PER_SPLIT,
        "cases_per_entity": TOTAL_SCENARIOS,
        "totals": {
            "entities": TOTAL_ENTITIES,
            "cases": TOTAL_ENTITIES * TOTAL_SCENARIOS,
        },
        "observation": {
            "approx_tokens_min": min(token_lengths),
            "approx_tokens_mean": statistics.mean(token_lengths),
            "approx_tokens_p50": statistics.median(token_lengths),
            "approx_tokens_p95": statistics.quantiles(token_lengths, n=20)[18] if len(token_lengths) >= 20 else max(token_lengths),
            "approx_tokens_max": max(token_lengths),
            "min_required": min_obs_tokens,
        },
        "nuisance_shortcut_check": {
            "mutual_information_badge_vs_target_action": mi_badge,
            "mutual_information_detector_vs_target_action": mi_detector,
            "threshold": 0.02,
            "status": "pass" if max(mi_badge, mi_detector) < 0.02 else "fail",
        },
        "files": {split: str(p.relative_to(ROOT)) for split, p in per_split_paths.items()},
        "sha256": {
            split: sha256_file(p) for split, p in per_split_paths.items()
        },
        "forbidden_observation_substrings": list(FORBIDDEN_OBS_SUBSTRINGS),
        "forbidden_query_substrings": list(FORBIDDEN_QUERY_SUBSTRINGS),
    }
    write_json(DATASET_DIR / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-obs-tokens", type=int, default=1500,
                        help="Minimum approx tokens per case observations (summed over steps).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_all(min_obs_tokens=args.min_obs_tokens)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
