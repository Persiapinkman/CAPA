#!/usr/bin/env python3
"""为单轮题库自动建议评测标签（interaction / intent / planner / finish）。"""
from __future__ import annotations

import re

# interaction
IX_ST01 = "ST-01"
IX_MT_FRAG = "MT-fragment"
IX_MT02 = "MT-02"
IX_CL_SLOT = "CL-slot"
IX_CL01 = "CL-01"

# intent
INT_LOOKUP = "lookup_fact"
INT_RECOMMEND = "recommend_model"
INT_MIGRATION = "migration_advisor"
INT_ADELA = "adela_benchmark"
INT_VISION_GEN = "vision_generate"
INT_VISION_DET = "vision_detect"
INT_VISION_PIPE = "vision_pipeline"
INT_GENERAL = "general_chat"
INT_CLARIFY = "needs_clarify"

# planner profile
PL_1 = "PL-1"
PL_2 = "PL-2"
PL_3 = "PL-3"
PL_4 = "PL-4"

# finish expectation (typical)
FIN_TRUE = "finish:true"
FIN_FALSE = "finish:false"
FIN_MIXED = "finish:mixed"

_ADELA_PLATFORM = re.compile(
    r"(cuda\d+\.\d+-trt[A-Za-z0-9.\-]+|acl-[A-Za-z0-9.\-]+|cpu-[A-Za-z0-9.\-]+|rknn-[A-Za-z0-9.\-]+)",
    re.I,
)
_SLOT_FORM = re.compile(r'"_structured_type"\s*:\s*"adela_slot_form"', re.I)


def _has_any(text: str, tokens: tuple[str, ...]) -> bool:
    return any(t in text for t in tokens)


def suggest_tags(
    question: str,
    *,
    is_clarification_reply_seen: bool = False,
    occurrence_count: int = 1,
) -> tuple[list[str], str]:
    q = re.sub(r"\s+", " ", str(question or "").strip())
    low = q.casefold()
    tags: list[str] = []
    notes: list[str] = []

    is_supplement = "用户补充说明" in q
    is_slot_only = bool(_ADELA_PLATFORM.search(q)) and len(q) < 80 and "部署" not in q
    is_slot_form = bool(_SLOT_FORM.search(q))

    # --- interaction ---
    if is_supplement or is_slot_form:
        tags.extend([IX_MT02, IX_CL_SLOT])
        notes.append("澄清续跑/槽位补充，须与前置轮次组成 case")
    elif is_clarification_reply_seen:
        if is_slot_only or _has_any(q, ("精度", "性能", "换成", "改成")):
            tags.extend([IX_CL_SLOT, IX_MT02])
            notes.append("Adela 槽位补充")
        elif len(q) <= 24 and _has_any(q, ("那", "这", "呢", "怎么样", "如何")):
            tags.append(IX_MT_FRAG)
            notes.append("追问片段，不能单独评测，需绑定上一轮")
        else:
            tags.append(IX_CL01)
    else:
        tags.append(IX_ST01)

    # --- intent ---
    if _has_any(
        q,
        (
            "介绍一下自己",
            "你能干什么",
            "你是谁",
            "苹果是当季",
            "条件随机场",
            "树葡萄",
        ),
    ):
        tags.append(INT_GENERAL)

    if q == "黑夜检测黑猫" or (("黑猫" in q or "黑夜" in q) and len(q) < 20):
        tags.append(INT_CLARIFY)
        notes.append("意图歧义，期望 clarify 而非直接选工具")

    if _ADELA_PLATFORM.search(q) or (
        "部署" in q
        and _has_any(q, ("精度", "性能", "benchmark", "评测", "准确率"))
    ):
        tags.append(INT_ADELA)
        if "struct_" in low or "rawmodel" in low or re.search(r"\b[a-z][a-z0-9_]{6,}\b", q):
            pass
        elif q.startswith("部署") and "模型" in q:
            notes.append("Adela 多轮槽位填充典型题")

    if _has_any(
        q,
        (
            "能不能做",
            "能否",
            "能不能",
            "直接支持",
            "基于现有",
            "迁移",
            "温故知新",
            "怎么办",
            "评估一下",
            "POC",
            "可行性",
            "能不能覆盖",
        ),
    ) or ("怎么办" in q and "模型" in q):
        tags.append(INT_MIGRATION)

    if _has_any(
        q,
        (
            "推荐",
            "用什么模型",
            "该用哪些",
            "该用什么",
            "有哪些模型",
            "设计一个算法流程",
            "算法流程",
        ),
    ) and INT_MIGRATION not in tags:
        tags.append(INT_RECOMMEND)

    if _has_any(
        q,
        (
            "多少",
            "几个",
            "是哪个",
            "最新",
            "版本",
            "维度",
            "oid",
            "标签",
            "有没有",
            "有部署",
            "统计",
            "发布了",
        ),
    ):
        if INT_RECOMMEND not in tags and INT_MIGRATION not in tags:
            tags.append(INT_LOOKUP)

    if _has_any(q, ("生成", "张", "图片")) and "检测" not in q[:6]:
        tags.append(INT_VISION_GEN)

    model_recommend_ctx = _has_any(
        q,
        (
            "用什么模型",
            "哪些模型",
            "该用哪些",
            "推荐",
            "有没有部署",
            "有部署",
            "最新的模型",
            "是目前最新",
            "模型么",
            "模型吗",
        ),
    )
    if _has_any(q, ("标注", "检测", "识别")) and "生成" not in q[:4]:
        if "评测报告" in q or "评估报告" in q:
            tags.append(INT_VISION_PIPE)
        elif not model_recommend_ctx and (
            len(q) < 80 or "qwen" in low or "rex" in low or q.startswith("检测")
        ):
            tags.append(INT_VISION_DET)

    if INT_VISION_GEN in tags and INT_VISION_DET in tags:
        tags.remove(INT_VISION_DET)
        tags.append(INT_VISION_PIPE)
        notes.append("生图+检测/报告，可能为多步链")

    # default intent
    if not any(
        t in tags
        for t in (
            INT_LOOKUP,
            INT_RECOMMEND,
            INT_MIGRATION,
            INT_ADELA,
            INT_VISION_GEN,
            INT_VISION_DET,
            INT_VISION_PIPE,
            INT_GENERAL,
            INT_CLARIFY,
        )
    ):
        tags.append(INT_LOOKUP)

    # --- planner + finish ---
    if INT_CLARIFY in tags:
        tags.extend([PL_1, FIN_TRUE])
    elif INT_GENERAL in tags and INT_LOOKUP not in tags:
        tags.extend([PL_1, FIN_TRUE])
    elif INT_ADELA in tags and IX_ST01 in tags and "部署" not in q:
        tags.extend([PL_1, FIN_TRUE])
        notes.append("含平台串的事实问，可能 forced adela 或 rag，需在 case 标明 acceptable_tools")
    elif INT_ADELA in tags:
        tags.extend([PL_1, FIN_TRUE])
    elif INT_MIGRATION in tags:
        tags.extend([PL_1, FIN_TRUE])
        notes.append("迁移顾问：抽象+多路检索+报告，对外单 query 收口")
    elif INT_VISION_PIPE in tags:
        tags.extend([PL_3, FIN_MIXED])
        notes.append("典型：flux finish=false -> detect -> pipeline_eval")
    elif INT_VISION_GEN in tags and INT_VISION_DET in tags:
        tags.extend([PL_2, FIN_MIXED])
    elif INT_VISION_GEN in tags:
        tags.extend([PL_1, FIN_TRUE])
    elif INT_VISION_DET in tags:
        tags.extend([PL_1, FIN_TRUE])
        notes.append("缺图时应 clarify；case 需带 image fixture")
    elif INT_RECOMMEND in tags:
        tags.extend([PL_4, FIN_MIXED])
        notes.append("常 rag miss -> re_question -> rag/answerer")
    elif INT_LOOKUP in tags:
        tags.extend([PL_1, FIN_TRUE])
        if occurrence_count >= 3:
            notes.append("高频题，适合回归")

    if IX_MT_FRAG in tags:
        tags.extend([PL_1, FIN_MIXED])

    # dedupe preserve order
    seen: set[str] = set()
    out: list[str] = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)

    note_text = "；".join(n for n in notes if n).strip()
    return out, note_text
