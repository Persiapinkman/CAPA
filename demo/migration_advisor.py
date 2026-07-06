from __future__ import annotations

"""
Migration advisor workflow helpers.

主要功能：
- 在 RAG 多轮未命中后，将用户需求抽象成迁移评估检索计划。
- 基于 /playbook/retrieve 的证据片段，生成结构化迁移顾问报告。
- 若用户提供图片：分字段检索后调用 Rex-Omni 检测画框，并用 Qwen3.5-4B 预估标注准确率。
- 保持业务 workflow 与 HTTP / Agent 主循环解耦。
"""

import json
import os
import re
import sys
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util.rex_label_extraction import extract_rex_detection_labels  # noqa: E402
from util.vlm_service import VLMService  # noqa: E402

_DEFAULT_ROUTE_API_BASE = os.environ.get("DEMO_ROUTE_API_BASE", "http://10.111.32.253:8000/v1")
_DEFAULT_ROUTE_API_KEY = os.environ.get("DEMO_ROUTE_API_KEY", "token.sdc@2026")
MIGRATION_ADVISOR_MODEL = os.environ.get("DEMO_MIGRATION_ADVISOR_MODEL") or os.environ.get(
    "DEMO_ANSWER_MODEL", "Qwen3.5-4B"
)
MIGRATION_REPORT_MODEL = os.environ.get("DEMO_MIGRATION_REPORT_MODEL", "Qwen3.5-35B-A3B")
MIGRATION_ADVISOR_API_BASE = (
    os.environ.get("DEMO_MIGRATION_ADVISOR_API_BASE")
    or os.environ.get("DEMO_ANSWER_API_BASE")
    or os.environ.get("DEMO_LLM_API_BASE")
    or _DEFAULT_ROUTE_API_BASE
)
MIGRATION_ADVISOR_API_KEY = (
    os.environ.get("DEMO_MIGRATION_ADVISOR_API_KEY")
    or os.environ.get("DEMO_ANSWER_API_KEY")
    or os.environ.get("DEMO_LLM_API_KEY")
    or _DEFAULT_ROUTE_API_KEY
)
LLM_DEBUG_ENABLED = str(os.environ.get("DEMO_LLM_DEBUG_ENABLED", "1")).strip().lower() not in {
    "0",
    "false",
    "off",
    "no",
}
LLM_DEBUG_DIR = ROOT / "demo" / "llm_debug"
BEIJING_TZ = timezone(timedelta(hours=8))
REPORT_EVIDENCE_PER_FIELD = 5
REPORT_ASSETS_PER_FIELD = 10
REPORT_EVIDENCE_PER_ASSET = 3
MIGRATION_FIELDS = [
    {
        "field": "existing_models",
        "label": "可直接复用的已有模型",
        "default_query_suffix": "现有模型 检测 分类 识别",
    },
    {
        "field": "similar_capabilities",
        "label": "可迁移的相似能力",
        "default_query_suffix": "相似能力 属性识别 细粒度分类 迁移",
    },
    {
        "field": "performance_baseline",
        "label": "性能与精度基线",
        "default_query_suffix": "精度 性能 benchmark 指标",
    },
    {
        "field": "data_requirements",
        "label": "数据、标注与迭代依赖",
        "default_query_suffix": "数据 标注 训练 迭代 成本",
    },
]

INDUSTRY_SYNONYM_GROUPS = [
    {
        "triggers": ["电子元器件", "元器件", "电路板", "pcb", "pcba", "smt", "aoi"],
        "terms": ["电子元器件", "PCB", "PCBA", "SMT", "AOI", "工业视觉"],
    },
    {
        "triggers": ["缺陷", "异常", "瑕疵", "外观", "表面"],
        "terms": ["缺陷检测", "异常检测", "表面缺陷", "外观缺陷"],
    },
]

RETRIEVAL_PLAN_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "migration_retrieval_plan",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "abstract_requirement": {
                    "type": "object",
                    "properties": {
                        "object": {"type": "string"},
                        "attribute": {"type": "string"},
                        "task_type": {"type": "string"},
                        "scene": {"type": "string"},
                        "constraints": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["object", "attribute", "task_type", "scene", "constraints"],
                    "additionalProperties": False,
                },
                "retrieve_fields": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {
                                "type": "string",
                                "enum": [item["field"] for item in MIGRATION_FIELDS],
                            },
                            "queries": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 2,
                                "maxItems": 4,
                                "description": "该字段下的多条短检索 query，每条只保留核心实体和检索意图。",
                            },
                        },
                        "required": ["field", "queries"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["abstract_requirement", "retrieve_fields"],
            "additionalProperties": False,
        },
    },
}

REPORT_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "migration_advisor_report",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "requirement_summary": {"type": "string"},
                "direct_match": {
                    "type": "object",
                    "properties": {
                        "exists": {"type": "boolean"},
                        "summary": {"type": "string"},
                        "evidence": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["exists", "summary", "evidence"],
                    "additionalProperties": False,
                },
                "similar_assets": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "model_or_solution": {"type": "string"},
                            "label_schema": {
                                "type": "string",
                                "description": "类别/标签定义；证据无则写「证据不足」。",
                            },
                            "training_data": {
                                "type": "string",
                                "description": "训练集、标注规范、样本量；证据无则写「证据不足」。",
                            },
                            "reported_metrics": {
                                "type": "string",
                                "description": "mAP/ACC/召回等精度指标及测试集/平台；证据无则写「证据不足」。",
                            },
                            "covered_capability": {"type": "string"},
                            "gap": {"type": "string"},
                            "evidence": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": [
                            "model_or_solution",
                            "label_schema",
                            "training_data",
                            "reported_metrics",
                            "covered_capability",
                            "gap",
                            "evidence",
                        ],
                        "additionalProperties": False,
                    },
                },
                "migration_plan": {
                    "type": "object",
                    "properties": {
                        "feasibility": {"type": "string", "enum": ["high", "medium", "low"]},
                        "approach": {"type": "string"},
                        "data_requirements": {"type": "string"},
                        "compute_requirements": {"type": "string"},
                        "engineering_work": {"type": "string"},
                        "estimated_timeline": {"type": "string"},
                        "estimated_cost": {"type": "string"},
                        "dependencies": {"type": "array", "items": {"type": "string"}},
                        "risks": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "feasibility",
                        "approach",
                        "data_requirements",
                        "compute_requirements",
                        "engineering_work",
                        "estimated_timeline",
                        "estimated_cost",
                        "dependencies",
                        "risks",
                    ],
                    "additionalProperties": False,
                },
                "expected_performance": {
                    "type": "object",
                    "properties": {
                        "baseline": {"type": "string"},
                        "target": {"type": "string"},
                        "uncertainty": {"type": "string"},
                    },
                    "required": ["baseline", "target", "uncertainty"],
                    "additionalProperties": False,
                },
                "recommendation": {"type": "string"},
            },
            "required": [
                "requirement_summary",
                "direct_match",
                "similar_assets",
                "migration_plan",
                "expected_performance",
                "recommendation",
            ],
            "additionalProperties": False,
        },
    },
}

FACT_EXTRACTION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "migration_fact_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "facts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "fact_type": {
                                "type": "string",
                                "enum": [
                                    "model_identity",
                                    "task_scope",
                                    "category_count",
                                    "accuracy",
                                    "performance",
                                    "model_size",
                                    "data_scale",
                                    "deployment",
                                    "limitation",
                                    "cost_or_timeline",
                                    "other",
                                ],
                            },
                            "field": {
                                "type": "string",
                                "enum": [item["field"] for item in MIGRATION_FIELDS],
                            },
                            "subject": {"type": "string"},
                            "fact": {"type": "string"},
                            "value": {"type": "string"},
                            "unit": {"type": "string"},
                            "evidence_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "doc_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "quote": {"type": "string"},
                            "confidence": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                            },
                        },
                        "required": [
                            "fact_type",
                            "field",
                            "subject",
                            "fact",
                            "value",
                            "unit",
                            "evidence_ids",
                            "doc_ids",
                            "quote",
                            "confidence",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["facts"],
            "additionalProperties": False,
        },
    },
}


def _safe_debug_name(value: str, fallback: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^a-zA-Z0-9_.-]+", "_", text).strip("._")
    return text or fallback


def _bj_date_prefix() -> str:
    return datetime.now(BEIJING_TZ).strftime("%Y%m%d")


COMPACT_EVIDENCE_FIELDS = [
    "evidence_id",
    "title",
    "snippet",
    "model_name",
    "algorithm_name",
    "algorithm_type",
    "application_scene",
    "section_type",
    "score",
]

REPORT_EVIDENCE_FIELDS = [
    "asset_name",
    "model_name",
    "target_name",
    "algorithm_name",
    "algorithm_type",
    "application_scene",
    "labels",
    "benchmark_info",
    "evidence[].evidence_id",
    "evidence[].section_type",
    "evidence[].index_text",
]

_FIELD_LABEL_MAP = {str(item["field"]): str(item["label"]) for item in MIGRATION_FIELDS}

_ENGLISH_KV_LINE = re.compile(r"^\s*[A-Za-z][A-Za-z0-9_]*\s*[:=]\s*(.+?)\s*$")
_ENGLISH_KV_INLINE = re.compile(r"[A-Za-z][A-Za-z0-9_]*\s*[:=]\s*")
_QUOTE_CHARS = "\"'“”‘’「」"
_PAREN_CONTENT = re.compile(r"\([^)]*\)|（[^）]*）|\[[^\]]*\]")

RETRIEVAL_PLAN_INPUT_SCHEMA = {
    "user_query": "用户原始问题",
    "failed_rag_trace": "最近最多 3 轮 RAG 失败轨迹（round/query/knowledge_base_fully_answered/evidence[]）",
    "required_fields": "固定 4 个检索维度 field + label",
}

RETRIEVAL_PLAN_OUTPUT_SCHEMA = {
    "abstract_requirement": "object, attribute, task_type, scene, constraints[]",
    "retrieve_fields": "每项 field + queries[]（2-4 条）",
}

REPORT_INPUT_SCHEMA = {
    "user_query": "用户原始问题",
    "field_results": (
        "分字段检索结果：field, assets[]。assets 按模型/方案聚合，包含模型身份字段、labels、"
        "benchmark_info 摘要，以及 evidence[]（evidence_id/section_type/index_text）。"
    ),
}

REPORT_OUTPUT_SCHEMA = {
    "requirement_summary": "需求摘要",
    "direct_match": "exists, summary, evidence[]",
    "similar_assets": (
        "model_or_solution, label_schema, training_data, reported_metrics, "
        "covered_capability, gap, evidence[]"
    ),
    "migration_plan": "feasibility, approach, data/compute/engineering, timeline, cost, dependencies[], risks[]",
    "expected_performance": "baseline, target, uncertainty",
    "recommendation": "最终建议",
}

REPORT_SYSTEM_PROMPT = (
    "你是视觉模型迁移顾问。请基于检索证据生成结构化迁移评估报告（中文）。\n"
    "本报告的首要目标是总结可用于迁移参考的相似模型信息，而不是强求找到与需求完全一致的直接模型。\n"
    "只要检索到的模型/方案在对象、属性、任务形态、标签体系、评测方式、部署平台中任一方面与需求相近，"
    "就应作为 similar_assets 的有效候选，优先提炼其可复用信息。\n"
    "必须区分证据支持和合理推断；证据不足时明确写「证据不足」，不要编造精度、周期或成本。\n"
    "若可估计成本/周期，只能给区间和依赖前提。\n"
    "direct_match 仅用于回答“是否存在直接匹配模型”。即使不存在直接匹配，也不影响 similar_assets 充分展开。\n"
    "similar_assets 中每个相似模型/方案必须逐项填写：\n"
    "- label_schema：该模型的类别/标签定义（含类别名、取值范围）；\n"
    "- training_data：训练集来源、标注规范、样本量或数据要求；\n"
    "- reported_metrics：文档中的 mAP、ACC、recall、benchmark 等数值及对应测试集/平台；\n"
    "field_results 按 assets 聚合；先判断 asset 与用户需求的相似性，再从同一 asset 的结构化字段、"
    "benchmark_info 与 evidence.index_text 中提取标签、训练数据和精度。无对应文本则写「证据不足」。\n"
    "若某个 asset 已有标签、精度、平台或场景中的部分信息，必须先如实填写已知部分，"
    "只对缺失字段写「证据不足」，不要因为“不是直接模型”就把整项写成「证据不足」。\n"
    "evidence 数组可引用 evidence_id 或证据原文短句。只输出 JSON。"
)

REPORT_REX_ACCURACY_COMPARE_PROMPT = (
    "\n若输入含 rex_omni_benchmark（用户上传图经 Rex-Omni 标注后的预估准确率），且 similar_assets 中检索到相似模型并填写了 reported_metrics：\n"
    "1) 在 migration_plan.approach 中对比两侧精度（勿编造未给出的数值）；\n"
    "2) 在 recommendation 中给出路径建议：检索模型精度更高 → 说明自训练/已有模型更有优势，建议优先训练或微调自有模型；"
    "Rex-Omni 预估更高 → 建议先用开源开集模型验证，再积累标注数据训练专用模型；接近则两种路径均可并说明取舍；\n"
    "3) expected_performance.baseline 写相似模型侧精度，target 写 Rex-Omni 样例侧预估。\n"
)

_ACCURACY_NUMBER = re.compile(r"(\d+(?:\.\d+)?)")

FACT_EXTRACTION_INPUT_SCHEMA = {
    "user_query": "用户原始问题",
    "retrieval_plan": "检索规划",
    "evidence_packages": "分字段证据包：field, queries, evidences[], full_documents[]",
}

FACT_EXTRACTION_OUTPUT_SCHEMA = {
    "facts": "结构化事实数组。每条必须有 fact_type/field/subject/fact/value/unit/evidence_ids/doc_ids/quote/confidence",
}

REX_ACCURACY_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "migration_rex_accuracy_estimate",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "accuracy": {
                    "type": "string",
                    "description": "预估准确率，例如 85% 或 70-80%",
                },
                "reason": {
                    "type": "string",
                    "description": "2-4 句话说明依据：目标数量、预测框数量、是否覆盖目标、漏检/误检/过检等",
                },
            },
            "required": ["accuracy", "reason"],
            "additionalProperties": False,
        },
    },
}

REX_ACCURACY_OUTPUT_SCHEMA = {
    "accuracy": "预估准确率（如 85% 或 70-80%）",
    "reason": "依据说明",
}


def _debug_request_payload(
    *,
    model: str,
    messages: list[dict],
    run_dir: str,
    structured_input: dict,
    input_schema: dict,
    response_format: dict | None = None,
    assistant_input_name: str = "",
    evidence_item_fields: list[str] | None = None,
) -> dict:
    payload = {
        "model": model,
        "run_dir": run_dir,
        "input_schema": input_schema,
        "evidence_item_fields": evidence_item_fields or COMPACT_EVIDENCE_FIELDS,
        "structured_input": structured_input,
        "llm_messages": messages,
    }
    if response_format is not None:
        payload["response_format"] = response_format
    if assistant_input_name:
        payload["assistant_input_name"] = assistant_input_name
        payload["assistant_input"] = {
            "messages": messages,
            "response_format": response_format,
        }
    return payload


def _debug_response_payload(
    *,
    model: str,
    raw: str,
    output_schema: dict | None = None,
    structured_output: dict | list | None = None,
    **extra,
) -> dict:
    payload: dict = {"model": model, "output_schema": output_schema or {}, **extra}
    if raw:
        payload["output_raw"] = raw
    if structured_output is not None:
        payload["structured_output"] = structured_output
    elif raw:
        try:
            payload["structured_output"] = _extract_first_json_object(raw)
        except Exception:
            payload["structured_output"] = None
    return payload


def _write_migration_llm_debug(
    *,
    debug_meta: dict | None,
    stage: str,
    step_index: int,
    payload: dict,
) -> None:
    if not LLM_DEBUG_ENABLED:
        return
    meta = debug_meta if isinstance(debug_meta, dict) else {}
    try:
        sid = _safe_debug_name(str(meta.get("session_id") or ""), "unknown_session")
        stamp = _safe_debug_name(str(meta.get("run_stamp") or ""), "unknown_run")
        stage_name = _safe_debug_name(stage, "unknown_stage")
        seq = max(1, int(step_index or 1))
        date_prefix = _bj_date_prefix()
        stamp_tail = re.sub(r"^\d{8}_", "", stamp).strip("_") or "unknown_run"
        target_dir = LLM_DEBUG_DIR / date_prefix
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / f"{date_prefix}_{sid}_{stamp_tail}_{seq:02d}_{stage_name}.json"
        body = {
            "session_id": str(meta.get("session_id") or "").strip(),
            "run_stamp": str(meta.get("run_stamp") or "").strip(),
            "stage": stage,
            "step_index": seq,
            "created_at": datetime.now(BEIJING_TZ).isoformat(timespec="seconds"),
            "payload": payload,
        }
        file_path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return


def _migration_vlm(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> VLMService:
    key = str(api_key or MIGRATION_ADVISOR_API_KEY).strip()
    url = str(base_url or MIGRATION_ADVISOR_API_BASE).rstrip("/")
    return VLMService(api_key=key, base_url=url)


def _generate_migration_text(
    *,
    messages: list[dict],
    response_format: dict | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    image_paths: list[str] | None = None,
) -> str:
    vlm = _migration_vlm(api_key=api_key, base_url=base_url)
    return vlm.generate_text(
        messages=messages,
        model=model or MIGRATION_ADVISOR_MODEL,
        response_format=response_format,
        image_paths=image_paths,
    )


def _extract_first_json_object(text: str) -> dict:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9]*\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw).strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except (TypeError, ValueError):
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        data = json.loads(raw[start : end + 1])
        if isinstance(data, dict):
            return data
    raise ValueError("no json object found")


def _trim_text(value: str, limit: int | None = 500) -> str:
    text = str(value or "").strip()
    if limit is None or limit <= 0:
        return text
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _normalize_snippet(value: str) -> str:
    """压缩 Playbook/PDF 片段中的对齐空白与空行，保留有意义的换行（如发版表 key: value）。"""
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        return ""
    lines: list[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"[ \t]{2,}", " ", line)
        lines.append(line)
    return "\n".join(lines)


def _clean_query_core(user_query: str) -> str:
    text = str(user_query or "").strip()
    replacements = (
        "我想",
        "请问",
        "帮我",
        "帮忙",
        "推荐",
        "相关的",
        "适合的",
        "有没有",
        "是否有",
        "是否",
        "能不能",
        "可以吗",
        "什么模型",
        "哪个模型",
        "模型",
        "算法",
        "可迁移",
        "迁移",
        "用于",
        "进行",
    )
    for item in replacements:
        text = text.replace(item, " ")
    text = re.sub(r"[，。！？、；：,.!?;:()\[\]{}<>《》\"'“”‘’/\\|]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return _trim_text(text, 80)


def _dedupe_queries(values: list[str], *, limit: int = 4) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        q = re.sub(r"\s+", " ", str(raw or "").strip())
        q = _trim_text(q, 80)
        if not q or q in seen:
            continue
        seen.add(q)
        out.append(q)
        if len(out) >= limit:
            break
    return out


def _industry_terms_for_query(user_query: str, core: str) -> list[str]:
    haystack = f"{user_query} {core}".lower()
    terms: list[str] = []
    for group in INDUSTRY_SYNONYM_GROUPS:
        triggers = [str(item).lower() for item in group.get("triggers", [])]
        if any(trigger and trigger in haystack for trigger in triggers):
            terms.extend(str(item) for item in group.get("terms", []))
    return _dedupe_queries(terms, limit=12)


def _fallback_query_profile(user_query: str) -> dict:
    core = _clean_query_core(user_query)
    if not core:
        core = _trim_text(user_query, 60)
    terms = _industry_terms_for_query(user_query, core)
    object_term = "电子元器件" if "电子元器件" in terms else core
    task_term = "缺陷检测" if "缺陷检测" in terms else "检测"
    pcb_query = "PCB AOI 异常检测" if {"PCB", "AOI", "异常检测"} <= set(terms) else ""
    surface_query = "工业视觉 表面缺陷" if {"工业视觉", "表面缺陷"} <= set(terms) else ""
    return {
        "core": core,
        "object_task": f"{object_term} {task_term}".strip(),
        "pcb_query": pcb_query,
        "surface_query": surface_query,
    }


def _fallback_queries_for_field(user_query: str, field: str) -> list[str]:
    profile = _fallback_query_profile(user_query)
    core = profile["core"]
    object_task = profile["object_task"]
    pcb_query = profile["pcb_query"]
    surface_query = profile["surface_query"]
    templates = {
        "existing_models": [
            f"{object_task} 模型",
            pcb_query,
            f"{surface_query} 算法" if surface_query else "",
            f"{core} 识别 模型",
        ],
        "similar_capabilities": [
            f"{object_task} 相似能力",
            f"{surface_query} 迁移" if surface_query else "",
            f"{core} 属性识别",
            f"{core} 细粒度分类",
        ],
        "performance_baseline": [
            f"{object_task} 精度",
            f"{pcb_query} benchmark" if pcb_query else "",
            f"{surface_query} 性能" if surface_query else "",
            f"{core} 模型 指标",
        ],
        "data_requirements": [
            f"{object_task} 数据 标注",
            f"{surface_query} 样本 标注" if surface_query else "",
            f"{core} 训练 数据需求",
            f"{core} 迭代 成本",
        ],
    }
    return _dedupe_queries(templates.get(field, [core]), limit=4)


def _fallback_plan(user_query: str) -> dict:
    return {
        "abstract_requirement": {
            "object": "",
            "attribute": "",
            "task_type": "检测/属性识别迁移评估",
            "scene": "",
            "constraints": [],
        },
        "retrieve_fields": [
            {
                "field": item["field"],
                "queries": _fallback_queries_for_field(user_query, item["field"]),
            }
            for item in MIGRATION_FIELDS
        ],
    }


def build_retrieval_plan(
    *,
    user_query: str,
    rag_trace: list[dict] | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    debug_meta: dict | None = None,
) -> dict:
    fallback = _fallback_plan(user_query)
    trace = rag_trace if isinstance(rag_trace, list) else []
    use_model = model or MIGRATION_ADVISOR_MODEL
    system = (
        "你是视觉模型迁移顾问的检索规划器。用户的问题没有被知识库直接回答。"
        "请先抽取目标对象、部件/场景、属性/缺陷、任务类型等核心实体，再补充行业常用同义词。"
        "例如电子元器件缺陷检测可扩展为工业视觉、表面缺陷、异常检测。"
        "为每个固定字段生成 2-4 条短检索 query，每条 query 只放一组核心实体/同义词和一个检索意图。"
        "existing_models 偏已有模型/算法，similar_capabilities 偏可迁移能力，performance_baseline 偏精度/性能/benchmark，data_requirements 偏数据/标注/训练。"
        "删除“推荐/有没有/帮我/是否/什么模型”等口语噪声，不要把多组同义词堆成一条长句。"
        "禁止编造模型结论，只做检索规划。注意检索词除了缩略词，要求纯中文。只输出 JSON。"
    )
    structured_input = {
        "user_query": str(user_query or "").strip(),
        "failed_rag_trace": _compact_rag_trace(trace),
        "required_fields": [
            {"field": item["field"], "label": item["label"]}
            for item in MIGRATION_FIELDS
        ],
    }
    user = json.dumps(structured_input, ensure_ascii=False, indent=2)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    _write_migration_llm_debug(
        debug_meta=debug_meta,
        stage="migration_retrieval_plan_request",
        step_index=1,
        payload=_debug_request_payload(
            model=use_model,
            messages=messages,
            run_dir=str((debug_meta or {}).get("run_dir") or ""),
            structured_input=structured_input,
            input_schema=RETRIEVAL_PLAN_INPUT_SCHEMA,
        ),
    )
    try:
        raw = _generate_migration_text(
            messages=messages,
            model=use_model,
            response_format=RETRIEVAL_PLAN_RESPONSE_FORMAT,
            api_key=api_key,
            base_url=base_url,
        )
        data = _extract_first_json_object(raw)
    except Exception as exc:
        _write_migration_llm_debug(
            debug_meta=debug_meta,
            stage="migration_retrieval_plan_response",
            step_index=1,
            payload=_debug_response_payload(
                model=use_model,
                raw="",
                output_schema=RETRIEVAL_PLAN_OUTPUT_SCHEMA,
                error=str(exc),
                used_fallback=True,
            ),
        )
        return fallback

    fields_in = data.get("retrieve_fields")
    if not isinstance(fields_in, list):
        _write_migration_llm_debug(
            debug_meta=debug_meta,
            stage="migration_retrieval_plan_response",
            step_index=1,
            payload=_debug_response_payload(
                model=use_model,
                raw=raw,
                output_schema=RETRIEVAL_PLAN_OUTPUT_SCHEMA,
                structured_output=data if isinstance(data, dict) else None,
                error="retrieve_fields is not a list",
                used_fallback=True,
            ),
        )
        return fallback
    by_field: dict[str, list[str]] = {}
    valid = {item["field"] for item in MIGRATION_FIELDS}
    for item in fields_in:
        if not isinstance(item, dict):
            continue
        field = str(item.get("field") or "").strip()
        raw_queries = item.get("queries")
        if not isinstance(raw_queries, list):
            raw_queries = [item.get("query")]
        queries = _dedupe_queries([str(q or "") for q in raw_queries], limit=4)
        if field in valid and queries:
            by_field[field] = queries
    out = {
        "abstract_requirement": data.get("abstract_requirement")
        if isinstance(data.get("abstract_requirement"), dict)
        else fallback["abstract_requirement"],
        "retrieve_fields": [],
    }
    for item in MIGRATION_FIELDS:
        out["retrieve_fields"].append(
            {
                "field": item["field"],
                "queries": by_field.get(
                    item["field"],
                    _fallback_queries_for_field(user_query, item["field"]),
                ),
            }
        )
    used_field_fallback = any(
        item["field"] not in by_field for item in MIGRATION_FIELDS
    )
    _write_migration_llm_debug(
        debug_meta=debug_meta,
        stage="migration_retrieval_plan_response",
        step_index=1,
        payload=_debug_response_payload(
            model=use_model,
            raw=raw,
            output_schema=RETRIEVAL_PLAN_OUTPUT_SCHEMA,
            structured_output=out,
            used_fallback=used_field_fallback,
        ),
    )
    return out


def _chunk_score(chunk: dict) -> float:
    for key in ("source_score", "score"):
        try:
            return float(chunk.get(key) or 0)
        except (TypeError, ValueError):
            continue
    return 0.0


def _top_chunks_by_score(
    chunks: list[dict],
    *,
    limit: int = REPORT_EVIDENCE_PER_FIELD,
) -> list[dict]:
    ranked = sorted(
        [c for c in chunks if isinstance(c, dict)],
        key=_chunk_score,
        reverse=True,
    )
    cap = max(1, int(limit or REPORT_EVIDENCE_PER_FIELD))
    return ranked[:cap]


def _first_chunk_field(chunk: dict, *keys: str) -> str:
    payload = chunk.get("payload") if isinstance(chunk.get("payload"), dict) else {}
    entity = payload.get("entity") if isinstance(payload.get("entity"), dict) else {}
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    canonical = (
        metadata.get("canonical_metadata")
        if isinstance(metadata.get("canonical_metadata"), dict)
        else payload.get("canonical_metadata")
        if isinstance(payload.get("canonical_metadata"), dict)
        else {}
    )
    for key in keys:
        for src in (chunk, metadata, canonical, payload, entity):
            if not isinstance(src, dict):
                continue
            value = str(src.get(key) or "").strip()
            if value:
                return value
    return ""


def _compact_chunk(chunk: dict, idx: int, *, include_rank_meta: bool = True) -> dict:
    payload = chunk.get("payload") if isinstance(chunk.get("payload"), dict) else {}
    entity = payload.get("entity") if isinstance(payload.get("entity"), dict) else {}
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    snippet = _normalize_snippet(
        str(
            chunk.get("snippet")
            or chunk.get("text")
            or chunk.get("content")
            or chunk.get("summary")
            or payload.get("text")
            or payload.get("content")
            or payload.get("summary")
            or entity.get("model_info")
            or ""
        )
    )
    title = str(
        chunk.get("title")
        or chunk.get("doc_name")
        or payload.get("title")
        or payload.get("doc_name")
        or entity.get("name")
        or entity.get("model_name")
        or f"evidence_{idx}"
    ).strip()
    eid = str(chunk.get("evidence_id") or chunk.get("legacy_evidence_id") or idx).strip()
    out: dict = {
        "evidence_id": eid,
        "title": _trim_text(title, 120),
        "snippet": _trim_text(snippet, 800),
    }
    for field, keys in (
        ("model_name", ("model_name", "canonical_model_name")),
        ("algorithm_name", ("algorithm_name", "canonical_algorithm_name")),
        ("algorithm_type", ("algorithm_type", "canonical_algorithm_type")),
        ("application_scene", ("application_scene", "canonical_application_scene")),
    ):
        value = _first_chunk_field(chunk, *keys)
        if value:
            out[field] = _trim_text(value, 120)
    if include_rank_meta:
        section_type = _first_chunk_field(chunk, "section_type")
        if section_type:
            out["section_type"] = _trim_text(section_type, 80)
        score = _chunk_score(chunk)
        if score > 0:
            out["score"] = round(score, 6)
    return out


def _compact_chunk_index_text(chunk: dict, idx: int) -> dict:
    payload = chunk.get("payload") if isinstance(chunk.get("payload"), dict) else {}
    eid = str(
        chunk.get("evidence_id")
        or chunk.get("legacy_evidence_id")
        or payload.get("chunk_id")
        or idx
    ).strip()
    return {
        "evidence_id": eid,
        "index_text": _normalize_snippet(str(payload.get("index_text") or "")),
    }


def _chunk_payload(chunk: dict) -> dict:
    return chunk.get("payload") if isinstance(chunk.get("payload"), dict) else {}


def _chunk_metadata(chunk: dict) -> dict:
    return chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}


def _chunk_canonical(chunk: dict) -> dict:
    payload = _chunk_payload(chunk)
    metadata = _chunk_metadata(chunk)
    if isinstance(payload.get("canonical_metadata"), dict):
        return payload.get("canonical_metadata") or {}
    if isinstance(metadata.get("canonical_metadata"), dict):
        return metadata.get("canonical_metadata") or {}
    return {}


def _first_chunk_raw(chunk: dict, *keys: str) -> Any:
    payload = _chunk_payload(chunk)
    metadata = _chunk_metadata(chunk)
    canonical = _chunk_canonical(chunk)
    entity = payload.get("entity") if isinstance(payload.get("entity"), dict) else {}
    for key in keys:
        for src in (chunk, metadata, canonical, payload, entity):
            if not isinstance(src, dict):
                continue
            value = src.get(key)
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            if isinstance(value, (list, dict)) and not value:
                continue
            return value
    return None


def _compact_json_value(value: Any, *, limit: int = 1800) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _trim_text(_normalize_snippet(value), limit)
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(value)
    return _trim_text(_normalize_snippet(text), limit)


def _normalize_label_values(value: Any, *, limit: int = 24) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str):
        sep = "|" if "|" in value else ","
        raw_items = value.split(sep) if sep in value else [value]
    else:
        raw_items = [value]
    out: list[str] = []
    seen: set[str] = set()
    for raw in raw_items:
        text = str(raw or "").strip().strip("'\"")
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(_trim_text(text, 80))
        if len(out) >= limit:
            break
    return out


def _index_text_for_report(chunk: dict) -> str:
    payload = _chunk_payload(chunk)
    text = str(payload.get("index_text") or "").strip()
    if not text:
        text = str(chunk.get("snippet") or "")
    return _normalize_snippet(text)


def _infer_model_name_from_text(text: str) -> str:
    for pattern in (
        r"\bmodel_name\s*[:=]\s*([^\n,，]+)",
        r"\bname\s*[:=]\s*([^\n,，]+)",
        r"\b(KM_[A-Za-z0-9_.\-/]+)",
    ):
        match = re.search(pattern, text)
        if match:
            return _trim_text(str(match.group(1) or "").strip(), 160)
    return ""


def _asset_key_for_chunk(chunk: dict, idx: int) -> str:
    model_name = str(
        _first_chunk_raw(
            chunk,
            "model_name",
            "canonical_model_name",
            "csv_model_name",
            "name",
        )
        or ""
    ).strip()
    if not model_name:
        model_name = _infer_model_name_from_text(_index_text_for_report(chunk))
    if model_name:
        return f"model:{model_name}"
    doc_id = str(_first_chunk_raw(chunk, "doc_id") or "").strip()
    if doc_id:
        return f"doc:{doc_id}"
    title = str(_first_chunk_raw(chunk, "title", "doc_name") or "").strip()
    if title:
        return f"title:{title}"
    return f"evidence:{str(chunk.get('evidence_id') or idx).strip()}"


def _asset_evidence_kind(chunk: dict, index_text: str) -> list[str]:
    section = str(_first_chunk_raw(chunk, "section_type") or "").lower()
    haystack = f"{section}\n{index_text}".lower()
    kinds: list[str] = []
    if any(x in haystack for x in ("label_schema", "label_list", "labels", "标签", "类别", "输出字段", "category_param")):
        kinds.append("label_schema")
    if any(x in haystack for x in ("performance", "benchmark", "metric", "map", "acc", "precision", "recall", "精度", "指标")):
        kinds.append("reported_metrics")
    if any(x in haystack for x in ("training", "dataset", "样本", "标注", "训练", "数据集", "数据要求")):
        kinds.append("training_data")
    if not kinds:
        kinds.append("general")
    return kinds


def _compact_asset_evidence(chunk: dict, idx: int) -> dict:
    text = _index_text_for_report(chunk)
    out: dict = {
        "evidence_id": str(
            chunk.get("evidence_id")
            or chunk.get("legacy_evidence_id")
            or _chunk_payload(chunk).get("chunk_id")
            or idx
        ).strip(),
        "source_type": _trim_text(str(_first_chunk_raw(chunk, "source_type") or ""), 64),
        "section_type": _trim_text(str(_first_chunk_raw(chunk, "section_type") or ""), 80),
        "evidence_kind": _asset_evidence_kind(chunk, text),
        "index_text": text,
    }
    for key, limit in (
        ("title", 160),
        ("doc_name", 160),
        ("page_label", 40),
    ):
        value = str(_first_chunk_raw(chunk, key) or "").strip()
        if value:
            out[key] = _trim_text(value, limit)
    return {k: v for k, v in out.items() if v not in ("", [], {})}


def _asset_identity_from_chunks(chunks: list[dict], fallback_name: str) -> dict:
    primary = chunks[0] if chunks else {}
    index_text = _index_text_for_report(primary)
    model_name = str(
        _first_chunk_raw(primary, "model_name", "canonical_model_name", "csv_model_name", "name")
        or _infer_model_name_from_text(index_text)
        or ""
    ).strip()
    display_name = str(_first_chunk_raw(primary, "title", "doc_name") or "").strip()
    asset: dict = {
        "asset_name": _trim_text(model_name or display_name or fallback_name, 160),
    }
    for out_key, keys in (
        ("model_name", ("model_name", "canonical_model_name", "csv_model_name", "name")),
        ("target_name", ("target_name", "canonical_target_name")),
        ("algorithm_name", ("algorithm_name", "canonical_algorithm_name")),
        ("algorithm_type", ("algorithm_type", "canonical_algorithm_type")),
        ("application_scene", ("application_scene", "canonical_application_scene")),
    ):
        value = str(_first_chunk_raw(primary, *keys) or "").strip()
        if value:
            asset[out_key] = _trim_text(value, 160)
    source_types = []
    doc_names = []
    for chunk in chunks:
        source_type = str(_first_chunk_raw(chunk, "source_type") or "").strip()
        doc_name = str(_first_chunk_raw(chunk, "doc_name") or "").strip()
        if source_type and source_type not in source_types:
            source_types.append(_trim_text(source_type, 64))
        if doc_name and doc_name not in doc_names:
            doc_names.append(_trim_text(doc_name, 160))
    if source_types:
        asset["source_types"] = source_types[:4]
    if doc_names:
        asset["doc_names"] = doc_names[:4]
    labels = (
        _normalize_label_values(_first_chunk_raw(primary, "label_list", "labels", "class_labels"))
        or _normalize_label_values(_first_chunk_raw(primary, "canonical_label_list", "canonical_labels"))
    )
    if labels:
        asset["labels"] = labels
    benchmark = _first_chunk_raw(primary, "benchmark_info", "reported_metrics", "metrics")
    if benchmark:
        asset["benchmark_info"] = _compact_json_value(benchmark, limit=2000)
    return {k: v for k, v in asset.items() if v not in ("", [], {})}


def _build_report_assets(chunks: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for idx, chunk in enumerate(chunks if isinstance(chunks, list) else [], start=1):
        if not isinstance(chunk, dict):
            continue
        key = _asset_key_for_chunk(chunk, idx)
        grouped.setdefault(key, []).append(chunk)
    ranked_groups = sorted(
        grouped.items(),
        key=lambda item: max((_chunk_score(chunk) for chunk in item[1]), default=0.0),
        reverse=True,
    )
    assets: list[dict] = []
    for key, group in ranked_groups[:REPORT_ASSETS_PER_FIELD]:
        ranked_chunks = sorted(group, key=_chunk_score, reverse=True)
        fallback = key.split(":", 1)[-1] if ":" in key else key
        asset = _asset_identity_from_chunks(ranked_chunks, fallback)
        evidence = [
            _compact_asset_evidence(chunk, idx)
            for idx, chunk in enumerate(ranked_chunks[:REPORT_EVIDENCE_PER_ASSET], start=1)
        ]
        evidence = [item for item in evidence if item.get("index_text")]
        if evidence:
            asset["evidence"] = evidence
            for kind, out_key in (
                ("label_schema", "label_schema_evidence_ids"),
                ("reported_metrics", "metric_evidence_ids"),
                ("training_data", "training_data_evidence_ids"),
            ):
                ids = [
                    str(item.get("evidence_id") or "")
                    for item in evidence
                    if kind in (item.get("evidence_kind") if isinstance(item.get("evidence_kind"), list) else [])
                ]
                if ids:
                    asset[out_key] = ids[:REPORT_EVIDENCE_PER_ASSET]
        if asset.get("evidence"):
            assets.append(asset)
    return assets


def _clean_text_for_report_markdown(text: str) -> str:
    out = str(text or "")
    for ch in _QUOTE_CHARS:
        out = out.replace(ch, "")
    out = _PAREN_CONTENT.sub(" ", out)
    out = _ENGLISH_KV_INLINE.sub("", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _snippet_to_plain_text(snippet: str) -> str:
    lines: list[str] = []
    for raw in str(snippet or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        match = _ENGLISH_KV_LINE.match(line)
        if match:
            line = str(match.group(1) or "").strip()
        else:
            line = _ENGLISH_KV_INLINE.sub("", line).strip()
        line = _clean_text_for_report_markdown(line)
        if not line:
            continue
        if re.fullmatch(r"[A-Za-z0-9_.\-/]+", line):
            continue
        lines.append(line)
    if not lines:
        fallback = _clean_text_for_report_markdown(snippet)
        return fallback if fallback else ""
    return "\n".join(lines)


def _compact_chunk_for_report(chunk: dict, idx: int) -> dict:
    base = _compact_chunk(chunk, idx)
    base.pop("title", None)
    base.pop("score", None)
    snippet = _snippet_to_plain_text(str(base.pop("snippet", "") or ""))
    out: dict = {"evidence_id": str(base.get("evidence_id") or "").strip()}
    if snippet:
        out["snippet"] = _trim_text(snippet, None)
    for key in ("model_name", "algorithm_name", "algorithm_type", "application_scene"):
        value = _clean_text_for_report_markdown(str(base.get(key) or ""))
        if value and not re.fullmatch(r"[A-Za-z0-9_.\-/]+", value):
            out[key] = _trim_text(value, None)
    return out


def _compact_document_for_fact(doc: dict, idx: int) -> dict:
    if not isinstance(doc, dict):
        return {}
    content = str(doc.get("content") or "").strip()
    metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
    matched_ids = metadata.get("matched_evidence_ids")
    return {
        "doc_id": _trim_text(str(doc.get("doc_id") or f"doc_{idx}").strip(), 120),
        "doc_name": _trim_text(str(doc.get("doc_name") or "").strip(), 160),
        "source_type": _trim_text(str(doc.get("source_type") or "").strip(), 64),
        "matched_evidence_ids": (
            [str(x).strip() for x in matched_ids if str(x).strip()]
            if isinstance(matched_ids, list)
            else []
        )[:20],
        "content": _trim_text(content, None),
    }


def _md_line(label: str, value: str) -> str:
    text = _clean_text_for_report_markdown(value)
    return f"- {label}：{text}" if text else ""


def _report_input_to_markdown(
    *,
    user_query: str,
    plan: dict,
    field_results: list[dict],
    evidence_facts: list[dict],
) -> str:
    lines: list[str] = ["## 用户需求", _clean_text_for_report_markdown(user_query), ""]
    plan_obj = plan if isinstance(plan, dict) else {}
    abstract = (
        plan_obj.get("abstract_requirement")
        if isinstance(plan_obj.get("abstract_requirement"), dict)
        else {}
    )
    lines.extend(["## 检索规划", "### 需求抽象"])
    for label, key in (
        ("对象", "object"),
        ("属性", "attribute"),
        ("任务类型", "task_type"),
        ("场景", "scene"),
    ):
        row = _md_line(label, str(abstract.get(key) or ""))
        if row:
            lines.append(row)
    constraints = abstract.get("constraints")
    if isinstance(constraints, list):
        items = [
            _clean_text_for_report_markdown(str(x))
            for x in constraints
            if _clean_text_for_report_markdown(str(x))
        ]
        if items:
            lines.append(f"- 约束：{'；'.join(items)}")
    retrieve_fields = (
        plan_obj.get("retrieve_fields") if isinstance(plan_obj.get("retrieve_fields"), list) else []
    )
    for item in retrieve_fields:
        if not isinstance(item, dict):
            continue
        field = str(item.get("field") or "").strip()
        heading = _FIELD_LABEL_MAP.get(field, field)
        lines.extend(["", f"### {heading} 检索词"])
        queries = item.get("queries") if isinstance(item.get("queries"), list) else []
        for query in queries:
            q = _clean_text_for_report_markdown(str(query or ""))
            if q:
                lines.append(f"- {q}")
    lines.extend(["", "## 分字段检索概览"])
    for result in field_results:
        if not isinstance(result, dict):
            continue
        field = str(result.get("field") or "").strip()
        heading = _FIELD_LABEL_MAP.get(field, field)
        lines.extend(["", f"### {heading}"])
        coverage = _clean_text_for_report_markdown(str(result.get("coverage") or ""))
        if coverage:
            lines.append(f"- 检索覆盖：{coverage}")
        queries = result.get("queries") if isinstance(result.get("queries"), list) else []
        if queries:
            lines.append("- 检索词：")
            for query in queries:
                q = _clean_text_for_report_markdown(str(query or ""))
                if q:
                    lines.append(f"  - {q}")
        evidence_count = result.get("evidence_count")
        full_document_count = result.get("full_document_count")
        lines.append(f"- 命中证据数：{int(evidence_count or 0)}")
        lines.append(f"- 命中完整文档数：{int(full_document_count or 0)}")
    lines.extend(
        [
            "",
            "## 已校验结构化事实",
            "报告只能基于以下事实作答。若以下事实不足以支持某个结论，必须写“证据不足”。",
        ]
    )
    for idx, fact in enumerate(evidence_facts if isinstance(evidence_facts, list) else [], start=1):
        if not isinstance(fact, dict):
            continue
        lines.extend(["", f"### 事实 {idx}"])
        for label, key in (
            ("类型", "fact_type"),
            ("字段", "field"),
            ("主体", "subject"),
            ("事实", "fact"),
            ("数值", "value"),
            ("单位", "unit"),
            ("置信度", "confidence"),
        ):
            row = _md_line(label, str(fact.get(key) or ""))
            if row:
                lines.append(row)
        evidence_ids = fact.get("evidence_ids") if isinstance(fact.get("evidence_ids"), list) else []
        doc_ids = fact.get("doc_ids") if isinstance(fact.get("doc_ids"), list) else []
        if evidence_ids:
            lines.append("- 证据编号：" + "、".join(str(x) for x in evidence_ids))
        if doc_ids:
            lines.append("- 文档编号：" + "、".join(str(x) for x in doc_ids))
        quote = _clean_text_for_report_markdown(str(fact.get("quote") or ""))
        if quote:
            lines.append(f"- 原文摘录：{quote}")
    return "\n".join(lines).strip()


def _compact_rag_trace(trace: list[dict] | None, *, chunks_per_round: int = 6) -> list[dict]:
    out: list[dict] = []
    for item in (trace if isinstance(trace, list) else [])[-3:]:
        if not isinstance(item, dict):
            continue
        obs = item.get("observation") if isinstance(item.get("observation"), dict) else {}
        chunks = item.get("retrieved_chunks")
        if not isinstance(chunks, list) or not chunks:
            obs_chunks = obs.get("retrieved_chunks")
            chunks = obs_chunks if isinstance(obs_chunks, list) else []
        ranked = sorted(
            [c for c in chunks if isinstance(c, dict)],
            key=_chunk_score,
            reverse=True,
        )
        row: dict = {
            "round": item.get("round"),
            "query": str(item.get("query") or "").strip(),
            "knowledge_base_fully_answered": obs.get("knowledge_base_fully_answered"),
            "evidence": [
                _compact_chunk(chunk, idx)
                for idx, chunk in enumerate(ranked[:chunks_per_round], start=1)
            ],
        }
        out.append(row)
    return out


def _chunk_dedupe_key(chunk: dict) -> str:
    payload = chunk.get("payload") if isinstance(chunk.get("payload"), dict) else {}
    entity = payload.get("entity") if isinstance(payload.get("entity"), dict) else {}
    for key in ("evidence_id", "legacy_evidence_id", "id"):
        value = str(chunk.get(key) or payload.get(key) or entity.get(key) or "").strip()
        if value:
            return f"id:{value}"
    url = str(
        chunk.get("url")
        or payload.get("url")
        or payload.get("reference")
        or payload.get("link")
        or entity.get("ones_release_link")
        or ""
    ).strip()
    if url:
        return f"url:{url}"
    compact = json.dumps(_compact_chunk(chunk, 0), ensure_ascii=False, sort_keys=True)
    return "hash:" + compact[:500]


def merge_retrieve_chunks(results: list[dict], *, limit: int = 20) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for result in results:
        if not isinstance(result, dict):
            continue
        chunks = result.get("retrieved_chunks") if isinstance(result.get("retrieved_chunks"), list) else []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            key = _chunk_dedupe_key(chunk)
            if key in seen:
                continue
            seen.add(key)
            out.append(chunk)
            if len(out) >= limit:
                return out
    return out


def _doc_dedupe_key(doc: dict) -> str:
    doc_id = str(doc.get("doc_id") or "").strip()
    if doc_id:
        return f"doc:{doc_id}"
    source_path = str(doc.get("source_path") or "").strip()
    if source_path:
        return f"path:{source_path}"
    name = str(doc.get("doc_name") or "").strip()
    content = str(doc.get("content") or "").strip()
    return f"hash:{name}:{content[:300]}"


def merge_full_documents(results: list[dict], *, limit: int = 12) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for result in results:
        if not isinstance(result, dict):
            continue
        docs = result.get("full_documents") if isinstance(result.get("full_documents"), list) else []
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            key = _doc_dedupe_key(doc)
            if key in seen:
                continue
            seen.add(key)
            out.append(doc)
            if len(out) >= limit:
                return out
    return out


def _build_evidence_packages(field_results: list[dict]) -> list[dict]:
    packages: list[dict] = []
    for result in field_results:
        if not isinstance(result, dict):
            continue
        chunks = result.get("retrieved_chunks") if isinstance(result.get("retrieved_chunks"), list) else []
        docs = result.get("full_documents") if isinstance(result.get("full_documents"), list) else []
        packages.append(
            {
                "field": str(result.get("field") or "").strip(),
                "queries": result.get("queries") if isinstance(result.get("queries"), list) else [],
                "coverage": str(result.get("coverage") or "").strip(),
                "evidences": [
                    _compact_chunk_for_report(chunk, idx)
                    for idx, chunk in enumerate(chunks, start=1)
                    if isinstance(chunk, dict)
                ],
                "full_documents": [
                    _compact_document_for_fact(doc, idx)
                    for idx, doc in enumerate(docs, start=1)
                    if isinstance(doc, dict)
                ],
            }
        )
    return packages


def _valid_fact_ids(evidence_packages: list[dict]) -> tuple[set[str], set[str]]:
    evidence_ids: set[str] = set()
    doc_ids: set[str] = set()
    for package in evidence_packages:
        if not isinstance(package, dict):
            continue
        for item in package.get("evidences") if isinstance(package.get("evidences"), list) else []:
            if not isinstance(item, dict):
                continue
            eid = str(item.get("evidence_id") or "").strip()
            if eid:
                evidence_ids.add(eid)
        for item in package.get("full_documents") if isinstance(package.get("full_documents"), list) else []:
            if not isinstance(item, dict):
                continue
            did = str(item.get("doc_id") or "").strip()
            if did:
                doc_ids.add(did)
    return evidence_ids, doc_ids


def _normalize_fact_list(raw_facts, *, evidence_packages: list[dict], limit: int = 60) -> list[dict]:
    if not isinstance(raw_facts, list):
        return []
    valid_eids, valid_dids = _valid_fact_ids(evidence_packages)
    valid_fields = {item["field"] for item in MIGRATION_FIELDS}
    valid_types = {
        "model_identity",
        "task_scope",
        "category_count",
        "accuracy",
        "performance",
        "model_size",
        "data_scale",
        "deployment",
        "limitation",
        "cost_or_timeline",
        "other",
    }
    out: list[dict] = []
    seen: set[str] = set()
    for item in raw_facts:
        if not isinstance(item, dict):
            continue
        fact = _trim_text(str(item.get("fact") or "").strip(), None)
        quote = _trim_text(str(item.get("quote") or "").strip(), None)
        if not fact or not quote:
            continue
        evidence_ids = [
            str(x).strip()
            for x in (item.get("evidence_ids") if isinstance(item.get("evidence_ids"), list) else [])
            if str(x).strip() and (not valid_eids or str(x).strip() in valid_eids)
        ]
        doc_ids = [
            str(x).strip()
            for x in (item.get("doc_ids") if isinstance(item.get("doc_ids"), list) else [])
            if str(x).strip() and (not valid_dids or str(x).strip() in valid_dids)
        ]
        if not evidence_ids and not doc_ids:
            continue
        field = str(item.get("field") or "").strip()
        if field not in valid_fields:
            field = "similar_capabilities"
        fact_type = str(item.get("fact_type") or "").strip()
        if fact_type not in valid_types:
            fact_type = "other"
        confidence = str(item.get("confidence") or "").strip()
        if confidence not in {"high", "medium", "low"}:
            confidence = "medium"
        normalized = {
            "fact_type": fact_type,
            "field": field,
            "subject": _trim_text(str(item.get("subject") or "").strip(), None),
            "fact": fact,
            "value": _trim_text(str(item.get("value") or "").strip(), None),
            "unit": _trim_text(str(item.get("unit") or "").strip(), None),
            "evidence_ids": evidence_ids[:8],
            "doc_ids": doc_ids[:8],
            "quote": quote,
            "confidence": confidence,
        }
        key = json.dumps(
            {
                "fact_type": normalized["fact_type"],
                "subject": normalized["subject"],
                "fact": normalized["fact"],
                "evidence_ids": normalized["evidence_ids"],
                "doc_ids": normalized["doc_ids"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(normalized)
        if len(out) >= limit:
            break
    return out


def extract_evidence_facts(
    *,
    user_query: str,
    plan: dict,
    field_results: list[dict],
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    debug_meta: dict | None = None,
) -> list[dict]:
    evidence_packages = _build_evidence_packages(field_results)
    if not any(
        (package.get("evidences") or package.get("full_documents"))
        for package in evidence_packages
        if isinstance(package, dict)
    ):
        return []
    structured_input = {
        "user_query": str(user_query or "").strip(),
        "retrieval_plan": plan if isinstance(plan, dict) else {},
        "evidence_packages": evidence_packages,
    }
    system = (
        "你是迁移顾问的证据事实抽取器，只能从输入的 evidences 和 full_documents 中抽取事实。"
        "重点抽取：已有模型/方案身份、任务范围、类别数、精度指标、性能指标、模型体量、数据规模、部署平台、限制条件、周期/成本。"
        "每条事实必须包含原文摘录 quote，并引用存在的 evidence_ids 或 doc_ids。"
        "不要生成建议，不要做未被文本支持的推断；文本没有明确说的内容不要输出。只输出 JSON。"
    )
    user = json.dumps(structured_input, ensure_ascii=False, indent=2)
    use_model = model or MIGRATION_ADVISOR_MODEL
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    _write_migration_llm_debug(
        debug_meta=debug_meta,
        stage="migration_fact_extraction_request",
        step_index=2,
        payload=_debug_request_payload(
            model=use_model,
            messages=messages,
            run_dir=str((debug_meta or {}).get("run_dir") or ""),
            structured_input=structured_input,
            input_schema=FACT_EXTRACTION_INPUT_SCHEMA,
            evidence_item_fields=COMPACT_EVIDENCE_FIELDS + ["full_documents"],
        ),
    )
    raw = _generate_migration_text(
        messages=messages,
        model=use_model,
        response_format=FACT_EXTRACTION_RESPONSE_FORMAT,
        api_key=api_key,
        base_url=base_url,
    )
    data = _extract_first_json_object(raw)
    facts = _normalize_fact_list(
        data.get("facts") if isinstance(data, dict) else [],
        evidence_packages=evidence_packages,
    )
    _write_migration_llm_debug(
        debug_meta=debug_meta,
        stage="migration_fact_extraction_response",
        step_index=2,
        payload=_debug_response_payload(
            model=use_model,
            raw=raw,
            output_schema=FACT_EXTRACTION_OUTPUT_SCHEMA,
            structured_output={"facts": facts},
            extracted_count=len(facts),
        ),
    )
    return facts


def extract_detection_targets(
    *,
    user_query: str,
    plan: dict,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    debug_meta: dict | None = None,
) -> dict:
    """用 LLM 从用户需求与检索规划中抽取 Rex-Omni 检测标签（无类别硬编码）。"""
    use_model = model or MIGRATION_ADVISOR_MODEL
    plan_obj = plan if isinstance(plan, dict) else {}
    structured_input = {
        "user_query": str(user_query or "").strip(),
        "retrieval_plan": plan_obj,
    }
    _write_migration_llm_debug(
        debug_meta=debug_meta,
        stage="migration_rex_label_extraction_request",
        step_index=2,
        payload=_debug_request_payload(
            model=use_model,
            messages=[
                {
                    "role": "system",
                    "content": "从用户需求抽取 Rex-Omni 开集检测目标（class_key/display_label/tokens）。",
                },
                {"role": "user", "content": json.dumps(structured_input, ensure_ascii=False, indent=2)},
            ],
            run_dir=str((debug_meta or {}).get("run_dir") or ""),
            structured_input=structured_input,
            input_schema={
                "user_query": "用户原始问题",
                "retrieval_plan": "迁移检索规划 abstract_requirement + retrieve_fields",
            },
        ),
    )
    extracted = extract_rex_detection_labels(
        str(user_query or "").strip(),
        plan=plan_obj,
        api_key=api_key,
        base_url=base_url,
        model=use_model,
    )
    _write_migration_llm_debug(
        debug_meta=debug_meta,
        stage="migration_rex_label_extraction_response",
        step_index=2,
        payload=_debug_response_payload(
            model=use_model,
            raw="",
            output_schema={
                "task_mode": "single_object | multi_class",
                "display_label": "任务简述",
                "object": "检测对象",
                "classes": "每项 class_key/label/tokens",
            },
            structured_output=extracted,
        ),
    )
    return extracted


def _resolve_rex_vis_paths(
    *,
    annotated_image_paths: list[str] | None = None,
    annotated_image_path: str | None = None,
    image_path: str = "",
    run_dir: str | Path | None = None,
    max_images: int = 10,
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        p = str(raw or "").strip()
        if not p or not Path(p).is_file():
            return
        key = str(Path(p).resolve())
        if key in seen:
            return
        seen.add(key)
        out.append(key)

    for raw in annotated_image_paths if isinstance(annotated_image_paths, list) else []:
        _add(str(raw or ""))
        if len(out) >= max_images:
            return out
    if run_dir:
        viz_dir = Path(run_dir) / "migration_rex_annotated"
        if viz_dir.is_dir():
            for p in sorted(viz_dir.iterdir(), key=lambda x: x.name):
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    _add(str(p.resolve()))
                    if len(out) >= max_images:
                        return out
    _add(str(annotated_image_path or ""))
    _add(str(image_path or ""))
    return out[:max_images]


def _build_rex_pred_reports(
    *,
    pred_reports: list[dict] | None,
    per_image: list[dict] | None,
    pred_bboxes: list[list[float]] | None,
    fallback_image: str,
) -> list[dict]:
    rows: list[dict] = []
    if isinstance(pred_reports, list):
        for rec in pred_reports:
            if not isinstance(rec, dict):
                continue
            models = rec.get("models") if isinstance(rec.get("models"), list) else []
            boxes: list = []
            if models and isinstance(models[0], dict):
                raw_boxes = models[0].get("pred_bboxes")
                if isinstance(raw_boxes, list):
                    boxes = raw_boxes
            rows.append(
                {
                    "image_idx": rec.get("image_idx"),
                    "image": str(rec.get("image") or "").strip(),
                    "source": str(rec.get("source") or "original").strip(),
                    "model": "rex-omni",
                    "pred_bboxes": boxes,
                    "num_boxes": len(boxes),
                }
            )
    if rows:
        return rows
    if isinstance(per_image, list) and per_image:
        for idx, row in enumerate(per_image):
            if not isinstance(row, dict):
                continue
            fname = str(row.get("file_name") or "").strip()
            rows.append(
                {
                    "image_idx": idx,
                    "image": fname or f"image_{idx}",
                    "source": "original",
                    "model": "rex-omni",
                    "pred_bboxes": [],
                    "num_boxes": int(row.get("num_boxes") or 0),
                }
            )
        return rows
    boxes = pred_bboxes if isinstance(pred_bboxes, list) else []
    name = Path(fallback_image).name if fallback_image else "image_0"
    return [
        {
            "image_idx": 0,
            "image": name,
            "source": "original",
            "model": "rex-omni",
            "pred_bboxes": boxes,
            "num_boxes": len(boxes),
        }
    ]


def estimate_rex_accuracy(
    *,
    image_path: str = "",
    annotated_image_path: str | None = None,
    annotated_image_paths: list[str] | None = None,
    user_query: str,
    detection_label: str,
    pred_bboxes: list[list[float]] | None = None,
    pred_reports: list[dict] | None = None,
    label_hit_summary: dict | None = None,
    per_image: list[dict] | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    debug_meta: dict | None = None,
) -> dict:
    meta = debug_meta if isinstance(debug_meta, dict) else {}
    vis_paths = _resolve_rex_vis_paths(
        annotated_image_paths=annotated_image_paths,
        annotated_image_path=annotated_image_path,
        image_path=image_path,
        run_dir=meta.get("run_dir"),
        max_images=10,
    )
    if not vis_paths:
        return {"accuracy": "证据不足", "reason": "缺少可用于视觉评估的标注图片。"}

    report_rows = _build_rex_pred_reports(
        pred_reports=pred_reports,
        per_image=per_image,
        pred_bboxes=pred_bboxes,
        fallback_image=vis_paths[0],
    )
    reports_json = json.dumps(report_rows, ensure_ascii=False, indent=2)
    multi = len(vis_paths) > 1
    system = (
        "你是视觉检测标注质量评估助手。请结合用户上传的标注图片（含 Rex-Omni 预测框）和预测框统计，"
        "对 Rex-Omni 开集检测标注效果做定性评估，给出预估准确率（如 85% 或 70-80%）和 2-4 句中文依据。"
        + (
            "本次包含多张图片，请综合所有图片的表现给出整体准确率，并说明是否存在某些图片明显更差。"
            if multi
            else "说明图中大约有多少目标、模型预测了多少框、框是否覆盖目标、是否存在明显漏检/误检/过检。"
        )
        + "无真实 GT 时基于视觉合理性估计，不要过于乐观。只输出 JSON。"
    )
    hit_json = json.dumps(
        {
            "label_hit_summary": label_hit_summary if isinstance(label_hit_summary, dict) else {},
            "per_image": per_image if isinstance(per_image, list) else [],
        },
        ensure_ascii=False,
        indent=2,
    )
    user = (
        f"用户需求：{str(user_query or '').strip()}\n"
        f"检测目标：{str(detection_label or '').strip()}\n"
        f"共 {len(vis_paths)} 张标注图（含预测框），请一并查看。\n"
        f"Rex-Omni 预测统计：\n{reports_json}\n"
        f"按类别命中情况（每图是否检出各 label）：\n{hit_json}"
    )
    use_model = model or MIGRATION_ADVISOR_MODEL
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    _write_migration_llm_debug(
        debug_meta=debug_meta,
        stage="migration_rex_accuracy_request",
        step_index=2,
        payload=_debug_request_payload(
            model=use_model,
            messages=messages,
            run_dir=str((debug_meta or {}).get("run_dir") or ""),
            structured_input={
                "user_query": user_query,
                "detection_label": detection_label,
                "pred_reports": report_rows,
                "image_paths": vis_paths,
                "image_count": len(vis_paths),
            },
            input_schema=REX_ACCURACY_OUTPUT_SCHEMA,
        ),
    )
    try:
        raw = _generate_migration_text(
            messages=messages,
            model=use_model,
            response_format=REX_ACCURACY_RESPONSE_FORMAT,
            api_key=api_key,
            base_url=base_url,
            image_paths=vis_paths,
        )
        data = _extract_first_json_object(raw)
        out = {
            "accuracy": str(data.get("accuracy") or "").strip() or "证据不足",
            "reason": str(data.get("reason") or "").strip() or "未能生成评估依据。",
        }
        _write_migration_llm_debug(
            debug_meta=debug_meta,
            stage="migration_rex_accuracy_response",
            step_index=2,
            payload=_debug_response_payload(
                model=use_model,
                raw=raw,
                output_schema=REX_ACCURACY_OUTPUT_SCHEMA,
                structured_output=out,
            ),
        )
        return out
    except Exception as exc:
        _write_migration_llm_debug(
            debug_meta=debug_meta,
            stage="migration_rex_accuracy_response",
            step_index=2,
            payload=_debug_response_payload(
                model=use_model,
                raw="",
                output_schema=REX_ACCURACY_OUTPUT_SCHEMA,
                error=str(exc),
            ),
        )
        return {
            "accuracy": "证据不足",
            "reason": f"准确率预估失败：{exc}",
        }


def rex_result_to_markdown(rex_result: dict | None) -> str:
    r = rex_result if isinstance(rex_result, dict) else {}
    if not r:
        return ""
    label = str(r.get("label") or "").strip()
    det = r.get("detection_targets") if isinstance(r.get("detection_targets"), dict) else {}
    classes = det.get("classes") if isinstance(det.get("classes"), list) else []
    task_mode = str(det.get("task_mode") or r.get("task_mode") or "").strip()
    num_boxes = int(r.get("num_boxes") or 0)
    acc = r.get("accuracy_estimate") if isinstance(r.get("accuracy_estimate"), dict) else {}
    accuracy = str(acc.get("accuracy") or "").strip()
    reason = str(acc.get("reason") or "").strip()
    lines = [
        "### Rex-Omni 模型标注",
        "",
        f"**检测任务**：{label or '未指定'}",
        f"**任务模式**：{'多类别' if task_mode == 'multi_class' else '单目标' if task_mode else '未知'}",
        f"**预测框总数**：{num_boxes}",
    ]
    if classes:
        class_labels = [
            f"{str(c.get('label') or '').strip()}({str(c.get('class_key') or '').strip()})"
            for c in classes
            if isinstance(c, dict) and str(c.get("label") or "").strip()
        ]
        if class_labels:
            lines.append(f"**待检类别（{len(class_labels)}）**：{', '.join(class_labels)}")
    per_image = r.get("per_image") if isinstance(r.get("per_image"), list) else []
    if per_image:
        lines.extend(["", "**各图类别命中**"])
        for row in per_image:
            if not isinstance(row, dict):
                continue
            fname = str(row.get("file_name") or "").strip()
            hits = row.get("hit_labels") if isinstance(row.get("hit_labels"), list) else []
            misses = row.get("miss_labels") if isinstance(row.get("miss_labels"), list) else []
            lines.append(
                f"- {fname or '图片'}：命中 {', '.join(str(x) for x in hits) if hits else '无'}；"
                f"未命中 {', '.join(str(x) for x in misses) if misses else '无'}"
            )
    if accuracy:
        lines.append(f"**准确率预估**：{accuracy}")
    if reason:
        lines.append(f"**评估说明**：{reason}")
    err = str(r.get("error") or "").strip()
    if err:
        lines.append(f"**备注**：{err}")
    return "\n".join(lines).strip()


def _parse_accuracy_percent(text: str) -> float | None:
    """从准确率/mAP/ACC 等文本解析为 0-100 的百分比；无法解析则返回 None。"""
    raw = str(text or "").strip()
    if not raw or "证据不足" in raw:
        return None
    range_m = re.search(r"(\d+(?:\.\d+)?)\s*[-~～至]\s*(\d+(?:\.\d+)?)", raw)
    if range_m:
        nums = [float(range_m.group(1)), float(range_m.group(2))]
    else:
        nums = [float(x) for x in _ACCURACY_NUMBER.findall(raw)]
    if not nums:
        return None
    lo, hi = min(nums), max(nums)
    if hi <= 1.0:
        pct = ((lo + hi) / 2) * 100 if len(nums) > 1 else hi * 100
    else:
        pct = (lo + hi) / 2 if len(nums) > 1 else hi
    return max(0.0, min(100.0, pct))


def _rex_accuracy_snapshot(rex_result: dict | None) -> dict:
    r = rex_result if isinstance(rex_result, dict) else {}
    if not r.get("success"):
        return {}
    acc = r.get("accuracy_estimate") if isinstance(r.get("accuracy_estimate"), dict) else {}
    accuracy_text = str(acc.get("accuracy") or "").strip()
    parsed = _parse_accuracy_percent(accuracy_text)
    if not accuracy_text and parsed is None:
        return {}
    return {
        "detection_label": str(r.get("label") or "").strip(),
        "accuracy": accuracy_text or (f"{parsed:.0f}%" if parsed is not None else ""),
        "accuracy_percent": parsed,
        "reason": str(acc.get("reason") or "").strip(),
        "num_boxes": int(r.get("num_boxes") or 0),
    }


def _best_similar_model_accuracy(report: dict) -> tuple[float | None, str, str]:
    best_pct: float | None = None
    best_name = ""
    best_metrics = ""
    for item in report.get("similar_assets") if isinstance(report.get("similar_assets"), list) else []:
        if not isinstance(item, dict):
            continue
        metrics = str(item.get("reported_metrics") or "").strip()
        if not metrics or "证据不足" in metrics:
            continue
        pct = _parse_accuracy_percent(metrics)
        if pct is None:
            continue
        if best_pct is None or pct > best_pct:
            best_pct = pct
            best_name = str(item.get("model_or_solution") or "").strip()
            best_metrics = metrics
    return best_pct, best_name, best_metrics


def _apply_accuracy_comparison_to_report(report: dict, rex_result: dict | None) -> dict:
    """对比 Rex-Omni 样例预估与相似模型 reported_metrics，写入方案与建议。"""
    out = dict(report) if isinstance(report, dict) else {}
    snap = _rex_accuracy_snapshot(rex_result)
    rex_pct = snap.get("accuracy_percent")
    if rex_pct is None:
        return out
    sim_pct, sim_name, sim_metrics = _best_similar_model_accuracy(out)
    if sim_pct is None:
        return out

    rex_text = str(snap.get("accuracy") or f"{rex_pct:.0f}%").strip()
    if sim_pct > rex_pct:
        compare_rec = (
            f"知识库相似模型「{sim_name or '未命名'}」报告精度为 {sim_metrics}（约 {sim_pct:.0f}%），"
            f"高于 Rex-Omni 在用户样例上的预估 {rex_text}（约 {rex_pct:.0f}%）。"
            "说明现有或可迁移的自训练模型在该类任务上更具优势，建议优先基于检索方案进行模型训练或微调。"
        )
        approach_note = (
            f"精度对比：检索相似模型（约 {sim_pct:.0f}%）高于 Rex-Omni 样例预估（约 {rex_pct:.0f}%），"
            "倾向训练/部署自有模型。"
        )
    elif rex_pct > sim_pct:
        compare_rec = (
            f"Rex-Omni 在用户样例上的预估为 {rex_text}（约 {rex_pct:.0f}%），"
            f"高于知识库相似模型「{sim_name or '未命名'}」的 {sim_metrics}（约 {sim_pct:.0f}%）。"
            "建议先使用 Rex-Omni 等开源开集模型快速验证与上线，同时构建标注数据，"
            "待场景与数据稳定后再训练专用模型。"
        )
        approach_note = (
            f"精度对比：Rex-Omni 样例预估（约 {rex_pct:.0f}%）高于检索相似模型（约 {sim_pct:.0f}%），"
            "可先采用开源模型，再积累数据训练专用模型。"
        )
    else:
        compare_rec = (
            f"Rex-Omni 预估（{rex_text}，约 {rex_pct:.0f}%）与相似模型「{sim_name or '未命名'}」"
            f"（{sim_metrics}，约 {sim_pct:.0f}%）接近。"
            "可按交付周期选择：短期用开源开集模型验证，中期并行数据积累与自训练。"
        )
        approach_note = (
            f"精度对比：Rex-Omni（约 {rex_pct:.0f}%）与检索模型（约 {sim_pct:.0f}%）接近，可并行评估两条路径。"
        )

    existing_rec = str(out.get("recommendation") or "").strip()
    out["recommendation"] = f"{compare_rec}\n\n{existing_rec}" if existing_rec else compare_rec

    plan = dict(out.get("migration_plan")) if isinstance(out.get("migration_plan"), dict) else {}
    old_approach = str(plan.get("approach") or "").strip()
    plan["approach"] = f"{approach_note} {old_approach}".strip() if old_approach else approach_note
    out["migration_plan"] = plan

    perf = dict(out.get("expected_performance")) if isinstance(out.get("expected_performance"), dict) else {}
    perf["baseline"] = (
        f"相似模型「{sim_name}」：{sim_metrics}" if sim_name else f"相似模型精度：{sim_metrics}"
    )
    perf["target"] = f"Rex-Omni 样例预估：{rex_text}"
    if not str(perf.get("uncertainty") or "").strip():
        perf["uncertainty"] = "中（基于样例图与文档指标对比，非同集实测）"
    out["expected_performance"] = perf
    return out


def build_report(
    *,
    user_query: str,
    plan: dict,
    field_results: list[dict],
    rex_result: dict | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    debug_meta: dict | None = None,
) -> dict:
    compact_fields = []
    for result in field_results:
        if not isinstance(result, dict):
            continue
        chunks = result.get("retrieved_chunks") if isinstance(result.get("retrieved_chunks"), list) else []
        assets = _build_report_assets(chunks)
        compact_fields.append(
            {
                "field": str(result.get("field") or "").strip(),
                "assets": assets,
            }
        )
    rex_snap = _rex_accuracy_snapshot(rex_result)
    system = REPORT_SYSTEM_PROMPT + (REPORT_REX_ACCURACY_COMPARE_PROMPT if rex_snap else "")
    structured_input = {
        "user_query": str(user_query or "").strip(),
        "field_results": compact_fields,
    }
    if rex_snap:
        structured_input["rex_omni_benchmark"] = rex_snap
    user = json.dumps(structured_input, ensure_ascii=False, indent=2)
    use_model = model or MIGRATION_REPORT_MODEL
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    _write_migration_llm_debug(
        debug_meta=debug_meta,
        stage="migration_report_request",
        step_index=2,
        payload=_debug_request_payload(
            model=use_model,
            messages=messages,
            run_dir=str((debug_meta or {}).get("run_dir") or ""),
            structured_input=structured_input,
            input_schema=REPORT_INPUT_SCHEMA,
            response_format=REPORT_RESPONSE_FORMAT,
            assistant_input_name="migration_advisor_report",
            evidence_item_fields=REPORT_EVIDENCE_FIELDS,
        ),
    )
    raw = _generate_migration_text(
        messages=messages,
        model=use_model,
        response_format=REPORT_RESPONSE_FORMAT,
        api_key=api_key,
        base_url=base_url,
    )
    report = _extract_first_json_object(raw)
    _write_migration_llm_debug(
        debug_meta=debug_meta,
        stage="migration_report_response",
        step_index=2,
        payload=_debug_response_payload(
            model=use_model,
            raw=raw,
            output_schema=REPORT_OUTPUT_SCHEMA,
            structured_output=report,
        ),
    )
    return report


def _referenced_fact_ids(evidence_facts: list[dict]) -> set[str]:
    ids: set[str] = set()
    for fact in evidence_facts if isinstance(evidence_facts, list) else []:
        if not isinstance(fact, dict):
            continue
        for key in ("evidence_ids", "doc_ids"):
            values = fact.get(key)
            if not isinstance(values, list):
                continue
            ids.update(str(x).strip() for x in values if str(x).strip())
    return ids


def enforce_report_evidence_bounds(report: dict, evidence_facts: list[dict]) -> dict:
    out = dict(report) if isinstance(report, dict) else {}
    allowed = _referenced_fact_ids(evidence_facts)
    has_facts = bool(allowed)

    def _filter_ids(values) -> list[str]:
        if not isinstance(values, list):
            return []
        return [str(x).strip() for x in values if str(x).strip() and str(x).strip() in allowed]

    direct = out.get("direct_match") if isinstance(out.get("direct_match"), dict) else {}
    direct_evidence = _filter_ids(direct.get("evidence"))
    if not has_facts:
        direct = {
            "exists": False,
            "summary": "证据不足，未抽取到可支撑直接匹配判断的事实。",
            "evidence": [],
        }
    else:
        direct["evidence"] = direct_evidence
        if bool(direct.get("exists")) and not direct_evidence:
            direct["exists"] = False
            direct["summary"] = "证据不足，未找到可引用事实支撑直接匹配。"
    out["direct_match"] = direct

    assets_out: list[dict] = []
    for item in out.get("similar_assets") if isinstance(out.get("similar_assets"), list) else []:
        if not isinstance(item, dict):
            continue
        ev = _filter_ids(item.get("evidence"))
        if not ev:
            continue
        row = dict(item)
        row["evidence"] = ev
        assets_out.append(row)
    out["similar_assets"] = assets_out

    if not has_facts:
        out["expected_performance"] = {
            "baseline": "证据不足",
            "target": "证据不足",
            "uncertainty": "高",
        }
        plan = out.get("migration_plan") if isinstance(out.get("migration_plan"), dict) else {}
        plan.update(
            {
                "feasibility": "low",
                "approach": "证据不足，无法基于现有检索结果形成可靠迁移路径。",
                "data_requirements": "证据不足",
                "compute_requirements": "证据不足",
                "engineering_work": "证据不足",
                "estimated_timeline": "证据不足",
                "estimated_cost": "证据不足",
                "dependencies": [],
                "risks": ["未抽取到可引用事实"],
            }
        )
        out["migration_plan"] = plan
        out["recommendation"] = "建议补充更明确的业务场景、样例图片、标签体系和目标指标后重新检索评估。"
    return out


def report_to_markdown(report: dict) -> str:
    r = report if isinstance(report, dict) else {}
    dm = r.get("direct_match") if isinstance(r.get("direct_match"), dict) else {}
    plan = r.get("migration_plan") if isinstance(r.get("migration_plan"), dict) else {}
    perf = r.get("expected_performance") if isinstance(r.get("expected_performance"), dict) else {}
    lines = [
        "### 迁移顾问报告",
        "",
        f"**需求摘要**：{str(r.get('requirement_summary') or '').strip() or '未提取到明确摘要'}",
        "",
        f"**是否存在直接同款**：{'是' if bool(dm.get('exists')) else '否'}",
        str(dm.get("summary") or "").strip(),
        "",
        "**相似模型（标签 / 训练数据 / 精度）**",
    ]
    assets = r.get("similar_assets") if isinstance(r.get("similar_assets"), list) else []
    if assets:
        for item in assets[:6]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("model_or_solution") or "未命名资产").strip()
            lines.append(f"- **{name}**")
            lines.append(
                f"  - 标签/类别：{str(item.get('label_schema') or '证据不足').strip()}"
            )
            lines.append(
                f"  - 训练数据：{str(item.get('training_data') or '证据不足').strip()}"
            )
            lines.append(
                f"  - 精度指标：{str(item.get('reported_metrics') or '证据不足').strip()}"
            )
            lines.append(
                f"  - 可覆盖能力：{str(item.get('covered_capability') or '未知').strip()}；"
                f"缺口：{str(item.get('gap') or '未说明').strip()}"
            )
            ev = item.get("evidence") if isinstance(item.get("evidence"), list) else []
            if ev:
                lines.append(f"  - 依据：{'; '.join(str(x).strip() for x in ev[:3] if str(x).strip())}")
    else:
        lines.append("- 未从证据中提取到可明确复用的相似模型。")
    deps = plan.get("dependencies") if isinstance(plan.get("dependencies"), list) else []
    risks = plan.get("risks") if isinstance(plan.get("risks"), list) else []
    lines.extend(
        [
            "",
            "**迁移方案**",
            f"- 可行性：{str(plan.get('feasibility') or 'unknown')}",
            f"- 技术路径：{str(plan.get('approach') or '证据不足，需补充评估')}",
            f"- 数据要求：{str(plan.get('data_requirements') or '证据不足')}",
            f"- 算力要求：{str(plan.get('compute_requirements') or '证据不足')}",
            f"- 工程工作：{str(plan.get('engineering_work') or '证据不足')}",
            f"- 周期估计：{str(plan.get('estimated_timeline') or '证据不足')}",
            f"- 成本估计：{str(plan.get('estimated_cost') or '证据不足')}",
            f"- 依赖项：{', '.join(str(x) for x in deps) if deps else '证据不足'}",
            f"- 风险：{', '.join(str(x) for x in risks) if risks else '证据不足'}",
            "",
            "**预期效果**",
            f"- 基线：{str(perf.get('baseline') or '证据不足')}",
            f"- 目标：{str(perf.get('target') or '证据不足')}",
            f"- 不确定性：{str(perf.get('uncertainty') or '证据不足')}",
            "",
            f"**建议**：{str(r.get('recommendation') or '').strip() or '建议补充业务场景、样例图片、标注规范和目标指标后再做立项评估。'}",
        ]
    )
    return "\n".join(lines).strip()


def run_workflow(
    *,
    user_query: str,
    rag_trace: list[dict] | None,
    retrieve: Callable[[str, str], dict],
    run_dir: Path,
    image_path: str | None = None,
    image_paths: list[str] | None = None,
    run_rex_annotation: Callable[[list[str], dict], dict] | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    debug_meta: dict | None = None,
    emit: Callable[[dict], None] | None = None,
) -> dict:
    emit = emit or (lambda _obj: None)
    meta = dict(debug_meta) if isinstance(debug_meta, dict) else {}
    meta.setdefault("run_dir", str(run_dir))
    plan = build_retrieval_plan(
        user_query=user_query,
        rag_trace=rag_trace,
        api_key=api_key,
        base_url=base_url,
        debug_meta=meta,
    )
    emit({"type": "migration_advisor_plan", "plan": plan})
    field_results: list[dict] = []
    for field in plan.get("retrieve_fields") if isinstance(plan.get("retrieve_fields"), list) else []:
        if not isinstance(field, dict):
            continue
        name = str(field.get("field") or "").strip()
        raw_queries = field.get("queries")
        if not isinstance(raw_queries, list):
            raw_queries = [field.get("query")]
        queries = _dedupe_queries([str(q or "") for q in raw_queries], limit=4)
        if not name or not queries:
            continue
        query_results: list[dict] = []
        for query in queries:
            emit({"type": "migration_advisor_retrieve", "field": name, "query": query, "status": "running"})
            result_raw = retrieve(name, query)
            result = result_raw if isinstance(result_raw, dict) else {}
            query_results.append(result)
            chunks_count = len(result.get("retrieved_chunks")) if isinstance(result.get("retrieved_chunks"), list) else 0
            emit(
                {
                    "type": "migration_advisor_retrieve",
                    "field": name,
                    "query": query,
                    "status": "query_done",
                    "count": chunks_count,
                }
            )
        chunks = merge_retrieve_chunks(query_results)
        full_documents = merge_full_documents(query_results)
        coverage = "strong" if len(chunks) >= 4 else ("weak" if chunks else "none")
        row = {
            "field": name,
            "queries": queries,
            "coverage": coverage,
            "retrieved_chunks": chunks,
            "full_documents": full_documents,
            "raw_results": query_results,
        }
        field_results.append(row)
        emit(
            {
                "type": "migration_advisor_retrieve",
                "field": name,
                "queries": queries,
                "status": "done",
                "coverage": coverage,
                "count": len(chunks),
                "full_document_count": len(full_documents),
            }
        )
    rex_result: dict[str, Any] | None = None
    img_paths: list[str] = []
    if isinstance(image_paths, list):
        for raw in image_paths:
            p = str(raw or "").strip()
            if p and Path(p).is_file() and p not in img_paths:
                img_paths.append(str(Path(p).resolve()))
    single = str(image_path or "").strip()
    if single and Path(single).is_file():
        sp = str(Path(single).resolve())
        if sp not in img_paths:
            img_paths.insert(0, sp)
    if img_paths and run_rex_annotation:
        detection_targets: dict[str, Any] = {}
        try:
            detection_targets = extract_detection_targets(
                user_query=user_query,
                plan=plan,
                api_key=api_key,
                base_url=base_url,
                debug_meta=meta,
            )
        except Exception as exc:
            rex_result = {
                "success": False,
                "label": "",
                "detection_targets": {},
                "num_boxes": 0,
                "pred_bboxes": [],
                "annotated_urls": [],
                "error": f"检测标签抽取失败: {exc}",
            }
            emit(
                {
                    "type": "migration_advisor_rex",
                    "status": "error",
                    "error": rex_result["error"],
                }
            )
            detection_targets = {}
        if detection_targets:
            label = str(detection_targets.get("display_label") or "").strip()
            emit(
                {
                    "type": "migration_advisor_rex",
                    "status": "running",
                    "label": label,
                    "detection_targets": detection_targets,
                }
            )
            try:
                rex_raw = run_rex_annotation(img_paths, detection_targets)
                rex_row = rex_raw if isinstance(rex_raw, dict) else {}
                if not rex_row.get("success"):
                    rex_result = {
                        "success": False,
                        "label": label,
                        "detection_targets": detection_targets,
                        "num_boxes": 0,
                        "pred_bboxes": [],
                        "annotated_urls": [],
                        "error": str(rex_row.get("error") or "Rex-Omni 检测失败"),
                    }
                    emit(
                        {
                            "type": "migration_advisor_rex",
                            "status": "error",
                            "label": label,
                            "error": rex_result["error"],
                        }
                    )
                else:
                    boxes = rex_row.get("pred_bboxes") if isinstance(rex_row.get("pred_bboxes"), list) else []
                    accuracy_estimate = estimate_rex_accuracy(
                        image_path=img_paths[0] if img_paths else "",
                        annotated_image_path=str(rex_row.get("annotated_image_path") or "").strip() or None,
                        annotated_image_paths=(
                            rex_row.get("annotated_image_paths")
                            if isinstance(rex_row.get("annotated_image_paths"), list)
                            else None
                        ),
                        user_query=user_query,
                        detection_label=str(rex_row.get("label") or label),
                        pred_bboxes=boxes,
                        pred_reports=(
                            rex_row.get("pred_reports")
                            if isinstance(rex_row.get("pred_reports"), list)
                            else None
                        ),
                        label_hit_summary=rex_row.get("label_hit_summary")
                        if isinstance(rex_row.get("label_hit_summary"), dict)
                        else {},
                        per_image=rex_row.get("per_image") if isinstance(rex_row.get("per_image"), list) else [],
                        api_key=api_key,
                        base_url=base_url,
                        debug_meta=meta,
                    )
                    rex_result = {
                        "success": True,
                        "label": str(rex_row.get("label") or label),
                        "task_mode": str(detection_targets.get("task_mode") or ""),
                        "detection_targets": detection_targets,
                        "num_boxes": int(rex_row.get("num_boxes") or len(boxes)),
                        "pred_bboxes": boxes,
                        "annotated_urls": rex_row.get("annotated_urls")
                        if isinstance(rex_row.get("annotated_urls"), list)
                        else [],
                        "annotated_image_path": str(rex_row.get("annotated_image_path") or ""),
                        "per_image": rex_row.get("per_image") if isinstance(rex_row.get("per_image"), list) else [],
                        "label_hit_summary": rex_row.get("label_hit_summary")
                        if isinstance(rex_row.get("label_hit_summary"), dict)
                        else {},
                        "accuracy_estimate": accuracy_estimate,
                    }
                    emit(
                        {
                            "type": "migration_advisor_rex",
                            "status": "done",
                            "label": rex_result["label"],
                            "task_mode": rex_result["task_mode"],
                            "detection_targets": detection_targets,
                            "num_boxes": rex_result["num_boxes"],
                            "annotated_urls": rex_result["annotated_urls"],
                            "per_image": rex_result["per_image"],
                            "accuracy_estimate": accuracy_estimate,
                        }
                    )
            except Exception as exc:
                rex_result = {
                    "success": False,
                    "label": label,
                    "detection_targets": detection_targets,
                    "num_boxes": 0,
                    "pred_bboxes": [],
                    "annotated_urls": [],
                    "error": str(exc),
                }
                emit(
                    {
                        "type": "migration_advisor_rex",
                        "status": "error",
                        "label": label,
                        "error": str(exc),
                    }
                )
    try:
        report = build_report(
            user_query=user_query,
            plan=plan,
            field_results=field_results,
            rex_result=rex_result,
            api_key=api_key,
            base_url=base_url,
            debug_meta=meta,
        )
        report = _apply_accuracy_comparison_to_report(report, rex_result)
    except Exception as exc:
        _write_migration_llm_debug(
            debug_meta=meta,
            stage="migration_report_response",
            step_index=2,
            payload=_debug_response_payload(
                model=MIGRATION_ADVISOR_MODEL,
                raw="",
                output_schema=REPORT_OUTPUT_SCHEMA,
                error=str(exc),
            ),
        )
        report = {
            "requirement_summary": str(user_query or "").strip(),
            "direct_match": {
                "exists": False,
                "summary": f"结构化报告生成失败：{exc}",
                "evidence": [],
            },
            "similar_assets": [],
            "migration_plan": {
                "feasibility": "low",
                "approach": "已完成分字段检索，但暂时无法生成完整结构化方案。",
                "data_requirements": "需人工复核检索证据。",
                "compute_requirements": "证据不足",
                "engineering_work": "证据不足",
                "estimated_timeline": "证据不足",
                "estimated_cost": "证据不足",
                "dependencies": [],
                "risks": ["结构化报告生成失败"],
            },
            "expected_performance": {
                "baseline": "证据不足",
                "target": "证据不足",
                "uncertainty": "高",
            },
            "recommendation": "建议先人工查看分字段检索结果，再补充样例图片、标注规范和目标指标。",
        }
    report_md = report_to_markdown(report)
    rex_md = rex_result_to_markdown(rex_result)
    markdown = "\n\n".join(part for part in (rex_md, report_md) if part).strip()
    output = {
        "plan": plan,
        "field_results": field_results,
        "rex_result": rex_result,
        "report": report,
        "markdown": markdown,
    }
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "migration_advisor_report.json").write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (run_dir / "migration_advisor_report.md").write_text(
            markdown,
            encoding="utf-8",
        )
    except OSError:
        pass
    emit({"type": "migration_advisor_report", "report": report, "markdown": markdown})
    return output
