from __future__ import annotations

import time
import json
from collections import Counter
from functools import lru_cache
import re
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from gbrain_rag.api.schemas import (
    EmbeddingObject,
    EmbeddingRequest,
    EmbeddingResponse,
    EvidenceItem,
    FullDocumentItem,
    LLMConfig,
    QueryRequest,
    QueryResponse,
    RetrieveRequest,
    RetrieveResponse,
    RoutePlanResponse,
)
from gbrain_rag.core.config import get_settings
from gbrain_rag.core.types import RoutePlan
from gbrain_rag.llm.client import (
    answer_structured_aggregate,
    answer_with_llm,
    evidence_important_values,
    expand_query_with_llm,
    plan_retrieval_sources,
    plan_structured_aggregate,
    score_knowledge_base_fully_answered,
    stream_answer_with_llm,
)
from gbrain_rag.retrieval.aspects import (
    ACCURACY_ASPECT,
    DEPLOYMENT_ASPECT,
    GENERAL_ASPECT,
    INPUT_OUTPUT_ASPECT,
    LABEL_ASPECT,
    LIMITATION_ASPECT,
    MODEL_ARTIFACT_ASPECT,
    OWNER_METADATA_ASPECT,
    PERFORMANCE_ASPECT,
    RELEASE_CHANGE_ASPECT,
)
from gbrain_rag.retrieval.embeddings import get_embedding_manager
from gbrain_rag.retrieval.query_understanding import (
    build_query_intent,
    canonical_value,
    parse_date,
    score_structured_row,
)
from gbrain_rag.retrieval.service import RetrievalService, evidence_payload

router = APIRouter(prefix="/rag", tags=["RAG"])


def _model_dump_compatible(model: Any) -> Any:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return model


def _sse_event(payload: Any) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _final_sse_payload(response: QueryResponse, **extra: Any) -> dict[str, Any]:
    payload = response.model_dump()
    payload.setdefault("content", "")
    payload.update(extra)
    return payload


def _sse_response(generator):
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _full_documents_for_evidences(request: RetrieveRequest, evidences: list[EvidenceItem]) -> list[FullDocumentItem]:
    if not request.include_full_documents or not evidences:
        return []
    doc_ids = [
        evidence.doc_id
        for evidence in evidences
        if evidence.doc_id and evidence.block_type != "aggregate"
    ]
    if not doc_ids:
        return []

    service = get_retrieval_service()
    chunks_by_doc_id = service.store.load_chunks_by_doc_ids(doc_ids)
    full_documents: list[FullDocumentItem] = []
    for doc_id in dict.fromkeys(doc_ids):
        chunks = chunks_by_doc_id.get(doc_id) or []
        if not chunks:
            continue
        first = chunks[0]
        content_parts: list[str] = []
        seen_parts: set[str] = set()
        for chunk in chunks:
            text = chunk.text.strip()
            if not text or text in seen_parts:
                continue
            seen_parts.add(text)
            page = f"page={chunk.page_label}" if chunk.page_label not in (None, "") else "page=-"
            title = chunk.title or chunk.block_type
            content_parts.append(f"[{page} block_type={chunk.block_type} title={title}]\n{text}")
        if not content_parts:
            continue
        full_documents.append(
            FullDocumentItem(
                doc_id=doc_id,
                doc_name=first.doc_name,
                source_type=first.source_type,
                source_path=first.source_path,
                content="\n\n".join(content_parts),
                chunk_count=len(chunks),
                metadata={
                    "matched_evidence_ids": [
                        evidence.evidence_id for evidence in evidences if evidence.doc_id == doc_id
                    ],
                },
            )
        )
    return full_documents


@lru_cache()
def get_retrieval_service() -> RetrievalService:
    return RetrievalService()


def _enabled_sources(request: RetrieveRequest) -> list[str]:
    if request.sources:
        return list(request.sources)
    sources = []
    if request.document.enabled:
        sources.append("document")
    if request.table.enabled:
        sources.append("table")
    if request.adela.enabled:
        sources.append("adela")
    return sources or ["document", "table", "adela"]


async def _resolve_query_expansion_terms(request: RetrieveRequest) -> tuple[list[str], dict[str, Any]]:
    configured_terms = [
        str(term).strip()
        for term in (request.query_expansion_terms or [])
        if str(term).strip()
    ]
    alias_terms = _domain_alias_terms(request.query)
    if not request.expand_query_with_llm:
        terms = list(dict.fromkeys([*configured_terms, *alias_terms]))
        method_parts = []
        if configured_terms:
            method_parts.append("request")
        if alias_terms:
            method_parts.append("alias")
        return terms, {
            "query_expansion_terms": terms,
            "query_expansion_method": "+".join(method_parts) if method_parts else "disabled",
            "query_expansion_ms": 0.0,
        }

    started = time.perf_counter()
    try:
        terms = await expand_query_with_llm(
            query=request.query,
            llm_config=getattr(request, "llm_config", None),
            max_terms=get_retrieval_service().settings.LLM_QUERY_EXPANSION_MAX_TERMS,
        )
        terms = list(dict.fromkeys([*configured_terms, *alias_terms, *terms]))
        method_parts = []
        if configured_terms:
            method_parts.append("request")
        if alias_terms:
            method_parts.append("alias")
        method_parts.append("llm")
        return terms, {
            "query_expansion_terms": terms,
            "query_expansion_method": "+".join(method_parts),
            "query_expansion_ms": round((time.perf_counter() - started) * 1000, 3),
        }
    except Exception as exc:
        terms = list(dict.fromkeys([*configured_terms, *alias_terms]))
        method_parts = []
        if configured_terms:
            method_parts.append("request")
        if alias_terms:
            method_parts.append("alias")
        method_parts.append("llm_error")
        return terms, {
            "query_expansion_terms": terms,
            "query_expansion_method": "+".join(method_parts),
            "query_expansion_error": str(exc),
            "query_expansion_ms": round((time.perf_counter() - started) * 1000, 3),
        }


def _domain_alias_terms(query: str) -> list[str]:
    """Deterministic domain aliases that should exist in RD model metadata."""

    normalized = str(query or "").lower()
    aliases: list[str] = []
    patterns = [
        (("安全绳",), ("safety_rope",)),
        (("安全带",), ("safetybelt", "safety_belt")),
        (("红马甲", "反光衣"), ("waistcoat", "PAR_waistcoat_safetybelt")),
        (("积水", "道路上是否有水", "道路有水"), ("water",)),
        (("人在水里", "不在水里", "水里"), ("swim", "in_water", "no_water")),
        (("图像级判断", "有无判断"), ("cls", "classify")),
        (("放烟花", "烟花", "烟火"), ("fire", "smoke")),
        (("烟雾", "只检测烟雾"), ("smoke",)),
        (("明火", "厨房明火"), ("fire",)),
        (("钓鱼",), ("fisher", "fishing")),
        (("睡岗",), ("sleep", "sleeper")),
        (("挖掘机", "挖机"), ("digger", "excavator")),
        (("裂缝",), ("crack",)),
        (("行为识别", "动作识别"), ("action", "behavior", "falcon")),
        (("结构化", "城市结构化"), ("struct",)),
        (("裁剪", "裁剪后的小图", "小图做结构化", "黑盒"), ("crop", "blackbox")),
        (("五类结构化",), ("full_struct", "struct")),
        (("文搜图", "图文检索"), ("shikra", "embedding")),
        (("大图小人脸", "监控大图", "小人脸"), ("smallface",)),
        (("大人脸",), ("largeface",)),
        (("人脸特征",), ("feature_face", "face-deepcode", "face-pro512d")),
        (("人脸识别",), ("ir7002024", "feature_ir7002024")),
        (("黄手套", "手套识别"), ("glove", "st_glove_v2")),
        (("厨房", "厨房场景"), ("kitchen",)),
        (("厨房工服", "工服识别"), ("kitchen", "uniform")),
        (("器具",), ("ware", "utensil")),
        (("更细的物品", "物品识别", "厨房物品"), ("item", "object")),
        (("横幅", "标语"), ("banner", "slogan")),
        (("车牌",), ("carplate", "plate")),
        (("文字识别",), ("textrecognition", "recognition")),
        (("文字检测",), ("textdetection", "detection")),
        (("城市结构化", "结构化检测"), ("struct",)),
    ]
    for triggers, terms in patterns:
        if any(trigger.lower() in normalized for trigger in triggers):
            aliases.extend(terms)
    return list(dict.fromkeys(aliases))


def _route_plan_response(plan) -> RoutePlanResponse:
    return RoutePlanResponse(
        document=plan.document,
        table=plan.table,
        adela=plan.adela,
        reason=plan.reason,
        sources=plan.sources,
    )


def _knowledge_base_fully_answered(answer: str, evidences: list[EvidenceItem]) -> float:
    answer_text = re.sub(r"\s+", "", str(answer or ""))
    if not answer_text or not evidences:
        return 0.0
    explicit_rejection_markers = (
        "抱歉,您提问的相关信息在知识库中没有找到",
        "抱歉，您提问的相关信息在知识库中没有找到",
        "证据不足",
        "当前未调用LLM",
        "LLM调用失败",
        "先给出可核对片段",
    )
    if any(re.sub(r"\s+", "", marker) in answer_text for marker in explicit_rejection_markers):
        return 0.0
    if any(evidence.block_type == "aggregate" for evidence in evidences):
        return 1.0
    roles = [str(evidence.payload.get("evidence_role") or "supporting") for evidence in evidences]
    return 0.7 if any(role != "caveat" for role in roles) else 0.2


async def _knowledge_base_fully_answered_confidence(
    query: str,
    answer: str,
    evidences: list[EvidenceItem],
    llm_config: LLMConfig | None,
) -> float:
    fallback = _knowledge_base_fully_answered(answer, evidences)
    try:
        llm_confidence = await score_knowledge_base_fully_answered(
            query=query,
            answer=answer,
            evidences=evidences,
            llm_config=llm_config,
        )
    except Exception:
        llm_confidence = None
    if llm_confidence is None:
        return fallback
    return round(max(0.0, min(1.0, float(llm_confidence))), 4)


def _answer_contains_token(answer: str, token: str) -> bool:
    normalized = str(answer or "").lower()
    compact_answer = re.sub(r"\s+", "", normalized)
    compact_token = re.sub(r"\s+", "", str(token or "").lower())
    return bool(compact_token and compact_token in compact_answer)


def _field_summary_records(field_summary: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for line in str(field_summary or "").splitlines():
        line = line.strip()
        if line.startswith("-"):
            line = line[1:].strip()
        if not line or "=" not in line:
            continue
        record: dict[str, str] = {}
        for part in line.split("|"):
            if "=" not in part:
                continue
            key, value = [item.strip() for item in part.split("=", 1)]
            if key and value:
                record[key] = value
        if record:
            records.append(record)
    return records


def _record_identity(record: dict[str, str]) -> str:
    for key in ("模型族", "模型名称", "算法名称", "目标名称"):
        value = record.get(key)
        if value:
            return value
    return ""


def _evidence_ref(index: int) -> str:
    return f"[证据{index}]"


def _complete_known_field_answer(query: str, answer: str, evidences: list[EvidenceItem]) -> str:
    normalized_query = str(query or "").lower()
    requested_labels = []
    for label, aliases in {
        "特征维度": ("特征维度", "维度"),
        "OID": ("oid",),
        "平台": ("平台",),
        "组件类型": ("组件类型", "组件"),
        "负责人": ("负责人", "owner"),
        "推荐配置": ("推荐配置", "recommended_config"),
        "支持设备": ("支持设备", "supported_device"),
        "最近更新时间": ("更新时间", "最近更新时间", "last_updated"),
        "did": ("did",),
        "rid": ("rid",),
    }.items():
        if any(alias.lower() in normalized_query for alias in aliases):
            requested_labels.append(label)
    if not requested_labels:
        return answer

    supplements: list[str] = []
    for evidence_idx, evidence in enumerate(evidences, start=1):
        for record in _field_summary_records(str(evidence.payload.get("field_summary") or "")):
            identity = _record_identity(record)
            for label in requested_labels:
                value = record.get(label)
                if not value or _answer_contains_token(answer, value):
                    continue
                if identity and not _answer_contains_token(answer, identity):
                    supplements.append(f"{identity} 的 {label} 为 {value}{_evidence_ref(evidence_idx)}")
                else:
                    supplements.append(f"{label} 为 {value}{_evidence_ref(evidence_idx)}")
            if len(supplements) >= 8:
                break
        if len(supplements) >= 8:
            break
    supplements = list(dict.fromkeys(supplements))
    if not supplements:
        return answer

    answer_core = str(answer or "").strip()
    if answer_core.endswith("。"):
        answer_core = answer_core[:-1]
    return f"{answer_core}；补充：{'，'.join(supplements)}。"


def _complete_important_values_answer(query: str, answer: str, evidences: list[EvidenceItem]) -> str:
    normalized_query = str(query or "").lower()
    if not any(
        token in normalized_query
        for token in (
            "优化",
            "提升",
            "精度",
            "指标",
            "相比",
            "追加",
            "标签",
            "release",
            "note",
            "版本",
        )
    ):
        return answer

    answer_text = str(answer or "")
    answer_compact = re.sub(r"\s+", "", answer_text.lower())
    missing: list[str] = []
    primary_evidences = [
        evidence
        for evidence in evidences
        if str(evidence.payload.get("evidence_role") or "") == "primary"
    ]
    candidate_evidences = primary_evidences or evidences
    for evidence in candidate_evidences[:8]:
        evidence_idx = evidences.index(evidence) + 1
        values = evidence_important_values(evidence)
        if not values:
            continue
        for value in values.split("；"):
            value = value.strip()
            if not value:
                continue
            value_tokens = [
                token
                for token in re.findall(r"[A-Za-z_][A-Za-z0-9_./+-]{2,}|\d+(?:\.\d+)?%?", value)
                if len(token) >= 2
            ]
            if not value_tokens:
                continue
            covered = any(re.sub(r"\s+", "", token.lower()) in answer_compact for token in value_tokens)
            if not covered:
                missing.append(f"{value}{_evidence_ref(evidence_idx)}")
            if len(missing) >= 6:
                break
        if len(missing) >= 6:
            break

    missing = list(dict.fromkeys(missing))
    if not missing:
        return answer

    answer_core = answer_text.strip()
    suffix = "；补充关键指标/原始标识：" + "；".join(missing) + "。"
    if answer_core.endswith("。"):
        return answer_core[:-1] + suffix
    return answer_core + suffix


def _evidence_row(evidence: EvidenceItem) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for container in (evidence.metadata, evidence.payload):
        if isinstance(container, dict):
            row.update(container)
            canonical = container.get("canonical_metadata")
            if isinstance(canonical, dict):
                row.update(canonical)
    return row


def _query_asks_model_choice(query: str) -> bool:
    compact = re.sub(r"\s+", "", str(query or "").lower())
    return any(
        token in compact
        for token in (
            "用哪个模型",
            "用哪个t4模型",
            "用哪个p4模型",
            "用哪个710模型",
            "哪个模型",
            "推荐哪个模型",
            "推荐用哪个",
            "现用模型",
            "模型是哪一个",
            "是哪一个",
            "可用方案",
            "推荐方案",
            "哪个方案",
            "什么方案",
        )
    ) or (
        ("用哪个" in compact and "模型" in compact)
    ) or (
        any(noun in compact for noun in ("模型", "方案"))
        and any(token in compact for token in ("现在", "目前", "推荐", "现用", "可用", "哪一个"))
    )


def _complete_model_record_answer(query: str, answer: str, evidences: list[EvidenceItem]) -> str:
    if not _query_asks_model_choice(query):
        return answer

    answer_text = str(answer or "")
    candidates: list[tuple[int, dict[str, Any]]] = []
    for evidence_idx, evidence in enumerate(evidences, start=1):
        if evidence.source_type != "table":
            continue
        row = _evidence_row(evidence)
        model_name = canonical_value(row, "model_name")
        if not model_name:
            continue
        candidates.append((evidence_idx, row))

    if any(_row_count_value(row) == "1" for _idx, row in candidates):
        candidates = [(idx, row) for idx, row in candidates if _row_count_value(row) == "1"]

    compact = re.sub(r"\s+", "", str(query or "").lower())
    limit = 2 if any(token in compact for token in ("哪几个", "几个模型", "哪些模型")) else 1
    supplements: list[str] = []
    for evidence_idx, row in candidates[:limit]:
        model_name = canonical_value(row, "model_name")
        oid = canonical_value(row, "oid")
        updated = canonical_value(row, "last_updated")
        count_value = _row_count_value(row)
        values = [value for value in (model_name, oid, updated, count_value) if value]
        if values and all(_answer_contains_token(answer_text, value) for value in values):
            continue
        supplements.append(_format_table_record_answer(row, evidence_idx))
        if len(supplements) >= limit:
            break

    supplements = list(dict.fromkeys(supplements))
    if not supplements:
        return answer
    answer_core = answer_text.strip()
    suffix = "；补充结构化发版记录：" + "；".join(supplements)
    if answer_core.endswith("。"):
        return answer_core[:-1] + suffix
    return answer_core + suffix


def _threshold_values_from_evidence(evidence: EvidenceItem) -> list[str]:
    text = "\n".join(
        part
        for part in (
            evidence.snippet,
            str(evidence.payload.get("index_text") or ""),
            str(evidence.payload.get("field_summary") or ""),
        )
        if part
    )
    if not re.search(r"阈值|threshold", text, re.I):
        return []
    values: list[str] = []
    number = r"(?<![\d.])(?:0\.\d+|1\.0+|\d+\.\d+)(?![\d.])"
    for match in re.finditer(rf"(?:torch|cuda[\w./+-]*|acl[\w./+-]*|t4|p4|710)\s+({number})", text, re.I):
        values.append(match.group(1))
    for pattern in (
        rf"(?:推荐)?阈值[^\d]{{0,80}}({number})",
        rf"threshold[^0-9]{{0,80}}({number})",
    ):
        for match in re.finditer(pattern, text, re.I):
            values.append(match.group(1))
    return list(dict.fromkeys(values))


def _complete_threshold_answer(query: str, answer: str, evidences: list[EvidenceItem]) -> str:
    normalized_query = str(query or "").lower()
    if "阈值" not in normalized_query and "threshold" not in normalized_query:
        return answer

    findings: list[str] = []
    for evidence_idx, evidence in enumerate(evidences, start=1):
        values = _threshold_values_from_evidence(evidence)
        if not values:
            continue
        for value in values:
            findings.append(f"证据{evidence_idx} 的阈值列/threshold 记录为 {value}")
            if len(findings) >= 3:
                break
        break

    findings = list(dict.fromkeys(findings))
    if not findings:
        return answer
    answer_core = str(answer or "").strip()
    uncertain = any(token in answer_core for token in ("未明确", "未找到", "未直接给出", "没有找到"))
    first_value_match = re.search(r"记录为\s*([\d.]+)", findings[0])
    first_value = first_value_match.group(0) if first_value_match else ""
    has_explicit_threshold = bool(
        first_value
        and re.search(rf"阈值(?:为|是|：|:)?\s*{re.escape(first_value)}", answer_core)
    )
    if not uncertain and has_explicit_threshold:
        return answer
    if first_value:
        suffix = f"；阈值结论：按{findings[0]}，阈值为 {first_value}。"
    else:
        suffix = "；补充阈值证据：" + "；".join(findings) + "。"
    if answer_core.endswith("。"):
        return answer_core[:-1] + suffix
    return answer_core + suffix


def _count_distinct(rows: list[dict[str, Any]], field: str) -> set[str]:
    values = set()
    for row in rows:
        value = str(row.get(field) or "").strip()
        if value:
            values.add(value)
    return values


def _canonical_distinct(rows: list[dict[str, Any]], field: str) -> set[str]:
    values = set()
    for row in rows:
        value = canonical_value(row, field).strip()
        if value:
            values.add(value)
    return values


def _row_text_for_match(row: dict[str, Any], fields: tuple[str, ...]) -> str:
    parts = []
    for field in fields:
        parts.append(canonical_value(row, field))
    for key in ("search_text", "label_list", "labels"):
        value = row.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value if item not in (None, ""))
        elif value not in (None, "", [], {}):
            parts.append(str(value))
    return "\n".join(part for part in parts if part).lower()


def _query_focus_terms(query: str, expansion_terms: list[str] | tuple[str, ...] | None = None) -> list[str]:
    intent = build_query_intent(query, extra_terms=expansion_terms)
    terms: list[str] = []
    terms.extend(intent.target_terms)
    terms.extend(intent.algorithm_terms)
    terms.extend(intent.model_terms)
    for mention in intent.query_frame.get("entity_mentions") or []:
        text = str(mention or "").strip()
        if text:
            terms.append(text)
            stripped = re.sub(r"(检测|识别|分类|属性|特征)?(模型|算法)?$", "", text).strip()
            if stripped and stripped != text:
                terms.append(stripped)
    for term in intent.semantic_terms:
        text = str(term or "").strip()
        if len(text) >= 2:
            terms.append(text)
    terms.extend(str(term).strip() for term in (expansion_terms or []) if str(term).strip())

    cleaned: list[str] = []
    seen: set[str] = set()
    stopwords = {
        "模型",
        "算法",
        "检测",
        "识别",
        "分类",
        "属性",
        "哪些",
        "有什么",
        "有什",
        "什么",
        "列表",
        "清单",
        "推荐",
        "相关",
    }
    for term in terms:
        value = re.sub(r"\s+", "", str(term or "").strip().lower())
        if not value or value in stopwords or len(value) < 2:
            continue
        if value in seen:
            continue
        seen.add(value)
        cleaned.append(value)
    return cleaned[:24]


def _table_primary_focus_terms(query: str, expansion_terms: list[str] | tuple[str, ...] | None = None) -> list[str]:
    """Terms that should bind tightly to table target/algorithm fields."""

    candidates: list[str] = []
    intent = build_query_intent(query, extra_terms=expansion_terms)
    candidates.extend(intent.target_terms)
    candidates.extend(intent.algorithm_terms)
    for mention in intent.query_frame.get("entity_mentions") or []:
        text = str(mention or "").strip()
        if not text:
            continue
        candidates.append(text)
        stripped = re.sub(r"(有什么|有哪些|什么|哪些|模型|算法|检测|识别|分类|属性|的|吗|么|？|\?)+$", "", text)
        if stripped and stripped != text:
            candidates.append(stripped)

    head = re.split(r"有什么|有哪些|哪些|什么|多少|几个|几款|推荐|用哪个|用什么", str(query or ""), maxsplit=1)[0]
    head = re.sub(r"(检测|识别|分类|属性|模型|算法|的)+$", "", head.strip(" 的？?，,。"))
    if head:
        candidates.append(head)

    cleaned: list[str] = []
    seen: set[str] = set()
    stopwords = {"检测", "识别", "分类", "模型", "算法", "有什么", "什么", "哪些", "有哪些"}
    for term in candidates:
        value = re.sub(r"\s+", "", str(term or "").strip().lower())
        value = re.sub(r"(检测|识别|分类|属性|模型|算法)+$", "", value)
        if len(value) < 2 or value in stopwords:
            continue
        if not re.search(r"[\u4e00-\u9fff]", value):
            continue
        if value in seen:
            continue
        seen.add(value)
        cleaned.append(value)
    return cleaned[:8]


def _structured_model_list_matches(
    query: str,
    rows: list[dict[str, Any]],
    *,
    source_type: str,
    expansion_terms: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    focus_terms = _query_focus_terms(query, expansion_terms)
    primary_terms = _table_primary_focus_terms(query, expansion_terms) if source_type == "table" else []
    if not focus_terms:
        return []
    fields = (
        "target_name",
        "algorithm_type",
        "algorithm_name",
        "application_scene",
        "model_name",
        "label_list",
    )
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for idx, row in enumerate(rows):
        if source_type == "table" and not canonical_value(row, "model_name"):
            continue
        if source_type == "adela" and not canonical_value(row, "model_name"):
            continue
        text = _row_text_for_match(row, fields)
        if not text:
            continue
        score = 0.0
        if source_type == "table" and primary_terms:
            primary_text = "\n".join(
                canonical_value(row, field)
                for field in ("target_name", "algorithm_name", "application_scene")
                if canonical_value(row, field)
            ).lower()
            primary_hits = [term for term in primary_terms if term in primary_text]
            if not primary_hits:
                continue
            score += 6.0 + len(primary_hits)
        for term in focus_terms:
            if term in text:
                if source_type == "table" and term in canonical_value(row, "target_name").lower():
                    score += 4.0
                elif source_type == "table" and term in canonical_value(row, "algorithm_name").lower():
                    score += 3.0
                elif source_type == "adela" and term in canonical_value(row, "model_name").lower():
                    score += 2.5
                elif source_type == "adela" and term in canonical_value(row, "label_list").lower():
                    score += 2.0
                else:
                    score += 0.8
        if source_type == "table" and "检测" in query and "检测" in (
            canonical_value(row, "algorithm_type") + canonical_value(row, "algorithm_name")
        ):
            score += 1.0
        if source_type == "adela" and any(term in text for term in ("fire", "smoke", "firesmog")):
            score += 0.4
        if score > 0:
            scored.append((score, idx, row))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [row for _score, _idx, row in scored]


def _try_structured_model_list(request: RetrieveRequest) -> tuple[RetrieveResponse, str] | None:
    intent = build_query_intent(request.query, extra_terms=request.query_expansion_terms)
    normalized = request.query.lower()
    compact = re.sub(r"\s+", "", normalized)
    if any(
        token in compact
        for token in (
            "阈值",
            "质量项",
            "判断哪些",
            "哪些质量",
            "哪些标签",
            "告警标签",
            "哪些文字类型",
            "输入输出",
            "提升多少",
        )
    ):
        return None
    if not intent.wants_list:
        return None
    if "模型" not in request.query and "model" not in normalized:
        return None
    enabled_sources = _enabled_sources(request)
    if not any(source in enabled_sources for source in ("table", "adela")):
        return None

    service = get_retrieval_service()
    evidences: list[EvidenceItem] = []
    answer_parts: list[str] = []
    source_counts: dict[str, int] = {}
    rank = 1

    if "table" in enabled_sources:
        table_rows = _structured_model_list_matches(
            request.query,
            service.store.load_metadata_rows("table"),
            source_type="table",
            expansion_terms=request.query_expansion_terms,
        )
        table_rows = _dedupe_rows(table_rows, ("model_name", "oid", "supported_device"))
        if table_rows:
            source_counts["table"] = len(table_rows)
            answer_parts.append(f"模型发版信息汇总表中匹配到 {len(table_rows)} 条模型记录：")
            for row in table_rows[:12]:
                evidence = _row_to_evidence(row, "table", rank, max(0.1, 1.0 - (rank - 1) * 0.03))
                evidences.append(evidence)
                answer_parts.append(
                    f"{len(evidences)}. {canonical_value(row, 'algorithm_name') or canonical_value(row, 'target_name')}："
                    f"{canonical_value(row, 'model_name')}，OID={canonical_value(row, 'oid') or '无'}，"
                    f"支持设备={canonical_value(row, 'supported_device') or '未标注'}，"
                    f"推荐配置={canonical_value(row, 'recommended_config') or '未标注'}。[证据{len(evidences)}]"
                )
                rank += 1

    if "adela" in enabled_sources and (not evidences or any(term in request.query for term in ("部署", "did", "rid", "平台"))):
        adela_rows = _structured_model_list_matches(
            request.query,
            service.store.load_metadata_rows("adela"),
            source_type="adela",
            expansion_terms=request.query_expansion_terms,
        )
        adela_rows = _dedupe_rows(adela_rows, ("model_name", "platform", "did"))
        if adela_rows:
            source_counts["adela"] = len(adela_rows)
            if answer_parts:
                answer_parts.append(f"Adela 部署记录另匹配到 {len(adela_rows)} 条：")
            else:
                answer_parts.append(f"Adela 部署记录中匹配到 {len(adela_rows)} 条模型记录：")
            for row in adela_rows[:8]:
                evidence = _row_to_evidence(row, "adela", rank, max(0.1, 1.0 - (rank - 1) * 0.03))
                evidences.append(evidence)
                answer_parts.append(
                    f"{len(evidences)}. {canonical_value(row, 'model_name')}，平台={canonical_value(row, 'platform') or '未标注'}，"
                    f"did={canonical_value(row, 'did') or '无'}，rid={canonical_value(row, 'rid') or '无'}，"
                    f"标签={canonical_value(row, 'label_list') or '未标注'}。[证据{len(evidences)}]"
                )
                rank += 1

    if not evidences:
        return None

    route_plan = RoutePlan(
        document=False,
        table="table" in source_counts,
        adela="adela" in source_counts,
        reason="deterministic structured model-list answer",
    )
    response = RetrieveResponse(
        query=request.query,
        route_plan=_route_plan_response(route_plan),
        evidences=evidences,
        full_documents=_full_documents_for_evidences(request, evidences),
        timings={"retrieve_ms": 0.0, "structured_model_list": source_counts},
        retrieved_count=len(evidences),
    )
    return response, "\n".join(answer_parts)


def _match_aggregate_condition(row: dict[str, Any], condition: dict[str, Any]) -> bool:
    field = str(condition.get("field") or "").strip()
    operator = str(condition.get("operator") or "contains").strip()
    values = [str(value).strip() for value in condition.get("values") or [] if str(value).strip()]
    if not field or not values:
        return True
    field_value = str(row.get(field) or "").strip()
    field_value_lower = field_value.lower()
    lowered_values = [value.lower() for value in values]
    if operator == "equals":
        return any(field_value_lower == value for value in lowered_values)
    if operator == "in":
        return field_value_lower in set(lowered_values)
    return any(value in field_value_lower for value in lowered_values)


def _row_matches_aggregate_conditions(
    row: dict[str, Any],
    field_conditions: list[dict[str, Any]],
    condition_logic: str,
) -> bool:
    if not field_conditions:
        return True
    matches = [_match_aggregate_condition(row, condition) for condition in field_conditions]
    return any(matches) if condition_logic == "any" else all(matches)


def _aggregate_source_profile(service: RetrievalService, enabled_sources: list[str]) -> str:
    profiles = []
    if "table" in enabled_sources:
        profiles.append(
            service.store.metadata_source_profile(
                "table",
                service.settings.TABLE_SEARCHABLE_FIELDS,
            )
        )
    if "adela" in enabled_sources:
        profiles.append(
            service.store.metadata_source_profile(
                "adela",
                service.settings.ADELA_SEARCHABLE_FIELDS,
            )
        )
    return "\n\n".join(profile for profile in profiles if profile)


def _retrieval_source_profile(enabled_sources: list[str]) -> str:
    profiles: list[str] = []
    if "document" in enabled_sources:
        profiles.append(
            "document: ONES/PDF 发版文档正文与表格，适合算法说明、精度指标、输入输出、限制条件、优化点、标签、模型文件等正文细节。"
        )
    if "table" in enabled_sources:
        profiles.append(
            "table: 模型发版信息汇总表，适合模型名称、OID、负责人、推荐配置、支持设备、更新时间、清单类结构化字段。"
        )
    if "adela" in enabled_sources:
        profiles.append(
            "adela: 部署记录，适合 did/rid、部署平台、部署状态、部署版本、部署清单等上线信息。"
        )
    return "\n".join(profiles)


async def _route_sources_with_llm(request: RetrieveRequest) -> RoutePlan:
    enabled_sources = _enabled_sources(request)
    llm_plan = await plan_retrieval_sources(
        query=request.query,
        enabled_sources=enabled_sources,
        source_profiles=_retrieval_source_profile(enabled_sources),
        llm_config=getattr(request, "llm_config", None),
    )
    selected = list(enabled_sources)
    reason = "Default all enabled sources"
    if llm_plan:
        llm_selected = [source for source in list(llm_plan.get("sources") or []) if source in enabled_sources]
        llm_confidence = str(llm_plan.get("confidence") or "medium").strip().lower()
        if llm_selected and llm_confidence == "high":
            selected = list(dict.fromkeys(llm_selected))
            reason = str(llm_plan.get("reason") or "LLM high-confidence source routing")
        else:
            fallback_reason = str(llm_plan.get("reason") or "").strip()
            if fallback_reason:
                reason = f"Default all enabled sources; LLM kept broad retrieval: {fallback_reason}"
            else:
                reason = "Default all enabled sources; LLM did not provide a high-confidence subset"
    return RoutePlan(
        document="document" in selected,
        table="table" in selected,
        adela="adela" in selected,
        reason=reason,
    )


def _is_document_value_lookup(query: str) -> bool:
    return any(token in query for token in ("特征维度", "维度", "测试精度", "精度", "追加", "标签", "优化", "输入", "输出")) and any(
        phrase in query for phrase in ("是多少", "是什么", "有哪些", "怎样", "什么")
    )


def _prefers_document_primary(query: str) -> bool:
    intent = build_query_intent(query)
    if any(token in query.lower() for token in ("来源", "为准", "不一致", "adela", "部署记录", "发版表")):
        return False
    return intent.aspect in {
        ACCURACY_ASPECT,
        LIMITATION_ASPECT,
        INPUT_OUTPUT_ASPECT,
        LABEL_ASPECT,
        PERFORMANCE_ASPECT,
        RELEASE_CHANGE_ASPECT,
    }


def _aspect_source_priority(query: str, source_type: str) -> int:
    intent = build_query_intent(query)
    if intent.aspect in {
        ACCURACY_ASPECT,
        LIMITATION_ASPECT,
        INPUT_OUTPUT_ASPECT,
        LABEL_ASPECT,
        PERFORMANCE_ASPECT,
        RELEASE_CHANGE_ASPECT,
    }:
        return {"document": 0, "table": 1, "adela": 2}.get(source_type, 3)
    if intent.aspect == DEPLOYMENT_ASPECT:
        return {"adela": 0, "table": 1, "document": 2}.get(source_type, 3)
    if intent.aspect in {MODEL_ARTIFACT_ASPECT, OWNER_METADATA_ASPECT}:
        return {"table": 0, "document": 1, "adela": 2}.get(source_type, 3)
    return {"document": 0, "table": 1, "adela": 2}.get(source_type, 3)


def _dedupe_rows(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(canonical_value(row, field).strip().lower() for field in fields)
        current = by_key.get(key)
        if current is None:
            by_key[key] = row
            continue
        if not current.get("status") and row.get("status"):
            by_key[key] = row
    return list(by_key.values())


def _row_count_value(row: dict[str, Any]) -> str:
    return str(row.get("count") or row.get("计数") or "").strip()


def _row_date_sort_value(row: dict[str, Any], *fields: str) -> str:
    for field in fields:
        value = canonical_value(row, field) or str(row.get(field) or "")
        value = value.strip()
        if not value:
            continue
        if re.fullmatch(r"\d{8}", value):
            return f"{value[:4]}-{value[4:6]}-{value[6:]}"
        parsed = parse_date(value)
        if parsed:
            return parsed.strftime("%Y-%m-%d")
        return value
    return ""


def _rank_structured_rows(
    query: str,
    rows: list[dict[str, Any]],
    *,
    source_type: str,
    expansion_terms: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    intent = build_query_intent(query, extra_terms=expansion_terms)
    scored: list[tuple[float, int, str, int, dict[str, Any]]] = []
    for idx, row in enumerate(rows):
        score = score_structured_row(row, intent, source_type)
        if score <= 0:
            continue
        current = 1 if _row_count_value(row) == "1" else 0
        if source_type == "adela":
            date_value = _row_date_sort_value(row, "version_train_date", "last_updated")
        else:
            date_value = _row_date_sort_value(row, "last_updated")
        scored.append((score, current, date_value, idx, row))
    scored.sort(key=lambda item: item[3])
    scored.sort(key=lambda item: item[2], reverse=True)
    scored.sort(key=lambda item: item[1], reverse=True)
    scored.sort(key=lambda item: item[0], reverse=True)
    return [row for _score, _current, _date, _idx, row in scored]


def _format_table_record_answer(row: dict[str, Any], evidence_idx: int = 1) -> str:
    algorithm = canonical_value(row, "algorithm_name") or canonical_value(row, "target_name") or "匹配模型"
    target = canonical_value(row, "target_name") or "未标注"
    algorithm_type = canonical_value(row, "algorithm_type") or "未标注"
    model_name = canonical_value(row, "model_name") or "未标注"
    device = canonical_value(row, "supported_device") or "未标注"
    recommended = canonical_value(row, "recommended_config") or "未标注"
    oid = canonical_value(row, "oid") or "未记录"
    updated = canonical_value(row, "last_updated") or "未标注"
    count_value = _row_count_value(row) or "未标注"
    row_id = str(row.get("row_id") or "").strip()
    row_text = f"（row_id={row_id}）" if row_id else ""
    return (
        f"按模型发版表{row_text}，{algorithm} 的目标是 {target}，算法类型为 {algorithm_type}；"
        f"推荐/现用记录是 {model_name}，支持设备 {device}，推荐配置 {recommended}，"
        f"OID 为 {oid}，最后更新日期 {updated}，count={count_value}。[证据{evidence_idx}]"
    )


def _format_adela_record_answer(row: dict[str, Any], evidence_idx: int = 1) -> str:
    model_name = canonical_value(row, "model_name") or "未标注"
    platform = canonical_value(row, "platform") or "未标注"
    rid = canonical_value(row, "rid") or "未记录"
    did = canonical_value(row, "did") or "未记录"
    status = str(row.get("status") or "未标注").strip() or "未标注"
    version = canonical_value(row, "version") or "未标注"
    train_date = str(row.get("version_train_date") or canonical_value(row, "last_updated") or "未标注").strip()
    labels = canonical_value(row, "label_list") or "未记录"
    return (
        f"Adela 部署记录显示 {model_name} 在平台 {platform} 上部署状态为 {status}，"
        f"rid={rid}，did={did}，版本 {version}，训练日期 {train_date}，标签为 {labels}。[证据{evidence_idx}]"
    )


def _is_deployment_record_query(query: str) -> bool:
    compact = re.sub(r"\s+", "", str(query or "").lower())
    if any(token in compact for token in ("did", "rid", "部署记录", "部署资料", "部署情况", "部署是什么情况", "部署里", "部署好的服务", "现成部署", "上线")):
        return True
    if "有部署" in compact or "t4部署" in compact:
        return True
    if "部署" in compact and "部署哪几个模型" not in compact and "需要部署哪几个模型" not in compact:
        return True
    return False


def _try_structured_deployment_lookup(request: RetrieveRequest) -> tuple[RetrieveResponse, str] | None:
    query = str(request.query or "")
    compact = re.sub(r"\s+", "", query.lower())
    if "adela" not in _enabled_sources(request):
        return None
    if any(token in compact for token in ("来源为准", "来源差异", "不一致", "到底以哪个来源")):
        return None
    if not _is_deployment_record_query(query):
        return None

    service = get_retrieval_service()
    rows = _rank_structured_rows(
        query,
        service.store.load_metadata_rows("adela"),
        source_type="adela",
        expansion_terms=request.query_expansion_terms,
    )
    rows = _dedupe_rows(rows, ("model_name", "platform", "did"))
    if not rows:
        return None

    limit = 2 if any(token in compact for token in ("不同精度", "不同版本", "哪些", "有什么")) else 1
    selected = rows[:limit]
    evidences = [
        _row_to_evidence(row, "adela", idx, max(0.1, 1.0 - (idx - 1) * 0.04))
        for idx, row in enumerate(selected, start=1)
    ]
    answer = "\n".join(_format_adela_record_answer(row, idx) for idx, row in enumerate(selected, start=1))
    response = RetrieveResponse(
        query=query,
        route_plan=_route_plan_response(
            RoutePlan(document=False, table=False, adela=True, reason="deterministic structured deployment lookup")
        ),
        evidences=evidences,
        full_documents=_full_documents_for_evidences(request, evidences),
        timings={"retrieve_ms": 0.0, "structured_deployment_lookup": {"matched": len(rows)}},
        retrieved_count=len(evidences),
    )
    return response, answer


def _is_straight_model_lookup(query: str) -> bool:
    compact = re.sub(r"\s+", "", str(query or "").lower())
    if _is_deployment_record_query(query):
        return False
    asks_model_choice = any(
        token in compact
        for token in (
            "用哪个模型",
            "哪个t4模型",
            "哪条t4模型",
            "推荐用哪个模型",
            "推荐哪个模型",
            "现用模型",
            "可用模型",
            "模型是哪一个",
            "模型是哪款",
            "是哪一个",
            "有没有适配",
            "有没有5合1",
            "哪几个模型",
            "几个模型",
            "用的是p4还是t4",
            "可用方案",
            "推荐方案",
            "哪个方案",
            "什么方案",
        )
    ) or (
        any(noun in compact for noun in ("模型", "方案"))
        and any(token in compact for token in ("现在", "目前", "推荐", "现用", "可用", "用哪个", "哪一个"))
    )
    if any(
        token in compact
        for token in (
            "阈值",
            "标签",
            "质量项",
            "哪些文字类型",
            "为什么",
            "提升多少",
            "怎么选",
            "取舍",
            "边界",
            "限制",
            "输入输出",
            "告警标签",
        )
    ):
        return False
    if "精度" in compact and not (asks_model_choice and any(token in compact for token in ("高精度", "更高精度", "准确率"))):
        return False
    return asks_model_choice


def _try_structured_model_lookup(request: RetrieveRequest) -> tuple[RetrieveResponse, str] | None:
    query = str(request.query or "")
    if "table" not in _enabled_sources(request):
        return None
    if not _is_straight_model_lookup(query):
        return None

    service = get_retrieval_service()
    rows = _rank_structured_rows(
        query,
        service.store.load_metadata_rows("table"),
        source_type="table",
        expansion_terms=request.query_expansion_terms,
    )
    rows = _dedupe_rows(rows, ("model_name", "oid", "supported_device"))
    if not rows:
        return None

    compact = re.sub(r"\s+", "", query.lower())
    limit = 3 if any(token in compact for token in ("哪几个", "几个模型", "哪些模型", "需要部署哪几个")) else 1
    selected = rows[:limit]
    evidences = [
        _row_to_evidence(row, "table", idx, max(0.1, 1.0 - (idx - 1) * 0.04))
        for idx, row in enumerate(selected, start=1)
    ]
    answer = "\n".join(_format_table_record_answer(row, idx) for idx, row in enumerate(selected, start=1))
    response = RetrieveResponse(
        query=query,
        route_plan=_route_plan_response(
            RoutePlan(document=False, table=True, adela=False, reason="deterministic structured model lookup")
        ),
        evidences=evidences,
        full_documents=_full_documents_for_evidences(request, evidences),
        timings={"retrieve_ms": 0.0, "structured_model_lookup": {"matched": len(rows)}},
        retrieved_count=len(evidences),
    )
    return response, answer


def _source_rank_fused_results(all_results: list[Any], source_order: list[str], top_k: int) -> list[Any]:
    """Rank cross-source evidence by source-local rank instead of raw scores."""
    if len(source_order) <= 1 or top_k <= 0:
        return all_results

    source_order = list(dict.fromkeys(source for source in source_order if source))
    source_priority = {source: idx for idx, source in enumerate(source_order)}
    by_source: dict[str, list[Any]] = {source: [] for source in source_order}
    leftovers: list[Any] = []
    for result in all_results:
        source_type = getattr(getattr(result, "chunk", None), "source_type", "")
        if source_type in by_source:
            by_source[source_type].append(result)
        else:
            leftovers.append(result)

    ranked: list[tuple[float, int, int, float, int, Any]] = []
    seen_ids: set[int] = set()
    for source in source_order:
        for rank, result in enumerate(by_source[source], start=1):
            result_id = id(result)
            if result_id in seen_ids:
                continue
            seen_ids.add(result_id)
            score = float(getattr(result, "score", 0.0) or 0.0)
            # Use source-local rank as the cross-source comparable signal.
            # The raw score is only a tie-breaker within similar rank bands.
            rank_score = 1.0 / float(60 + rank)
            ranked.append(
                (
                    -rank_score,
                    rank,
                    source_priority.get(source, len(source_order)),
                    -score,
                    result_id,
                    result,
                )
            )

    for result in leftovers:
        result_id = id(result)
        if result_id in seen_ids:
            continue
        seen_ids.add(result_id)
        score = float(getattr(result, "score", 0.0) or 0.0)
        ranked.append((0.0, top_k + 1, len(source_order), -score, result_id, result))

    return [item[-1] for item in sorted(ranked)[:top_k]]


_source_balanced_results = _source_rank_fused_results


def _source_rank_fused_results_for_query(
    query: str,
    all_results: list[Any],
    source_order: list[str],
    top_k: int,
) -> list[Any]:
    if not all_results or top_k <= 0:
        return []
    if not _prefers_document_primary(query):
        return _source_rank_fused_results(all_results, source_order, top_k)

    ranked = []
    for idx, result in enumerate(all_results):
        chunk = getattr(result, "chunk", None)
        source_type = getattr(chunk, "source_type", "")
        signals = getattr(result, "retrieval_signals", {}) or {}
        answerability = float(signals.get("answerability") or 0.0)
        score = float(getattr(result, "score", 0.0) or 0.0)
        ranked.append(
            (
                _aspect_source_priority(query, source_type),
                -answerability,
                -score,
                idx,
                result,
            )
        )
    return [item[-1] for item in sorted(ranked)[:top_k]]


def _merge_support_evidences(
    *,
    request: RetrieveRequest,
    response: RetrieveResponse,
    primary_sources: set[str],
    top_k: int = 2,
) -> RetrieveResponse:
    enabled_sources = _enabled_sources(request)
    missing_sources = [
        source
        for source in enabled_sources
        if source not in primary_sources
        and source not in {evidence.source_type for evidence in response.evidences}
    ]
    if not request.sources or not missing_sources:
        return response

    service = get_retrieval_service()
    started = time.perf_counter()
    support_evidences: list[EvidenceItem] = []
    support_timings: dict[str, dict[str, float]] = {}
    for source_type in missing_sources:
        source_cfg = getattr(request, source_type)
        results, timing = service.retrieve(
            query=request.query,
            source_types=[source_type],
            retrieval_method=source_cfg.retrieval_method or request.retrieval_method,
            top_k=min(source_cfg.top_k or request.top_k, top_k),
            candidate_limit=request.candidate_limit,
            query_expansion_terms=request.query_expansion_terms,
            embedding_model=request.embedding_model,
            embedding_models=request.embedding_models
            or (
                list(service.settings.EMBEDDING_MODELS)
                if source_type == "document"
                else [service.settings.EMBEDDING_MODEL]
            ),
            embedding_backend=request.embedding_backend,
            similarity_threshold=(
                source_cfg.similarity_threshold
                if source_cfg.similarity_threshold is not None
                else request.similarity_threshold
            ),
        )
        support_timings[source_type] = timing
        support_evidences.extend(EvidenceItem(**evidence_payload(result, request.query)) for result in results)

    if not support_evidences:
        return response

    seen = {evidence.evidence_id for evidence in response.evidences}
    merged = list(response.evidences)
    for evidence in support_evidences:
        if evidence.evidence_id in seen:
            continue
        merged.append(evidence)
        seen.add(evidence.evidence_id)

    route_plan = RoutePlan(
        document="document" in enabled_sources,
        table="table" in enabled_sources,
        adela="adela" in enabled_sources,
        reason=f"{response.route_plan.reason}; support evidence added for explicit request.sources",
    )
    timings = dict(response.timings)
    timings["support_sources"] = support_timings
    timings["support_retrieve_ms"] = round((time.perf_counter() - started) * 1000, 3)
    timings["retrieve_ms"] = round(float(timings.get("retrieve_ms") or 0.0) + timings["support_retrieve_ms"], 3)
    return RetrieveResponse(
        query=response.query,
        route_plan=_route_plan_response(route_plan),
        evidences=merged,
        full_documents=_full_documents_for_evidences(request, merged),
        timings=timings,
        retrieved_count=len(merged),
    )


def _row_to_evidence(row: dict[str, Any], source_type: str, rank: int, score: float) -> EvidenceItem:
    row_chunk_id = str(row.get("chunk_id") or "").strip()
    raw_source_path = str(row.get("source_path") or "").strip()
    source_file = str(row.get("source_file") or "").strip()
    if not raw_source_path and source_file.startswith("/"):
        raw_source_path = source_file
    if not raw_source_path and source_file.startswith("data_source/"):
        raw_source_path = str((get_settings().DATA_SOURCE_DIR.parent / source_file).resolve())
    reference_url = str(row.get("reference") or row.get("ones_release_link") or "").strip() or None
    if source_type == "adela":
        title = canonical_value(row, "model_name") or canonical_value(row, "did") or f"adela row {rank}"
        row_id = str(row.get("row_id") or "").strip()
        if not row_id:
            model_name = canonical_value(row, "model_name")
            platform = canonical_value(row, "platform")
            rid = canonical_value(row, "rid")
            did = canonical_value(row, "did")
            if model_name and platform and rid and did:
                row_id = f"{model_name}-{platform}_{rid}_{did}"
        snippet_parts = [
            f"row_id: {row_id}",
            f"model_name: {canonical_value(row, 'model_name')}",
            f"platform: {canonical_value(row, 'platform')}",
            f"status: {row.get('status') or ''}",
            f"did: {canonical_value(row, 'did')}",
            f"rid: {canonical_value(row, 'rid')}",
        ]
        legacy_id = f"adela::{row_id}" if row_id else None
        evidence_id = row_chunk_id or f"structured::adela::{row_id or canonical_value(row, 'did') or rank}"
        doc_name = source_file or str(row.get("source_filename") or "").strip() or "adela/adela_release_records.jsonl"
        doc_id = row_chunk_id or f"structured_adela::{row_id or canonical_value(row, 'did') or rank}"
    else:
        title = canonical_value(row, "algorithm_name") or canonical_value(row, "model_name") or f"table row {rank}"
        row_id = str(row.get("row_id") or "").strip()
        if not row_id:
            sheet_name = str(row.get("sheet_name") or "").strip()
            source_row = str(row.get("source_row_number") or "").strip()
            if sheet_name and source_row:
                try:
                    row_id = f"{sheet_name}-{int(float(source_row)):04d}"
                except ValueError:
                    row_id = f"{sheet_name}-{source_row}"
        snippet_parts = [
            f"row_id: {row_id}",
            f"目标名称: {canonical_value(row, 'target_name')}",
            f"算法类型: {canonical_value(row, 'algorithm_type')}",
            f"算法名称: {canonical_value(row, 'algorithm_name')}",
            f"模型名称: {canonical_value(row, 'model_name')}",
            f"OID: {canonical_value(row, 'oid')}",
            f"支持设备: {canonical_value(row, 'supported_device')}",
            f"推荐配置: {canonical_value(row, 'recommended_config')}",
        ]
        legacy_id = f"table::{row_id}" if row_id else None
        evidence_id = row_chunk_id or f"structured::table::{row_id or canonical_value(row, 'oid') or rank}"
        source_label = source_file or "data_source/tables/model_release_records.jsonl"
        sheet_name = str(row.get("sheet_name") or "").strip()
        source_row = str(row.get("source_row_number") or "").strip()
        row_suffix = f"#{sheet_name}!row={source_row}" if sheet_name and source_row else ""
        doc_name = f"{source_label}{row_suffix}"
        doc_id = row_chunk_id or f"structured_table::{row_id or canonical_value(row, 'oid') or rank}"
    payload = dict(row)
    if row_id:
        payload["row_id"] = row_id
    if legacy_id:
        payload["legacy_evidence_id"] = legacy_id
    payload["doc_name"] = doc_name
    payload["source_path"] = raw_source_path or None
    payload["reference_url"] = reference_url
    for field in (
        "target_name",
        "algorithm_type",
        "algorithm_name",
        "model_name",
        "oid",
        "supported_device",
        "recommended_config",
        "platform",
        "did",
        "rid",
        "label_list",
    ):
        value = canonical_value(row, field)
        if value:
            payload.setdefault(field, value)
    payload["canonical_metadata"] = {
        key: value
        for key, value in {
            "row_id": row_id,
            "target_name": canonical_value(row, "target_name"),
            "algorithm_type": canonical_value(row, "algorithm_type"),
            "algorithm_name": canonical_value(row, "algorithm_name"),
            "model_name": canonical_value(row, "model_name"),
            "oid": canonical_value(row, "oid"),
            "supported_device": canonical_value(row, "supported_device"),
            "recommended_config": canonical_value(row, "recommended_config"),
            "did": canonical_value(row, "did"),
            "rid": canonical_value(row, "rid"),
            "platform": canonical_value(row, "platform"),
            "status": str(row.get("status") or ""),
            "label_list": canonical_value(row, "label_list"),
        }.items()
        if value
    }
    metadata = dict(payload)
    snippet = "\n".join(part for part in snippet_parts if not part.endswith(": "))
    return EvidenceItem(
        evidence_id=evidence_id,
        legacy_evidence_id=legacy_id,
        source_type=source_type,
        score=round(float(score), 6),
        source_rank=rank,
        source_score=round(float(score), 6),
        title=title,
        snippet=snippet,
        doc_id=doc_id,
        doc_name=doc_name,
        block_type="structured_row",
        source_path=raw_source_path or None,
        reference_url=reference_url,
        metadata=metadata,
        payload=payload,
    )


def _structured_answer_from_rows(query: str, source_type: str, rows: list[dict[str, Any]]) -> str:
    intent = build_query_intent(query)
    if not rows:
        return "抱歉,您提问的相关信息在知识库中没有找到"

    normalized = query.lower()
    if "总共" in query and "发布" in query and "部署" in query:
        service = get_retrieval_service()
        table_rows = _dedupe_rows(service.store.load_metadata_rows("table"), ("model_name", "oid", "supported_device"))
        adela_rows = _dedupe_rows(service.store.load_metadata_rows("adela"), ("model_name", "platform", "did"))
        table_models = _canonical_distinct(table_rows, "model_name")
        adela_models = _canonical_distinct(adela_rows, "model_name")
        return (
            f"按当前知识库统计：模型发版汇总表中共有 {len(table_models)} 个发布模型"
            f"（去重后，原始记录 {len(table_rows)} 条）；Adela 中共有 {len(adela_models)} 个已部署模型"
            f"（去重后部署记录 {len(adela_rows)} 条）。"
        )

    if intent.wants_count:
        records = len(rows)
        if source_type == "adela":
            models = _canonical_distinct(rows, "model_name")
            return f"按 Adela 部署记录统计，匹配到 {records} 条部署记录，按 model_name 去重为 {len(models)} 个模型。"
        models = _canonical_distinct(rows, "model_name")
        algorithms = _canonical_distinct(rows, "algorithm_name")
        if intent.wants_list:
            names = "、".join(sorted(algorithms)[:30])
            summary = f"匹配到 {records} 条模型记录、{len(algorithms)} 个算法、{len(models)} 个模型。算法包括：{names}。"
        else:
            summary = f"按模型发版信息汇总表统计，匹配到 {records} 条模型记录，按 model_name 去重为 {len(models)} 个模型；涉及算法 {len(algorithms)} 个。"
        return summary

    if intent.wants_latest_check:
        dates = [parse_date(canonical_value(row, "last_updated")) for row in rows]
        dates = [date for date in dates if date]
        latest = max(dates).strftime("%Y-%m-%d") if dates else "未知"
        return f"知识库中匹配到 {len(rows)} 条相关记录，最新更新时间为 {latest}。请结合证据中的模型名称和版本判断是否满足当前项目口径。"

    lines: list[str] = []
    if intent.wants_deployment and source_type == "adela":
        lines.append(f"有，匹配到 {len(rows)} 条部署记录：")
        for idx, row in enumerate(rows[:8], start=1):
            lines.append(
                f"{idx}. {canonical_value(row, 'model_name')}，平台 {canonical_value(row, 'platform')}，"
                f"did={canonical_value(row, 'did')}，rid={canonical_value(row, 'rid')}，状态 {row.get('status') or '未知'}。"
            )
        return "\n".join(lines)

    if intent.wants_oid:
        lines.append(f"匹配到 {len(rows)} 条模型记录：")
        for idx, row in enumerate(rows[:12], start=1):
            lines.append(
                f"{idx}. {canonical_value(row, 'algorithm_name') or canonical_value(row, 'target_name')}："
                f"{canonical_value(row, 'model_name')}，OID={canonical_value(row, 'oid') or '无'}，"
                f"支持设备={canonical_value(row, 'supported_device') or '未标注'}。"
            )
        return "\n".join(lines)

    if intent.wants_recommendation or intent.wants_list:
        lines.append(f"推荐优先查看以下 {min(len(rows), 8)} 条模型记录：")
        for idx, row in enumerate(rows[:8], start=1):
            if source_type == "adela":
                lines.append(
                    f"{idx}. {canonical_value(row, 'model_name')}，平台 {canonical_value(row, 'platform')}，"
                    f"did={canonical_value(row, 'did')}，rid={canonical_value(row, 'rid')}。"
                )
            else:
                lines.append(
                    f"{idx}. {canonical_value(row, 'algorithm_name') or canonical_value(row, 'target_name')}："
                    f"{canonical_value(row, 'model_name')}，OID={canonical_value(row, 'oid') or '无'}，"
                    f"支持设备={canonical_value(row, 'supported_device') or '未标注'}。"
                )
        return "\n".join(lines)

    row = rows[0]
    return (
        f"匹配到：{canonical_value(row, 'algorithm_name') or canonical_value(row, 'model_name')}，"
        f"模型 {canonical_value(row, 'model_name')}，OID={canonical_value(row, 'oid') or '无'}。"
    )


def _build_aggregate_response(
    *,
    query: str,
    plan: dict[str, Any],
    rows: list[dict[str, Any]],
    answer: str,
) -> RetrieveResponse:
    source_type = str(plan.get("source_type") or "table")
    operation = str(plan.get("operation") or "count_unique")
    count_field = (
        str(plan.get("dedupe_field") or "").strip()
        or str(plan.get("count_field") or "").strip()
        or ("row_id" if operation == "count_records" else "model_name")
    )
    field_conditions = [
        condition
        for condition in list(plan.get("field_conditions") or [])
        if isinstance(condition, dict)
    ]
    condition_logic = str(plan.get("condition_logic") or "all")
    if condition_logic not in {"all", "any"}:
        condition_logic = "all"
    filtered_rows = [
        row
        for row in rows
        if _row_matches_aggregate_conditions(row, field_conditions, condition_logic)
    ]
    if operation == "count_records":
        result_count = len(filtered_rows)
        unique_values: set[str] = set()
    else:
        unique_values = _count_distinct(filtered_rows, count_field)
        result_count = len(unique_values)
    source_label = "Adela 部署记录" if source_type == "adela" else "模型发版信息汇总表"
    sample_values = sorted(unique_values)[:20] if unique_values else []
    result_payload = {
        "aggregate_type": operation,
        "source_type": source_type,
        "source_label": source_label,
        "count_field": count_field,
        "count": result_count,
        "record_count": len(rows),
        "filtered_record_count": len(filtered_rows),
        "condition_logic": condition_logic,
        "field_conditions": field_conditions,
        "planner_reason": plan.get("reason"),
        "confidence": plan.get("confidence"),
        "ambiguity": plan.get("ambiguity"),
        "count_semantics": plan.get("count_semantics"),
        "sample_values": sample_values,
    }
    metric_label = "记录数" if operation == "count_records" else f"按 `{count_field}` 去重后的数量"
    sample_text = "；样例：" + "、".join(sample_values[:10]) if sample_values else ""
    evidence = EvidenceItem(
        evidence_id=f"aggregate::{source_type}::{operation}",
        source_type=source_type,
        score=1.0,
        source_rank=1,
        source_score=1.0,
        title=f"{source_label}智能统计",
        snippet=(
            f"{metric_label}: {result_count}; "
            f"filtered_record_count: {len(filtered_rows)}; "
            f"record_count: {len(rows)}{sample_text}"
        ),
        doc_id=f"{source_type}_aggregate",
        doc_name=f"{source_type}_aggregate",
        block_type="aggregate",
        metadata=result_payload,
        payload=result_payload,
    )
    route_plan = RoutePlan(
        document=False,
        table=source_type == "table",
        adela=source_type == "adela",
        reason=str(plan.get("reason") or "LLM structured aggregate plan"),
    )
    response = RetrieveResponse(
        query=query,
        route_plan=_route_plan_response(route_plan),
        evidences=[evidence],
        timings={"retrieve_ms": 0.0, "aggregate": result_payload, "aggregate_plan": plan},
        retrieved_count=1,
    )
    return response


def _aggregate_evidence(
    *,
    query: str,
    source_type: str,
    title: str,
    snippet: str,
    payload: dict[str, Any],
    route_sources: set[str] | None = None,
) -> RetrieveResponse:
    evidence = EvidenceItem(
        evidence_id=f"aggregate::{source_type}::{payload.get('aggregate_type') or 'summary'}",
        source_type=source_type,
        score=1.0,
        source_rank=1,
        source_score=1.0,
        title=title,
        snippet=snippet,
        doc_id=f"{source_type}_aggregate",
        doc_name=f"{source_type}_aggregate",
        block_type="aggregate",
        metadata=payload,
        payload=payload,
    )
    route_sources = route_sources or {source_type}
    route_plan = RoutePlan(
        document="document" in route_sources,
        table="table" in route_sources,
        adela="adela" in route_sources,
        reason="deterministic structured aggregate",
    )
    return RetrieveResponse(
        query=query,
        route_plan=_route_plan_response(route_plan),
        evidences=[evidence],
        timings={"retrieve_ms": 0.0, "aggregate": payload},
        retrieved_count=1,
    )


def _hardware_counter(rows: list[dict[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        device = canonical_value(row, "supported_device")
        if not device:
            continue
        parts = re.split(r"[,，/、]+", device)
        for part in parts:
            value = part.strip()
            if value:
                counter[value] += 1
    return counter


def _try_deterministic_aggregate(request: RetrieveRequest) -> tuple[RetrieveResponse, str] | None:
    query = str(request.query or "")
    compact = re.sub(r"\s+", "", query.lower())
    enabled_sources = set(_enabled_sources(request))
    service = get_retrieval_service()

    if "adela" in enabled_sources and (
        ("部署资料" in compact and "状态" in compact)
        or ("线上部署规模" in compact)
        or ("部署规模" in compact and "状态" in compact)
    ):
        rows = service.store.load_metadata_rows("adela")
        status_counts = Counter(str(row.get("status") or "未标注").strip() or "未标注" for row in rows)
        model_names = _canonical_distinct(rows, "model_name")
        status_text = "、".join(f"{status} {count}" for status, count in status_counts.most_common())
        answer = (
            f"按 adela_release_records.jsonl 统计，共 {len(rows)} 条部署记录，"
            f"status 分布为 {status_text}；按 model_name 去重为 {len(model_names)} 个模型名。"
        )
        payload = {
            "aggregate_type": "deployment_status_summary",
            "source_type": "adela",
            "record_count": len(rows),
            "model_name_unique_count": len(model_names),
            "status_counts": dict(status_counts),
            "deterministic_answer": answer,
            "sample_fields": ["did", "rid", "model_name", "status"],
        }
        snippet = (
            f"adela_release_records.jsonl deployment records: {len(rows)}; "
            f"unique model_name: {len(model_names)}; status: {status_text}; "
            "fields: did rid model_name status SUCCESS"
        )
        return _aggregate_evidence(
            query=query,
            source_type="adela",
            title="Adela 部署规模与状态统计",
            snippet=snippet,
            payload=payload,
        ), answer

    if "table" in enabled_sources and any(token in compact for token in ("适配哪些硬件", "主要适配", "硬件")):
        rows = [
            row
            for row in service.store.load_metadata_rows("table")
            if str(row.get("count") or "").strip() == "1"
        ]
        device_counts = _hardware_counter(rows)
        if device_counts:
            ordered = device_counts.most_common()
            counts_text = "、".join(f"{device} {count} 条" for device, count in ordered)
            answer = f"按模型发版信息汇总表中 count=1 的当前记录统计，当前模型主要适配：{counts_text}。"
            payload = {
                "aggregate_type": "current_hardware_distribution",
                "source_type": "table",
                "record_count": len(rows),
                "count_filter": "count=1",
                "hardware_counts": dict(ordered),
                "deterministic_answer": answer,
            }
            snippet = f"count=1 hardware distribution: {counts_text}"
            return _aggregate_evidence(
                query=query,
                source_type="table",
                title="当前模型硬件适配统计",
                snippet=snippet,
                payload=payload,
            ), answer

    return None


def _fallback_aggregate_answer(plan: dict[str, Any], result: dict[str, Any]) -> str:
    source_label = str(result.get("source_label") or result.get("source_type") or "结构化数据源")
    count = int(result.get("count") or 0)
    operation = str(result.get("aggregate_type") or "")
    count_field = str(result.get("count_field") or "")
    if operation == "count_records":
        main = f"当前{source_label}中共有 {count} 条记录。"
    else:
        main = f"当前{source_label}中按 `{count_field}` 去重后共有 {count} 个。"
    main += (
        f"统计口径：{result.get('count_semantics') or operation}；"
        f"参与统计记录数 {result.get('filtered_record_count')} / 原始记录数 {result.get('record_count')}。"
    )
    if result.get("ambiguity"):
        main += f"口径提示：{result['ambiguity']}。"
    return main


async def _try_structured_aggregate(request: RetrieveRequest) -> tuple[RetrieveResponse, str | None] | None:
    service = get_retrieval_service()
    enabled_sources = [source for source in _enabled_sources(request) if source in {"table", "adela"}]
    if not enabled_sources:
        return None

    try:
        plan = await plan_structured_aggregate(
            query=request.query,
            enabled_sources=enabled_sources,
            source_profiles=_aggregate_source_profile(service, enabled_sources),
            llm_config=getattr(request, "llm_config", None),
        )
    except Exception:
        plan = None
    if not plan or not plan.get("should_aggregate"):
        return None

    source_type = str(plan.get("source_type") or "")
    if source_type not in enabled_sources:
        return None
    rows = service.store.load_metadata_rows(source_type)
    placeholder = _build_aggregate_response(query=request.query, plan=plan, rows=rows, answer="")
    result_payload = placeholder.evidences[0].payload
    try:
        answer = await answer_structured_aggregate(
            query=request.query,
            plan=plan,
            result=result_payload,
            llm_config=getattr(request, "llm_config", None),
        )
    except Exception:
        answer = None
    answer = answer or _fallback_aggregate_answer(plan, result_payload)
    response = _build_aggregate_response(
        query=request.query,
        plan=plan,
        rows=rows,
        answer=answer,
    )
    response = _merge_support_evidences(
        request=request,
        response=response,
        primary_sources={source_type},
    )
    return response, answer


async def _retrieve(request: RetrieveRequest) -> RetrieveResponse:
    service = get_retrieval_service()
    started = time.perf_counter()
    expansion_terms, expansion_timing = await _resolve_query_expansion_terms(request)
    request.query_expansion_terms = expansion_terms
    deterministic_aggregate = _try_deterministic_aggregate(request)
    if deterministic_aggregate is not None:
        response = deterministic_aggregate[0]
        timings = dict(response.timings)
        timings.update(expansion_timing)
        timings["retrieve_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return RetrieveResponse(
            query=response.query,
            route_plan=response.route_plan,
            evidences=response.evidences,
            full_documents=_full_documents_for_evidences(request, response.evidences),
            timings=timings,
            retrieved_count=response.retrieved_count,
        )
    deployment_lookup = _try_structured_deployment_lookup(request)
    if deployment_lookup is not None:
        response = deployment_lookup[0]
        timings = dict(response.timings)
        timings.update(expansion_timing)
        timings["retrieve_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return RetrieveResponse(
            query=response.query,
            route_plan=response.route_plan,
            evidences=response.evidences,
            full_documents=response.full_documents,
            timings=timings,
            retrieved_count=response.retrieved_count,
        )
    model_lookup = _try_structured_model_lookup(request)
    if model_lookup is not None:
        response = model_lookup[0]
        timings = dict(response.timings)
        timings.update(expansion_timing)
        timings["retrieve_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return RetrieveResponse(
            query=response.query,
            route_plan=response.route_plan,
            evidences=response.evidences,
            full_documents=response.full_documents,
            timings=timings,
            retrieved_count=response.retrieved_count,
        )
    model_list = _try_structured_model_list(request)
    if model_list is not None:
        response = model_list[0]
        timings = dict(response.timings)
        timings.update(expansion_timing)
        timings["retrieve_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return RetrieveResponse(
            query=response.query,
            route_plan=response.route_plan,
            evidences=response.evidences,
            full_documents=response.full_documents,
            timings=timings,
            retrieved_count=response.retrieved_count,
        )
    aggregate = await _try_structured_aggregate(request)
    if aggregate is not None:
        response = aggregate[0]
        timings = dict(response.timings)
        timings.update(expansion_timing)
        timings["retrieve_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return RetrieveResponse(
            query=response.query,
            route_plan=response.route_plan,
            evidences=response.evidences,
            full_documents=_full_documents_for_evidences(request, response.evidences),
            timings=timings,
            retrieved_count=response.retrieved_count,
        )
    if request.sources:
        route_plan = RoutePlan(
            document="document" in request.sources,
            table="table" in request.sources,
            adela="adela" in request.sources,
            reason="manual source selection from request.sources",
        )
    elif not request.route_with_llm:
        route_plan = service.route_sources(request.query, enabled_sources=_enabled_sources(request))
    else:
        route_plan = await _route_sources_with_llm(request)

    all_results = []
    source_timings: dict[str, dict[str, float]] = {}
    for source_type in route_plan.sources:
        source_cfg = getattr(request, source_type)
        source_top_k = source_cfg.top_k or request.top_k
        source_method = source_cfg.retrieval_method or request.retrieval_method
        source_threshold = (
            source_cfg.similarity_threshold
            if source_cfg.similarity_threshold is not None
            else request.similarity_threshold
        )
        results, timing = service.retrieve(
            query=request.query,
            source_types=[source_type],
            retrieval_method=source_method,
            top_k=source_top_k,
            candidate_limit=request.candidate_limit,
            query_expansion_terms=expansion_terms,
            embedding_model=request.embedding_model,
            embedding_models=request.embedding_models
            or (
                list(service.settings.EMBEDDING_MODELS)
                if source_type == "document"
                else [service.settings.EMBEDDING_MODEL]
            ),
            embedding_backend=request.embedding_backend,
            similarity_threshold=source_threshold,
        )
        all_results.extend(results)
        source_timings[source_type] = timing

    # Retrieve per source to preserve source diversity, then rank across
    # sources by source-local rank. Raw scores are not calibrated across
    # document/table/adela, so only use them as a tie-breaker.
    if _is_document_value_lookup(request.query) and any(result.chunk.source_type == "document" for result in all_results):
        all_results.sort(key=lambda item: (item.chunk.source_type == "document", item.score), reverse=True)
    else:
        all_results.sort(key=lambda item: item.score, reverse=True)
    all_results = _source_rank_fused_results_for_query(
        request.query,
        all_results,
        route_plan.sources,
        request.top_k * max(len(route_plan.sources), 1),
    )
    evidences = [EvidenceItem(**evidence_payload(result, request.query)) for result in all_results]
    timings = {
        "retrieve_ms": round((time.perf_counter() - started) * 1000, 3),
        "sources": source_timings,
        **expansion_timing,
    }
    return RetrieveResponse(
        query=request.query,
        route_plan=_route_plan_response(route_plan),
        evidences=evidences,
        full_documents=_full_documents_for_evidences(request, evidences),
        timings=timings,
        retrieved_count=len(evidences),
    )


async def _query_non_stream(
    request: QueryRequest,
    *,
    start: float | None = None,
) -> QueryResponse:
    started = start or time.perf_counter()
    expansion_terms, expansion_timing = await _resolve_query_expansion_terms(request)
    request.query_expansion_terms = expansion_terms
    deterministic_aggregate = _try_deterministic_aggregate(request)
    if deterministic_aggregate is not None:
        retrieved, answer = deterministic_aggregate
        timings = dict(retrieved.timings)
        timings.update(expansion_timing)
        timings["answer_ms"] = round((time.perf_counter() - started) * 1000, 3)
        payload = retrieved.model_dump()
        payload["timings"] = timings
        return QueryResponse(
            **payload,
            answer=answer,
            knowledge_base_fully_answered=await _knowledge_base_fully_answered_confidence(
                request.query,
                answer,
                retrieved.evidences,
                request.llm_config,
            ),
            llm_config={},
        )
    deployment_lookup = _try_structured_deployment_lookup(request)
    if deployment_lookup is not None:
        retrieved, answer = deployment_lookup
        timings = dict(retrieved.timings)
        timings.update(expansion_timing)
        timings["answer_ms"] = round((time.perf_counter() - started) * 1000, 3)
        payload = retrieved.model_dump()
        payload["timings"] = timings
        return QueryResponse(
            **payload,
            answer=answer,
            knowledge_base_fully_answered=await _knowledge_base_fully_answered_confidence(
                request.query,
                answer,
                retrieved.evidences,
                request.llm_config,
            ),
            llm_config={},
        )
    model_lookup = _try_structured_model_lookup(request)
    if model_lookup is not None:
        retrieved, answer = model_lookup
        timings = dict(retrieved.timings)
        timings.update(expansion_timing)
        timings["answer_ms"] = round((time.perf_counter() - started) * 1000, 3)
        payload = retrieved.model_dump()
        payload["timings"] = timings
        return QueryResponse(
            **payload,
            answer=answer,
            knowledge_base_fully_answered=await _knowledge_base_fully_answered_confidence(
                request.query,
                answer,
                retrieved.evidences,
                request.llm_config,
            ),
            llm_config={},
        )
    model_list = _try_structured_model_list(request)
    if model_list is not None:
        retrieved, answer = model_list
        timings = dict(retrieved.timings)
        timings.update(expansion_timing)
        timings["answer_ms"] = round((time.perf_counter() - started) * 1000, 3)
        payload = retrieved.model_dump()
        payload["timings"] = timings
        return QueryResponse(
            **payload,
            answer=answer,
            knowledge_base_fully_answered=await _knowledge_base_fully_answered_confidence(
                request.query,
                answer,
                retrieved.evidences,
                request.llm_config,
            ),
            llm_config={},
        )
    aggregate = await _try_structured_aggregate(request)
    if aggregate is not None:
        retrieved, answer = aggregate
        timings = dict(retrieved.timings)
        timings.update(expansion_timing)
        timings["answer_ms"] = round((time.perf_counter() - started) * 1000, 3)
        payload = retrieved.model_dump()
        payload["timings"] = timings
        return QueryResponse(
            **payload,
            answer=answer or "",
            knowledge_base_fully_answered=await _knowledge_base_fully_answered_confidence(
                request.query,
                answer or "",
                retrieved.evidences,
                request.llm_config,
            ),
            llm_config={},
        )

    retrieved = await _retrieve(request)
    answer, llm_config = await answer_with_llm(
        request.query,
        retrieved.evidences,
        request.llm_config,
    )
    answer = _complete_known_field_answer(request.query, answer, retrieved.evidences)
    answer = _complete_important_values_answer(request.query, answer, retrieved.evidences)
    answer = _complete_model_record_answer(request.query, answer, retrieved.evidences)
    answer = _complete_threshold_answer(request.query, answer, retrieved.evidences)
    timings = dict(retrieved.timings)
    if expansion_timing.get("query_expansion_method") != "disabled":
        timings.update(expansion_timing)
        timings["retrieve_ms"] = round(
            float(timings.get("retrieve_ms") or 0.0)
            + float(expansion_timing.get("query_expansion_ms") or 0.0),
            3,
        )
    timings["answer_ms"] = round((time.perf_counter() - started) * 1000 - timings.get("retrieve_ms", 0), 3)
    payload = retrieved.model_dump()
    payload["timings"] = timings
    return QueryResponse(
        **payload,
        answer=answer,
        knowledge_base_fully_answered=await _knowledge_base_fully_answered_confidence(
            request.query,
            answer,
            retrieved.evidences,
            request.llm_config,
        ),
        llm_config=llm_config,
    )


async def _query_stream(request: QueryRequest, start: float):
    expansion_terms, expansion_timing = await _resolve_query_expansion_terms(request)
    request.query_expansion_terms = expansion_terms
    deterministic_aggregate = _try_deterministic_aggregate(request)
    if deterministic_aggregate is not None:
        retrieved, answer = deterministic_aggregate
        timings = dict(retrieved.timings)
        timings.update(expansion_timing)
        timings["answer_ms"] = round((time.perf_counter() - start) * 1000, 3)
        payload = retrieved.model_dump()
        payload["timings"] = timings
        final_response = QueryResponse(
            **payload,
            answer=answer,
            knowledge_base_fully_answered=await _knowledge_base_fully_answered_confidence(
                request.query,
                answer,
                retrieved.evidences,
                request.llm_config,
            ),
            llm_config={},
        )
        yield _sse_event({"content": answer})
        yield _sse_event(_final_sse_payload(final_response))
        yield "data: [DONE]\n\n"
        return
    deployment_lookup = _try_structured_deployment_lookup(request)
    if deployment_lookup is not None:
        retrieved, answer = deployment_lookup
        timings = dict(retrieved.timings)
        timings.update(expansion_timing)
        timings["answer_ms"] = round((time.perf_counter() - start) * 1000, 3)
        payload = retrieved.model_dump()
        payload["timings"] = timings
        final_response = QueryResponse(
            **payload,
            answer=answer,
            knowledge_base_fully_answered=await _knowledge_base_fully_answered_confidence(
                request.query,
                answer,
                retrieved.evidences,
                request.llm_config,
            ),
            llm_config={},
        )
        yield _sse_event({"content": answer})
        yield _sse_event(_final_sse_payload(final_response))
        yield "data: [DONE]\n\n"
        return
    model_lookup = _try_structured_model_lookup(request)
    if model_lookup is not None:
        retrieved, answer = model_lookup
        timings = dict(retrieved.timings)
        timings.update(expansion_timing)
        timings["answer_ms"] = round((time.perf_counter() - start) * 1000, 3)
        payload = retrieved.model_dump()
        payload["timings"] = timings
        final_response = QueryResponse(
            **payload,
            answer=answer,
            knowledge_base_fully_answered=await _knowledge_base_fully_answered_confidence(
                request.query,
                answer,
                retrieved.evidences,
                request.llm_config,
            ),
            llm_config={},
        )
        yield _sse_event({"content": answer})
        yield _sse_event(_final_sse_payload(final_response))
        yield "data: [DONE]\n\n"
        return
    model_list = _try_structured_model_list(request)
    if model_list is not None:
        retrieved, answer = model_list
        timings = dict(retrieved.timings)
        timings.update(expansion_timing)
        timings["answer_ms"] = round((time.perf_counter() - start) * 1000, 3)
        payload = retrieved.model_dump()
        payload["timings"] = timings
        final_response = QueryResponse(
            **payload,
            answer=answer,
            knowledge_base_fully_answered=await _knowledge_base_fully_answered_confidence(
                request.query,
                answer,
                retrieved.evidences,
                request.llm_config,
            ),
            llm_config={},
        )
        yield _sse_event({"content": answer})
        yield _sse_event(_final_sse_payload(final_response))
        yield "data: [DONE]\n\n"
        return
    aggregate = await _try_structured_aggregate(request)
    if aggregate is not None:
        retrieved, answer = aggregate
        timings = dict(retrieved.timings)
        timings.update(expansion_timing)
        timings["answer_ms"] = round((time.perf_counter() - start) * 1000, 3)
        payload = retrieved.model_dump()
        payload["timings"] = timings
        final_response = QueryResponse(
            **payload,
            answer=answer or "",
            knowledge_base_fully_answered=await _knowledge_base_fully_answered_confidence(
                request.query,
                answer or "",
                retrieved.evidences,
                request.llm_config,
            ),
            llm_config={},
        )
        if answer:
            yield _sse_event({"content": answer})
        yield _sse_event(_final_sse_payload(final_response))
        yield "data: [DONE]\n\n"
        return

    retrieved = await _retrieve(request)
    answer_start = time.perf_counter()
    output_parts: list[str] = []
    llm_config = (request.llm_config or LLMConfig()).model_dump()
    timings = dict(retrieved.timings)
    if expansion_timing.get("query_expansion_method") != "disabled":
        timings.update(expansion_timing)
        timings["retrieve_ms"] = round(
            float(timings.get("retrieve_ms") or 0.0)
            + float(expansion_timing.get("query_expansion_ms") or 0.0),
            3,
        )

    try:
        async for chunk in stream_answer_with_llm(
            request.query,
            retrieved.evidences,
            request.llm_config,
        ):
            if chunk:
                output_parts.append(chunk)
                yield _sse_event({"content": chunk})
        answer = "".join(output_parts)
        answer = _complete_known_field_answer(request.query, answer, retrieved.evidences)
        answer = _complete_important_values_answer(request.query, answer, retrieved.evidences)
        answer = _complete_model_record_answer(request.query, answer, retrieved.evidences)
        answer = _complete_threshold_answer(request.query, answer, retrieved.evidences)
        raw_answer = "".join(output_parts)
        if answer != raw_answer and answer.startswith(raw_answer):
            delta = answer[len(raw_answer):]
            if delta:
                yield _sse_event({"content": delta})
        timings["answer_ms"] = round((time.perf_counter() - answer_start) * 1000, 3)
        payload = retrieved.model_dump()
        payload["timings"] = timings
        final_response = QueryResponse(
            **payload,
            answer=answer,
            knowledge_base_fully_answered=await _knowledge_base_fully_answered_confidence(
                request.query,
                answer,
                retrieved.evidences,
                request.llm_config,
            ),
            llm_config=llm_config,
        )
        yield _sse_event(_final_sse_payload(final_response))
        yield "data: [DONE]\n\n"
    except Exception as exc:
        if not output_parts:
            fallback = await _query_non_stream(request, start=start)
            if fallback.answer:
                yield _sse_event({"content": fallback.answer})
            yield _sse_event(
                _final_sse_payload(
                    fallback,
                    message="stream answer failed before first chunk, fell back to non-stream answer",
                )
            )
            yield "data: [DONE]\n\n"
            return

        answer = "".join(output_parts)
        timings["answer_ms"] = round((time.perf_counter() - answer_start) * 1000, 3)
        payload = retrieved.model_dump()
        payload["timings"] = timings
        final_response = QueryResponse(
            **payload,
            answer=answer,
            knowledge_base_fully_answered=await _knowledge_base_fully_answered_confidence(
                request.query,
                answer,
                retrieved.evidences,
                request.llm_config,
            ),
            llm_config=llm_config,
        )
        yield _sse_event(
            _final_sse_payload(
                final_response,
                success=False,
                message=f"RAG 流式回答生成失败: {exc}",
            )
        )
        yield "data: [DONE]\n\n"


@router.get("/health")
async def health():
    service = get_retrieval_service()
    source_counts = service.store.source_counts()
    minimums = service.settings.MIN_HEALTHY_SOURCE_COUNTS
    missing_or_small = {
        source: {
            "count": source_counts.get(source, 0),
            "expected_min": expected_min,
        }
        for source, expected_min in minimums.items()
        if source_counts.get(source, 0) < expected_min
    }
    return {
        "status": "ok" if not missing_or_small else "degraded",
        "chunks": service.store.count_chunks(),
        "sources": source_counts,
        "embeddings": service.store.embedding_counts(),
        "index_warnings": missing_or_small,
    }


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest):
    try:
        return await _retrieve(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        start = time.perf_counter()
        if request.stream:
            return _sse_response(_query_stream(request, start))
        return await _query_non_stream(request, start=start)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/chat_engine/unified_retrieve", response_model=RetrieveResponse)
async def unified_retrieve(request: RetrieveRequest):
    return await retrieve(request)


@router.post("/chat_engine/unified_query", response_model=QueryResponse)
async def unified_query(request: QueryRequest):
    return await query(request)


@router.post("/chat_engine/query", response_model=QueryResponse)
async def document_query(request: QueryRequest):
    request.sources = ["document"]
    return await query(request)


@router.post("/chat_engine/table_query", response_model=QueryResponse)
async def table_query(request: QueryRequest):
    request.sources = ["table"]
    return await query(request)


@router.post("/chat_engine/adela_query", response_model=QueryResponse)
async def adela_query(request: QueryRequest):
    request.sources = ["adela"]
    return await query(request)


@router.post("/embedding", response_model=EmbeddingResponse)
async def embedding(request: EmbeddingRequest):
    try:
        texts = request.input if isinstance(request.input, list) else [request.input]
        model = request.model
        backend = get_embedding_manager().get(model, request.embedding_backend)
        vectors = backend.encode(texts)
        return EmbeddingResponse(
            data=[
                EmbeddingObject(index=idx, embedding=vector.astype(float).tolist())
                for idx, vector in enumerate(vectors)
            ],
            model=model or backend.model_name,
            usage={"prompt_tokens": 0, "total_tokens": 0},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
