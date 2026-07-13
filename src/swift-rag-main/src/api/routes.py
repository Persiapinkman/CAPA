import asyncio
import json
import re
from collections import defaultdict
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import time
from typing import Any, Dict, List, Literal, Tuple

from src.api.schemas import (
    AdelaChatRequest,
    AdelaChatResponse,
    AdelaChatTimings,
    ChunkingEmbeddingRequest,
    ChunkingEmbeddingResponse,
    RetrievingRequest,
    TableChatRequest,
    TableChatResponse,
    TableChatTimings,
    EmbeddingRequest,
    EmbeddingList,
    Embedding,
    Usage,
    RAGChatRequest,
    RAGChatResponse,
    RAGChatTimings,
    ReferenceItem,
    TableMatchedRow,
    UnifiedEvidenceItem,
    UnifiedRetrieveRequest,
    UnifiedRetrieveResponse,
    UnifiedRetrieveTimings,
    UnifiedQueryRequest,
    UnifiedQueryResponse,
    UnifiedRoutePlan,
    UnifiedQueryTimings,
    UnifiedSourceStatus,
)

from src.core.config import get_settings
from src.core.logging import get_logger
from src.rag.qa_service import (
    answer_adela_question,
    answer_question,
    answer_table_question,
    plan_structured_aggregate_query,
    review_structured_aggregate_result,
    route_unified_sources,
    answer_unified_question,
    answer_unified_question_stream,
)
from src.rag.reference_service import PDFReferenceStore
from src.rag.runtime import get_rag_service
from src.rag.timing_store import rag_chat_timing_store

# 配置日志
logger = get_logger(__name__)
settings = get_settings()

# 创建路由
router = APIRouter(prefix="/rag", tags=["RAG"])

pdf_reference_store = PDFReferenceStore()


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


def _model_dump_compatible(model: Any) -> Any:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return model


def _truncate_text(text: str | None, max_len: int = 500) -> str:
    if text is None:
        return ""
    content = str(text).strip()
    if len(content) <= max_len:
        return content
    return content[: max_len - 3] + "..."


def _could_be_structured_aggregate_query(query: str) -> bool:
    normalized = str(query or "").lower()
    return any(
        keyword in normalized
        for keyword in ("多少", "几个", "几款", "数量", "总数", "总共", "一共", "共有", "统计")
    )


def _build_aggregate_source_profile(
    rows: List[Dict[str, Any]],
    source_type: str,
    max_values_per_field: int = 30,
) -> str:
    if source_type == "adela":
        fields = ["model_name", "name", "platform", "status", "version", "label_list"]
    elif source_type == "public_cloud":
        fields = ["id", "owned_by", "object"]
    else:
        fields = [
            "target_name",
            "algorithm_type",
            "algorithm_name",
            "application_scene",
            "owner",
            "model_name",
            "supported_device",
            "recommended_config",
        ]

    lines = [f"{source_type}: total_records={len(rows)}"]
    for field in fields:
        counts: Dict[str, int] = defaultdict(int)
        for row in rows:
            value = str(row.get(field) or "").strip()
            if not value:
                continue
            counts[value] += 1
        if not counts:
            continue
        values = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:max_values_per_field]
        value_text = "; ".join(f"{value}({count})" for value, count in values)
        lines.append(f"- {field}: {value_text}")
    return "\n".join(lines)


def _match_aggregate_condition(row: Dict[str, Any], condition: Dict[str, Any]) -> bool:
    field = str(condition.get("field") or "").strip()
    operator = str(condition.get("operator") or "contains").strip()
    values = [
        str(value).strip()
        for value in condition.get("values") or []
        if str(value).strip()
    ]
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
    row: Dict[str, Any],
    field_conditions: List[Dict[str, Any]],
    condition_logic: str,
) -> bool:
    if not field_conditions:
        return True

    matches = [_match_aggregate_condition(row, condition) for condition in field_conditions]
    if condition_logic == "any":
        return any(matches)
    return all(matches)


def _default_count_field(source_type: str) -> str:
    if source_type == "adela":
        return "model_name"
    if source_type == "public_cloud":
        return "id"
    return "model_name"


def _aggregate_source_label(source_type: str) -> str:
    if source_type == "adela":
        return "Adela 部署记录"
    if source_type == "public_cloud":
        return "公有云在线模型列表"
    return "模型发版信息汇总表"


def _build_aggregate_breakdown(
    rows: List[Dict[str, Any]],
    breakdown_fields: List[str],
    count_field: str,
    max_items_per_field: int = 12,
) -> Dict[str, List[Dict[str, Any]]]:
    breakdown: Dict[str, List[Dict[str, Any]]] = {}
    for field in breakdown_fields:
        field = str(field or "").strip()
        if not field:
            continue
        buckets: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            value = str(row.get(field) or "").strip()
            if not value:
                continue
            bucket = buckets.setdefault(value, {"value": value, "record_count": 0, "unique_count": 0})
            bucket["record_count"] += 1
            if "_values" not in bucket:
                bucket["_values"] = set()
            count_value = str(row.get(count_field) or "").strip()
            if count_value:
                bucket["_values"].add(count_value)
        items: List[Dict[str, Any]] = []
        for bucket in buckets.values():
            unique_values = bucket.pop("_values", set())
            bucket["unique_count"] = len(unique_values)
            items.append(bucket)
        items.sort(key=lambda item: (-int(item["record_count"]), str(item["value"])))
        breakdown[field] = items[:max_items_per_field]
    return breakdown


def _format_aggregate_breakdown_for_answer(breakdown: Dict[str, List[Dict[str, Any]]]) -> str:
    parts = []
    for field, items in breakdown.items():
        if not items:
            continue
        value_text = "、".join(
            f"{item['value']}({item['unique_count']}个/{item['record_count']}条)"
            for item in items[:8]
        )
        if value_text:
            parts.append(f"{field}: {value_text}")
    return "；".join(parts)


def _build_structured_aggregate_answer(
    query: str,
    rows: List[Dict[str, Any]],
    data_path: str,
    plan: Dict[str, Any],
) -> Tuple[str, TableMatchedRow]:
    source_type = str(plan.get("source_type") or "table")
    operation = str(plan.get("operation") or "count_unique")
    dedupe_field = str(plan.get("dedupe_field") or "")
    count_field = dedupe_field or str(plan.get("count_field") or "") or _default_count_field(source_type)
    count_semantics = str(plan.get("count_semantics") or "")
    confidence = str(plan.get("confidence") or "medium")
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"
    ambiguity = str(plan.get("ambiguity") or "").strip()
    breakdown_fields = [
        str(field).strip()
        for field in list(plan.get("breakdown_fields") or [])
        if str(field).strip()
    ]
    condition_logic = str(plan.get("condition_logic") or "all")
    if condition_logic not in {"all", "any"}:
        condition_logic = "all"
    field_conditions = [
        condition
        for condition in list(plan.get("field_conditions") or [])
        if isinstance(condition, dict)
    ]

    filtered_rows = [
        row
        for row in rows
        if _row_matches_aggregate_conditions(row, field_conditions, condition_logic)
    ]

    record_count = len(rows)
    filtered_record_count = len(filtered_rows)
    source_label = _aggregate_source_label(source_type)
    breakdown = _build_aggregate_breakdown(filtered_rows, breakdown_fields, count_field)

    if operation == "count_records":
        result_count = filtered_record_count
        dedupe_count = 0
        count_desc = "记录"
        answer = f"RD 部门当前{source_label}中"
        if field_conditions:
            answer += f"满足字段条件 {field_conditions} 的"
        answer += f"记录共有 {result_count} 条。"
    else:
        values = {
            str(row.get(count_field)).strip()
            for row in filtered_rows
            if row.get(count_field) not in (None, "")
        }
        result_count = len(values)
        dedupe_count = max(0, filtered_record_count - result_count)
        count_desc = f"唯一 `{count_field}`"
        answer = f"RD 部门当前{source_label}中"
        if field_conditions:
            answer += f"满足字段条件 {field_conditions} 的"
        answer += (
            f"{count_desc} 共有 {result_count} 个；"
            f"参与统计的记录数为 {filtered_record_count} 条。"
        )
        if dedupe_count:
            answer += f"其中有 {dedupe_count} 条记录与其他记录共享相同 `{count_field}`。"

    breakdown_text = _format_aggregate_breakdown_for_answer(breakdown)
    if breakdown_text:
        answer += f" 分组概览：{breakdown_text}。"
    if count_semantics:
        answer += f" 统计口径：{count_semantics}。"
    if ambiguity:
        answer += f" 口径提示：{ambiguity}"
    if confidence != "high":
        answer += f" 置信度：{confidence}。"

    if not field_conditions and filtered_record_count != record_count:
        answer += f"原始数据源共有 {record_count} 条记录。"

    condition_fields = [
        str(condition.get("field") or "").strip()
        for condition in field_conditions
        if str(condition.get("field") or "").strip()
    ]
    row = TableMatchedRow(
        row_id=f"aggregate::{source_type}::{operation}",
        score=1.0,
        matched_fields=[count_field, *condition_fields],
        entity={
            "row_id": f"aggregate::{source_type}::{operation}",
            "name": "RD 结构化统计",
            "aggregate_type": operation,
            "source_type": source_type,
            "result_count": result_count,
            "count_desc": count_desc,
            "count_field": count_field,
            "dedupe_field": count_field,
            "count_semantics": count_semantics,
            "record_count": record_count,
            "filtered_record_count": filtered_record_count,
            "deduplicated_record_count": dedupe_count,
            "breakdown_fields": breakdown_fields,
            "breakdown": breakdown,
            "confidence": confidence,
            "ambiguity": ambiguity,
            "condition_logic": condition_logic,
            "field_conditions": field_conditions,
            "planner_reason": plan.get("reason"),
            "data_path": str(data_path),
            "query": query,
        },
    )
    return answer, row


async def _try_answer_structured_aggregate_question(
    query: str,
    data_paths: Dict[str, str],
    enabled_sources: List[Literal["table", "adela", "public_cloud"]],
    rag_service: Any,
    llm_config: Any = None,
    source_dirs: Dict[str, str | None] | None = None,
) -> Tuple[str, TableMatchedRow] | None:
    if not _could_be_structured_aggregate_query(query):
        return None

    preloaded_rows: Dict[str, List[Dict[str, Any]]] = {}
    profile_parts: List[str] = []
    for source_type in enabled_sources:
        if source_type == "table":
            rows = rag_service._load_table_rows(data_paths[source_type])
            preloaded_rows[source_type] = rows
            profile_parts.append(_build_aggregate_source_profile(rows, source_type))
        elif source_type == "adela":
            source_dir = (source_dirs or {}).get("adela")
            data_path = rag_service._ensure_adela_rows_file(
                data_path=data_paths[source_type],
                source_dir=source_dir,
                searchable_fields=settings.ADELA_SEARCHABLE_FIELDS,
            )
            rows = rag_service._load_table_rows(data_path)
            preloaded_rows[source_type] = rows
            data_paths[source_type] = data_path
            profile_parts.append(_build_aggregate_source_profile(rows, source_type))
        elif source_type == "public_cloud":
            profile_parts.append(
                "public_cloud: fields=id, owned_by, object, created; values are fetched from public cloud API when selected."
            )

    plan = await plan_structured_aggregate_query(
        query=query,
        enabled_sources=enabled_sources,
        llm_config=llm_config,
        source_profiles="\n\n".join(profile_parts),
    )
    if not plan or not plan.get("should_aggregate"):
        return None

    source_type = str(plan.get("source_type") or "")
    if source_type not in enabled_sources:
        return None

    if source_type == "adela":
        data_path = data_paths[source_type]
        rows = preloaded_rows.get(source_type) or rag_service._load_table_rows(data_path)
    elif source_type == "public_cloud":
        data_path = data_paths[source_type]
        rows = _fetch_public_cloud_models(
            api_url=data_path,
            api_token=(source_dirs or {}).get("public_cloud_token") or settings.PUBLIC_CLOUD_MODELS_API_TOKEN,
        )
    else:
        data_path = data_paths[source_type]
        rows = preloaded_rows.get(source_type) or rag_service._load_table_rows(data_path)

    answer, aggregate_row = _build_structured_aggregate_answer(
        query=query,
        rows=rows,
        data_path=data_path,
        plan=plan,
    )
    result_summary = {
        key: value
        for key, value in aggregate_row.entity.items()
        if key
        in {
            "aggregate_type",
            "source_type",
            "result_count",
            "count_desc",
            "count_field",
            "dedupe_field",
            "count_semantics",
            "record_count",
            "filtered_record_count",
            "deduplicated_record_count",
            "condition_logic",
            "field_conditions",
            "breakdown_fields",
            "breakdown",
            "confidence",
            "ambiguity",
            "planner_reason",
        }
    }
    review = await review_structured_aggregate_result(
        query=query,
        plan=plan,
        result_summary=result_summary,
        llm_config=llm_config,
    )
    if review:
        aggregate_row.entity["review"] = review
        aggregate_row.entity["confidence"] = review.get("confidence") or aggregate_row.entity.get("confidence")
        aggregate_row.entity["ambiguity"] = review.get("ambiguity") or aggregate_row.entity.get("ambiguity")
        if review.get("answer"):
            answer = str(review["answer"])

    return answer, aggregate_row


def _build_table_like_snippet(entity: Dict[str, Any]) -> str:
    preferred_keys = [
        "model_name",
        "name",
        "platform",
        "version",
        "target_name",
        "algorithm_name",
        "supported_device",
        "status",
        "did",
        "rid",
    ]
    pairs = []
    for key in preferred_keys:
        value = entity.get(key)
        if value in (None, ""):
            continue
        pairs.append(f"{key}: {value}")
    if not pairs:
        pairs = [
            f"{key}: {value}"
            for key, value in entity.items()
            if value not in (None, "")
        ]
    return _truncate_text("; ".join(pairs), max_len=500)


def _resolve_adela_json_path(entity: Dict[str, Any]) -> Path | None:
    source_path = entity.get("source_path")
    if source_path:
        path = Path(str(source_path))
        if path.exists():
            return path.resolve()

    source_file = entity.get("source_file")
    if source_file:
        path = Path(str(source_file))
        if path.is_absolute() and path.exists():
            return path.resolve()
        adela_root = Path(settings.ADELA_DATA_DIR).resolve().parent
        candidate = adela_root / path
        if candidate.exists():
            return candidate.resolve()

    return None


def _build_adela_reference_url(did: Any) -> str | None:
    if did in (None, ""):
        return None
    did_text = str(did).strip()
    if not did_text:
        return None
    return settings.ADELA_DEPLOYMENT_URL_TEMPLATE.format(did=did_text)


def _resolve_adela_references(rows: List[TableMatchedRow]) -> List[ReferenceItem]:
    references: List[ReferenceItem] = []
    seen_dids = set()
    for row in rows:
        entity = dict(row.entity or {})
        did = entity.get("did")
        did_text = str(did).strip() if did not in (None, "") else ""
        if not did_text or did_text in seen_dids:
            continue

        url = entity.get("reference") or _build_adela_reference_url(did_text)
        if not url:
            continue

        model_name = entity.get("model_name") or entity.get("name")
        doc_name = f"{model_name} (did={did_text})" if model_name else f"did={did_text}"
        references.append(ReferenceItem(doc_name=doc_name, url=url))
        seen_dids.add(did_text)

    return references


def _resolve_unified_references(
    fused_evidences: List[UnifiedEvidenceItem],
) -> List[ReferenceItem]:
    references: List[ReferenceItem] = []
    seen_keys = set()

    doc_names: List[str] = []
    for evidence in fused_evidences:
        if evidence.source_type != "document":
            continue
        payload = dict(evidence.payload or {})
        doc_name = payload.get("doc_name") or evidence.title
        if not doc_name:
            continue
        doc_names.append(str(doc_name))

    for item in pdf_reference_store.resolve_doc_names(doc_names):
        key = (item.doc_name, item.url)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        references.append(item)

    seen_dids = set()
    for evidence in fused_evidences:
        if evidence.source_type != "adela":
            continue
        payload = dict(evidence.payload or {})
        entity = dict(payload.get("entity") or {})

        did = entity.get("did")
        did_text = str(did).strip() if did not in (None, "") else ""
        if not did_text or did_text in seen_dids:
            continue

        url = entity.get("reference") or _build_adela_reference_url(did_text)
        if not url:
            continue

        model_name = entity.get("model_name") or entity.get("name")
        doc_name = f"{model_name} (did={did_text})" if model_name else f"did={did_text}"
        item = ReferenceItem(doc_name=doc_name, url=url)
        key = (item.doc_name, item.url)
        if key in seen_keys:
            seen_dids.add(did_text)
            continue

        seen_keys.add(key)
        seen_dids.add(did_text)
        references.append(item)

    return references


def _build_document_evidences(chunks: List[Any]) -> List[UnifiedEvidenceItem]:
    evidences: List[UnifiedEvidenceItem] = []
    for rank, chunk in enumerate(chunks, start=1):
        snippet = _truncate_text(chunk.index_text or chunk.text, max_len=500)
        evidence_id = f"document::{chunk.id}"
        evidences.append(
            UnifiedEvidenceItem(
                evidence_id=evidence_id,
                source_type="document",
                score=0.0,
                source_rank=rank,
                source_score=chunk.score,
                title=chunk.doc_name or chunk.metadata.get("file_name") or chunk.id,
                snippet=snippet,
                payload={
                    "id": chunk.id,
                    "doc_id": chunk.doc_id,
                    "doc_name": chunk.doc_name,
                    "index_id": chunk.index_id,
                    "text": _truncate_text(chunk.text, max_len=300),
                    "index_text": _truncate_text(chunk.index_text, max_len=500),
                    "metadata": chunk.metadata,
                },
            )
        )
    return evidences


def _build_structured_evidences(
    rows: List[TableMatchedRow],
    source_type: Literal["table", "adela", "public_cloud"],
) -> List[UnifiedEvidenceItem]:
    evidences: List[UnifiedEvidenceItem] = []
    for rank, row in enumerate(rows, start=1):
        entity = dict(row.entity or {})
        title = (
            str(entity.get("model_name") or entity.get("name") or row.row_id)
            if entity
            else row.row_id
        )
        payload = {
            "row_id": row.row_id,
            "matched_fields": row.matched_fields,
            "entity": entity,
        }
        if source_type == "adela":
            adela_json_path = _resolve_adela_json_path(entity)
            if adela_json_path is not None:
                payload["json_path"] = str(adela_json_path)
                payload["json_link"] = adela_json_path.as_uri()

        evidences.append(
            UnifiedEvidenceItem(
                evidence_id=f"{source_type}::{row.row_id}",
                source_type=source_type,
                score=0.0,
                source_rank=rank,
                source_score=row.score,
                title=title,
                snippet=_build_table_like_snippet(entity),
                payload=payload,
            )
        )
    return evidences


def _rrf_fuse_evidences(
    source_lists: List[List[UnifiedEvidenceItem]],
    fused_top_k: int,
    rrf_k: int,
) -> List[UnifiedEvidenceItem]:
    score_map: Dict[str, float] = defaultdict(float)
    evidence_map: Dict[str, UnifiedEvidenceItem] = {}

    for evidences in source_lists:
        for rank, evidence in enumerate(evidences, start=1):
            score_map[evidence.evidence_id] += 1.0 / float(rrf_k + rank)
            if evidence.evidence_id not in evidence_map:
                evidence_map[evidence.evidence_id] = evidence

    source_priority = {
        "document": 0,
        "table": 1,
        "adela": 2,
        "public_cloud": 3,
    }
    sorted_ids = sorted(
        score_map.keys(),
        key=lambda key: (
            -float(score_map[key]),
            source_priority.get(evidence_map[key].source_type, 99),
            int(evidence_map[key].source_rank),
            -float(evidence_map[key].source_score),
            key,
        ),
    )
    fused: List[UnifiedEvidenceItem] = []
    for evidence_id in sorted_ids[:fused_top_k]:
        base = evidence_map[evidence_id]
        fused.append(
            UnifiedEvidenceItem(
                evidence_id=base.evidence_id,
                source_type=base.source_type,
                score=round(float(score_map[evidence_id]), 6),
                source_rank=base.source_rank,
                source_score=base.source_score,
                title=base.title,
                snippet=base.snippet,
                payload=base.payload,
            )
        )
    return fused


def _build_rag_chat_timing_record(
    request: RAGChatRequest,
    timings: Dict[str, float],
    success: bool,
    retrieved_count: int = 0,
    reference_count: int = 0,
    answer_length: int = 0,
    error_message: str | None = None,
) -> Dict[str, Any]:
    llm_model = None
    llm_base_url = None
    if request.llm_config is not None:
        llm_model = request.llm_config.model
        llm_base_url = request.llm_config.base_url

    vector_store_configs = None
    if request.vector_store_configs is not None:
        vector_store_configs = {
            model_name: cfg.model_dump()
            for model_name, cfg in request.vector_store_configs.items()
        }

    return {
        "event": "rag_chat_query",
        "success": success,
        "query": request.query,
        "retrieval_method": request.retrieval_method,
        "query_length": len(request.query),
        "top_k": request.top_k,
        "similarity_threshold": request.similarity_threshold,
        "filter": request.filter,
        "uri": request.uri,
        "collection_name": request.collection_name,
        "embedding_models": request.embedding_models,
        "vector_store_configs": vector_store_configs,
        "llm_model": llm_model,
        "llm_base_url": llm_base_url,
        "retrieved_count": retrieved_count,
        "reference_count": reference_count,
        "answer_length": answer_length,
        "timings_ms": timings,
        "error_message": error_message,
    }

@router.post(
    "/doc_engine/chunking_embedding",
    response_model=ChunkingEmbeddingResponse,
    tags=["Document Engine"],
)
async def process_chunking_embedding(request: ChunkingEmbeddingRequest):
    """
    处理文档分块和向量化
    """
    logger.info(f"Received document processing request: doc_id={request.doc_id}, doc_name={request.doc_name}")
    try:
        rag_service = get_rag_service()
        start_time = time.time()
        index_nodes = rag_service.chunking_embedding(request)

        logger.info(f"Document processing successful: doc_id={request.doc_id}, time={time.time() - start_time:.2f}s")
        return ChunkingEmbeddingResponse(
            doc_id=request.doc_id, doc_name=request.doc_name, index_nodes=index_nodes
        )

    except ValueError as e:
        logger.warning(f"Invalid document processing request: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Document processing error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during document processing")


@router.post(
    "/chat_engine/query",
    response_model=RAGChatResponse,
    tags=["Chat Engine"],
)
async def query_with_retrieval(request: RAGChatRequest):
    """document 文档问答接口，返回检索结果和最终回答。"""
    logger.info(
        "Received RAG chat request: query='%s', method=%s, top_k=%s",
        request.query,
        request.retrieval_method,
        request.top_k,
    )
    total_start = time.perf_counter()
    timings: Dict[str, float] = {
        "retrieve_ms": 0.0,
        "answer_ms": 0.0,
        "reference_ms": 0.0,
        "total_ms": 0.0,
    }
    retrieved_chunks = []
    references = []
    answer = ""
    try:
        rag_service = get_rag_service()
        retrieving_request = RetrievingRequest(
            query=request.query,
            retrieval_method=request.retrieval_method,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            filter=request.filter,
            uri=request.uri,
            collection_name=request.collection_name,
            embedding_models=request.embedding_models,
            vector_store_configs=request.vector_store_configs,
        )

        retrieve_start = time.perf_counter()
        retrieved_chunks = rag_service.retrieving(retrieving_request)
        timings["retrieve_ms"] = _elapsed_ms(retrieve_start)

        answer_start = time.perf_counter()
        answer = await answer_question(
            query=request.query,
            retrieved_chunks=retrieved_chunks,
            llm_config=request.llm_config,
        )
        timings["answer_ms"] = _elapsed_ms(answer_start)

        reference_start = time.perf_counter()
        references = pdf_reference_store.resolve_references(retrieved_chunks)
        timings["reference_ms"] = _elapsed_ms(reference_start)
        timings["total_ms"] = _elapsed_ms(total_start)

        rag_chat_timing_store.append(
            _build_rag_chat_timing_record(
                request=request,
                timings=timings,
                success=True,
                retrieved_count=len(retrieved_chunks),
                reference_count=len(references),
                answer_length=len(answer),
            )
        )
        logger.info(
            "RAG chat completed: retrieved=%s, retrieve_ms=%.3f, answer_ms=%.3f, reference_ms=%.3f, total_ms=%.3f",
            len(retrieved_chunks),
            timings["retrieve_ms"],
            timings["answer_ms"],
            timings["reference_ms"],
            timings["total_ms"],
        )
        return RAGChatResponse(
            query=request.query,
            retrieved_chunks=retrieved_chunks,
            reference=references,
            answer=answer,
            timings=RAGChatTimings(**timings),
        )
    except ValueError as e:
        timings["total_ms"] = _elapsed_ms(total_start)
        rag_chat_timing_store.append(
            _build_rag_chat_timing_record(
                request=request,
                timings=timings,
                success=False,
                retrieved_count=len(retrieved_chunks),
                reference_count=len(references),
                answer_length=len(answer),
                error_message=str(e),
            )
        )
        logger.warning("Invalid RAG chat request: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        timings["total_ms"] = _elapsed_ms(total_start)
        rag_chat_timing_store.append(
            _build_rag_chat_timing_record(
                request=request,
                timings=timings,
                success=False,
                retrieved_count=len(retrieved_chunks),
                reference_count=len(references),
                answer_length=len(answer),
                error_message=str(e),
            )
        )
        logger.error("RAG chat processing error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"RAG问答处理过程中发生错误: {str(e)}")


@router.post(
    "/chat_engine/table_query",
    response_model=TableChatResponse,
    tags=["Chat Engine"],
)
async def query_table_with_retrieval(request: TableChatRequest):
    """table 表格问答接口，返回命中行和最终回答。"""
    logger.info(
        "Received table chat request: query='%s', method=%s, top_k=%s",
        request.query,
        request.retrieval_method,
        request.top_k,
    )
    total_start = time.perf_counter()
    timings: Dict[str, float] = {
        "retrieve_ms": 0.0,
        "answer_ms": 0.0,
        "total_ms": 0.0,
    }
    matched_rows = []
    answer = ""
    try:
        rag_service = get_rag_service()

        retrieve_start = time.perf_counter()
        aggregate_result = await _try_answer_structured_aggregate_question(
            query=request.query,
            data_paths={"table": request.data_path},
            enabled_sources=["table"],
            rag_service=rag_service,
            llm_config=request.llm_config,
        )
        if aggregate_result is not None:
            answer, aggregate_row = aggregate_result
            matched_rows = [aggregate_row]
            timings["retrieve_ms"] = _elapsed_ms(retrieve_start)
            timings["answer_ms"] = 0.0
            timings["total_ms"] = _elapsed_ms(total_start)
            logger.info(
                "Table aggregate chat completed: matched=%s, retrieve_ms=%.3f, total_ms=%.3f",
                len(matched_rows),
                timings["retrieve_ms"],
                timings["total_ms"],
            )
            return TableChatResponse(
                query=request.query,
                matched_rows=matched_rows,
                answer=answer,
                timings=TableChatTimings(**timings),
                message="answered_by_agentic_aggregate",
            )

        matched_rows = rag_service.table_chat_retrieving(request)
        timings["retrieve_ms"] = _elapsed_ms(retrieve_start)

        answer_start = time.perf_counter()
        answer = await answer_table_question(
            query=request.query,
            matched_rows=matched_rows,
            llm_config=request.llm_config,
        )
        timings["answer_ms"] = _elapsed_ms(answer_start)
        timings["total_ms"] = _elapsed_ms(total_start)

        logger.info(
            "Table chat completed: matched=%s, retrieve_ms=%.3f, answer_ms=%.3f, total_ms=%.3f",
            len(matched_rows),
            timings["retrieve_ms"],
            timings["answer_ms"],
            timings["total_ms"],
        )
        return TableChatResponse(
            query=request.query,
            matched_rows=matched_rows,
            answer=answer,
            timings=TableChatTimings(**timings),
        )
    except ValueError as e:
        timings["total_ms"] = _elapsed_ms(total_start)
        logger.warning("Invalid table chat request: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        timings["total_ms"] = _elapsed_ms(total_start)
        logger.error("Table chat processing error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"表格问答处理过程中发生错误: {str(e)}")


@router.post(
    "/chat_engine/adela_query",
    response_model=AdelaChatResponse,
    tags=["Chat Engine"],
)
async def query_adela_with_retrieval(request: AdelaChatRequest):
    """adela 部署记录问答接口，返回命中记录和最终回答。"""
    logger.info(
        "Received adela chat request: query='%s', method=%s, top_k=%s",
        request.query,
        request.retrieval_method,
        request.top_k,
    )
    total_start = time.perf_counter()
    timings: Dict[str, float] = {
        "retrieve_ms": 0.0,
        "answer_ms": 0.0,
        "total_ms": 0.0,
    }
    matched_records = []
    references = []
    answer = ""
    try:
        rag_service = get_rag_service()

        retrieve_start = time.perf_counter()
        matched_records = rag_service.adela_chat_retrieving(request)
        timings["retrieve_ms"] = _elapsed_ms(retrieve_start)
        references = _resolve_adela_references(matched_records)

        answer_start = time.perf_counter()
        answer = await answer_adela_question(
            query=request.query,
            matched_rows=matched_records,
            llm_config=request.llm_config,
        )
        timings["answer_ms"] = _elapsed_ms(answer_start)
        timings["total_ms"] = _elapsed_ms(total_start)

        logger.info(
            "Adela chat completed: matched=%s, retrieve_ms=%.3f, answer_ms=%.3f, total_ms=%.3f",
            len(matched_records),
            timings["retrieve_ms"],
            timings["answer_ms"],
            timings["total_ms"],
        )
        return AdelaChatResponse(
            query=request.query,
            matched_records=matched_records,
            reference=references,
            answer=answer,
            timings=AdelaChatTimings(**timings),
        )
    except ValueError as e:
        timings["total_ms"] = _elapsed_ms(total_start)
        logger.warning("Invalid adela chat request: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        timings["total_ms"] = _elapsed_ms(total_start)
        logger.error("Adela chat processing error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"adela问答处理过程中发生错误: {str(e)}")


def _build_disabled_source_status(
    source_type: Literal["document", "table", "adela", "public_cloud"],
) -> UnifiedSourceStatus:
    return UnifiedSourceStatus(
        source_type=source_type,
        enabled=False,
        success=True,
        retrieve_ms=0.0,
        retrieved_count=0,
        used_count=0,
        message="disabled",
    )


def _build_skipped_source_status(
    source_type: Literal["document", "table", "adela", "public_cloud"],
    reason: str = "skipped_by_router",
) -> UnifiedSourceStatus:
    return UnifiedSourceStatus(
        source_type=source_type,
        enabled=True,
        success=True,
        retrieve_ms=0.0,
        retrieved_count=0,
        used_count=0,
        message=reason,
    )


def _tokenize_public_cloud_query(text: str) -> List[str]:
    normalized = str(text or "").lower()
    raw_tokens = re.split(r"[^0-9a-z\u4e00-\u9fff]+", normalized)
    return [token for token in raw_tokens if token]


def _fetch_public_cloud_models(api_url: str, api_token: str, timeout: float = 8.0) -> List[Dict[str, Any]]:
    req = urllib_request.Request(
        url=api_url,
        headers={"Authorization": f"Bearer {api_token}"},
        method="GET",
    )
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        raise ValueError(f"public_cloud 接口返回异常状态码: {exc.code}") from exc
    except urllib_error.URLError as exc:
        raise ValueError(f"public_cloud 接口访问失败: {exc.reason}") from exc
    except Exception as exc:
        raise ValueError(f"public_cloud 接口解析失败: {exc}") from exc

    data = payload.get("data")
    if not isinstance(data, list):
        return []

    models: List[Dict[str, Any]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id") or "").strip()
        if not model_id:
            continue
        models.append(
            {
                "row_id": f"public_cloud::{model_id}::{idx}",
                "id": model_id,
                "object": item.get("object"),
                "owned_by": item.get("owned_by"),
                "created": item.get("created"),
            }
        )
    return models


def _retrieve_public_cloud_rows(query: str, api_url: str, api_token: str, top_k: int) -> List[TableMatchedRow]:
    rows = _fetch_public_cloud_models(api_url=api_url, api_token=api_token)
    if not rows:
        return []

    query_tokens = _tokenize_public_cloud_query(query)
    lowered_query = query.lower()
    ranked: List[TableMatchedRow] = []

    for row in rows:
        model_id = str(row.get("id") or "")
        model_id_lower = model_id.lower()
        matched_fields: List[str] = []
        score = 0.0

        if model_id_lower and model_id_lower in lowered_query:
            score += 1.0
            matched_fields.append("id")

        token_hits = 0
        for token in query_tokens:
            if token in model_id_lower:
                token_hits += 1
        if token_hits > 0:
            score += min(0.8, token_hits * 0.2)
            if "id" not in matched_fields:
                matched_fields.append("id")

        if any(keyword in lowered_query for keyword in ("公有云", "云上", "vllm", "在线模型", "models")):
            score += 0.15

        if score <= 0.0:
            continue

        ranked.append(
            TableMatchedRow(
                row_id=row["row_id"],
                score=round(float(score), 6),
                matched_fields=matched_fields,
                entity={
                    "id": row.get("id"),
                    "owned_by": row.get("owned_by"),
                    "object": row.get("object"),
                    "created": row.get("created"),
                    "source_api": api_url,
                },
            )
        )

    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked[: max(1, top_k)]


def _resolve_selected_sources(
    source_enabled_map: Dict[Literal["document", "table", "adela", "public_cloud"], bool],
    requested_sources: List[Literal["document", "table", "adela", "public_cloud"]] | None,
) -> List[Literal["document", "table", "adela", "public_cloud"]]:
    if requested_sources is None:
        selected = [
            source_type
            for source_type, enabled in source_enabled_map.items()
            if enabled
        ]
        if not selected:
            raise ValueError("没有可用的数据源，请至少启用一个数据源配置。")
        return selected

    if not requested_sources:
        raise ValueError("source_types 不能为空；可选值为 document/table/adela/public_cloud。")

    deduplicated: List[Literal["document", "table", "adela", "public_cloud"]] = []
    seen = set()
    for source_type in requested_sources:
        if source_type in seen:
            continue
        deduplicated.append(source_type)
        seen.add(source_type)

    selected = [
        source_type
        for source_type in deduplicated
        if source_enabled_map.get(source_type, False)
    ]
    if not selected:
        raise ValueError(
            "source_types 指定的数据源均未启用，请检查 document/table/adela/public_cloud 的 enabled 配置。"
        )
    return selected


@router.post(
    "/chat_engine/unified_retrieve",
    response_model=UnifiedRetrieveResponse,
    tags=["Chat Engine"],
)
async def retrieve_unified_with_gateway(request: UnifiedRetrieveRequest):
    """统一检索接口：并行查询 documents / tables / adela，并返回融合后的检索证据。"""
    logger.info(
        "Received unified retrieve request: query='%s', source_types=%s, fused_top_k=%s",
        request.query,
        request.source_types,
        request.fused_top_k,
    )
    total_start = time.perf_counter()
    timings: Dict[str, float] = {
        "retrieve_ms": 0.0,
        "fuse_ms": 0.0,
        "total_ms": 0.0,
    }
    try:
        rag_service = get_rag_service()

        source_enabled_map: Dict[Literal["document", "table", "adela", "public_cloud"], bool] = {
            "document": request.document_config.enabled,
            "table": request.table_config.enabled,
            "adela": request.adela_config.enabled,
            "public_cloud": request.public_cloud_config.enabled,
        }
        selected_sources = _resolve_selected_sources(
            source_enabled_map=source_enabled_map,
            requested_sources=request.source_types,
        )
        selected_source_set = set(selected_sources)
        skipped_reason = (
            "filtered_by_source_types"
            if request.source_types is not None
            else "not_selected"
        )

        async def _retrieve_documents() -> Tuple[List[UnifiedEvidenceItem], UnifiedSourceStatus]:
            cfg = request.document_config
            if not cfg.enabled:
                return [], _build_disabled_source_status("document")
            if "document" not in selected_source_set:
                return [], _build_skipped_source_status("document", reason=skipped_reason)
            start = time.perf_counter()
            try:
                retrieving_request = RetrievingRequest(
                    query=request.query,
                    retrieval_method=cfg.retrieval_method,
                    top_k=cfg.top_k,
                    similarity_threshold=cfg.similarity_threshold,
                    filter=cfg.filter,
                    uri=cfg.uri,
                    collection_name=cfg.collection_name,
                    embedding_models=cfg.embedding_models,
                    vector_store_configs=cfg.vector_store_configs,
                )
                chunks = await asyncio.to_thread(rag_service.retrieving, retrieving_request)
                evidences = _build_document_evidences(chunks)
                return (
                    evidences,
                    UnifiedSourceStatus(
                        source_type="document",
                        enabled=True,
                        success=True,
                        retrieve_ms=_elapsed_ms(start),
                        retrieved_count=len(chunks),
                        used_count=0,
                        message=None,
                    ),
                )
            except Exception as exc:
                logger.error(
                    "Unified document retrieval failed: query=%s",
                    request.query,
                    exc_info=True,
                )
                return (
                    [],
                    UnifiedSourceStatus(
                        source_type="document",
                        enabled=True,
                        success=False,
                        retrieve_ms=_elapsed_ms(start),
                        retrieved_count=0,
                        used_count=0,
                        message=str(exc),
                    ),
                )

        async def _retrieve_tables() -> Tuple[List[UnifiedEvidenceItem], UnifiedSourceStatus]:
            cfg = request.table_config
            if not cfg.enabled:
                return [], _build_disabled_source_status("table")
            if "table" not in selected_source_set:
                return [], _build_skipped_source_status("table", reason=skipped_reason)
            start = time.perf_counter()
            try:
                table_request = TableChatRequest(
                    query=request.query,
                    retrieval_method=cfg.retrieval_method,
                    top_k=cfg.top_k,
                    similarity_threshold=cfg.similarity_threshold,
                    data_path=cfg.data_path,
                    searchable_fields=cfg.searchable_fields,
                    return_fields=cfg.return_fields,
                    embedding_models=cfg.embedding_models,
                )
                rows = await asyncio.to_thread(rag_service.table_chat_retrieving, table_request)
                evidences = _build_structured_evidences(rows, source_type="table")
                return (
                    evidences,
                    UnifiedSourceStatus(
                        source_type="table",
                        enabled=True,
                        success=True,
                        retrieve_ms=_elapsed_ms(start),
                        retrieved_count=len(rows),
                        used_count=0,
                        message=None,
                    ),
                )
            except Exception as exc:
                return (
                    [],
                    UnifiedSourceStatus(
                        source_type="table",
                        enabled=True,
                        success=False,
                        retrieve_ms=_elapsed_ms(start),
                        retrieved_count=0,
                        used_count=0,
                        message=str(exc),
                    ),
                )

        async def _retrieve_adela() -> Tuple[List[UnifiedEvidenceItem], UnifiedSourceStatus]:
            cfg = request.adela_config
            if not cfg.enabled:
                return [], _build_disabled_source_status("adela")
            if "adela" not in selected_source_set:
                return [], _build_skipped_source_status("adela", reason=skipped_reason)
            start = time.perf_counter()
            try:
                adela_request = AdelaChatRequest(
                    query=request.query,
                    retrieval_method=cfg.retrieval_method,
                    top_k=cfg.top_k,
                    similarity_threshold=cfg.similarity_threshold,
                    data_path=cfg.data_path,
                    source_dir=cfg.source_dir,
                    searchable_fields=cfg.searchable_fields,
                    return_fields=cfg.return_fields,
                    embedding_models=cfg.embedding_models,
                )
                rows = await asyncio.to_thread(rag_service.adela_chat_retrieving, adela_request)
                evidences = _build_structured_evidences(rows, source_type="adela")
                return (
                    evidences,
                    UnifiedSourceStatus(
                        source_type="adela",
                        enabled=True,
                        success=True,
                        retrieve_ms=_elapsed_ms(start),
                        retrieved_count=len(rows),
                        used_count=0,
                        message=None,
                    ),
                )
            except Exception as exc:
                return (
                    [],
                    UnifiedSourceStatus(
                        source_type="adela",
                        enabled=True,
                        success=False,
                        retrieve_ms=_elapsed_ms(start),
                        retrieved_count=0,
                        used_count=0,
                        message=str(exc),
                    ),
                )

        async def _retrieve_public_cloud() -> Tuple[List[UnifiedEvidenceItem], UnifiedSourceStatus]:
            cfg = request.public_cloud_config
            if not cfg.enabled:
                return [], _build_disabled_source_status("public_cloud")
            if "public_cloud" not in selected_source_set:
                return [], _build_skipped_source_status("public_cloud", reason=skipped_reason)
            start = time.perf_counter()
            try:
                rows = await asyncio.to_thread(
                    _retrieve_public_cloud_rows,
                    request.query,
                    cfg.api_url,
                    cfg.api_token,
                    int(cfg.top_k or 20),
                )
                evidences = _build_structured_evidences(rows, source_type="public_cloud")
                return (
                    evidences,
                    UnifiedSourceStatus(
                        source_type="public_cloud",
                        enabled=True,
                        success=True,
                        retrieve_ms=_elapsed_ms(start),
                        retrieved_count=len(rows),
                        used_count=0,
                        message=None,
                    ),
                )
            except Exception as exc:
                return (
                    [],
                    UnifiedSourceStatus(
                        source_type="public_cloud",
                        enabled=True,
                        success=False,
                        retrieve_ms=_elapsed_ms(start),
                        retrieved_count=0,
                        used_count=0,
                        message=str(exc),
                    ),
                )

        retrieve_start = time.perf_counter()
        (
            (doc_evidences, doc_status),
            (table_evidences, table_status),
            (adela_evidences, adela_status),
            (public_cloud_evidences, public_cloud_status),
        ) = await asyncio.gather(
            _retrieve_documents(),
            _retrieve_tables(),
            _retrieve_adela(),
            _retrieve_public_cloud(),
        )
        timings["retrieve_ms"] = _elapsed_ms(retrieve_start)

        fuse_start = time.perf_counter()
        fused_top_k = max(1, int(request.fused_top_k or 12))
        rrf_k = max(1, int(request.rrf_k or 60))
        fused_evidences = _rrf_fuse_evidences(
            source_lists=[doc_evidences, table_evidences, adela_evidences, public_cloud_evidences],
            fused_top_k=fused_top_k,
            rrf_k=rrf_k,
        )
        references = _resolve_unified_references(fused_evidences)
        timings["fuse_ms"] = _elapsed_ms(fuse_start)
        timings["total_ms"] = _elapsed_ms(total_start)

        status_list = [doc_status, table_status, adela_status, public_cloud_status]
        for status in status_list:
            status.used_count = sum(
                1 for evidence in fused_evidences if evidence.source_type == status.source_type
            )

        logger.info(
            "Unified retrieve completed: selected=%s, fused=%s, retrieve_ms=%.3f, fuse_ms=%.3f, total_ms=%.3f",
            selected_sources,
            len(fused_evidences),
            timings["retrieve_ms"],
            timings["fuse_ms"],
            timings["total_ms"],
        )
        return UnifiedRetrieveResponse(
            query=request.query,
            selected_sources=selected_sources,
            fused_evidences=fused_evidences,
            reference=references,
            source_status=status_list,
            timings=UnifiedRetrieveTimings(**timings),
        )
    except ValueError as e:
        timings["total_ms"] = _elapsed_ms(total_start)
        logger.warning("Invalid unified retrieve request: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        timings["total_ms"] = _elapsed_ms(total_start)
        logger.error("Unified retrieve processing error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"统一检索处理过程中发生错误: {str(e)}")


@router.post(
    "/chat_engine/unified_query",
    response_model=UnifiedQueryResponse,
    tags=["Chat Engine"],
)
async def query_unified_with_gateway(request: UnifiedQueryRequest):
    """统一检索网关接口：并行查询 documents / tables / adela，并进行 RRF 融合。"""
    logger.info(
        "Received unified query request: query='%s', fused_top_k=%s",
        request.query,
        request.fused_top_k,
    )
    total_start = time.perf_counter()
    timings: Dict[str, float] = {
        "route_ms": 0.0,
        "retrieve_ms": 0.0,
        "fuse_ms": 0.0,
        "answer_ms": 0.0,
        "total_ms": 0.0,
    }
    try:
        rag_service = get_rag_service()

        source_enabled_map: Dict[Literal["document", "table", "adela", "public_cloud"], bool] = {
            "document": request.document_config.enabled,
            "table": request.table_config.enabled,
            "adela": request.adela_config.enabled,
            "public_cloud": request.public_cloud_config.enabled,
        }
        enabled_sources: List[Literal["document", "table", "adela", "public_cloud"]] = [
            source_type
            for source_type, enabled in source_enabled_map.items()
            if enabled
        ]

        route_start = time.perf_counter()
        route_reason = "router_disabled"
        fallback_used = False
        selected_sources = enabled_sources
        route_task = None
        if request.route_with_llm:
            route_task = asyncio.create_task(
                route_unified_sources(
                    query=request.query,
                    enabled_sources=enabled_sources,
                    llm_config=request.llm_config,
                )
            )
        else:
            route_reason = "router disabled by request, use all enabled sources"

        aggregate_sources: List[Literal["table", "adela", "public_cloud"]] = [
            source_type
            for source_type in ("table", "adela", "public_cloud")
            if source_enabled_map[source_type]
        ]
        if aggregate_sources:
            retrieve_start = time.perf_counter()
            aggregate_result = await _try_answer_structured_aggregate_question(
                query=request.query,
                data_paths={
                    "table": request.table_config.data_path,
                    "adela": request.adela_config.data_path,
                    "public_cloud": request.public_cloud_config.api_url,
                },
                enabled_sources=aggregate_sources,
                rag_service=rag_service,
                llm_config=request.llm_config,
                source_dirs={
                    "adela": request.adela_config.source_dir,
                    "public_cloud_token": request.public_cloud_config.api_token,
                },
            )
            if aggregate_result is not None:
                answer, aggregate_row = aggregate_result
                timings["retrieve_ms"] = _elapsed_ms(retrieve_start)

                fuse_start = time.perf_counter()
                aggregate_source_type = aggregate_row.entity.get("source_type") or "table"
                fused_evidences = _build_structured_evidences(
                    [aggregate_row],
                    source_type=aggregate_source_type,
                )
                for evidence in fused_evidences:
                    evidence.score = 1.0
                timings["fuse_ms"] = _elapsed_ms(fuse_start)
                timings["answer_ms"] = 0.0
                timings["total_ms"] = _elapsed_ms(total_start)

                source_status = []
                for source_type in ("document", "table", "adela", "public_cloud"):
                    if source_type == aggregate_source_type:
                        source_status.append(
                            UnifiedSourceStatus(
                                source_type=aggregate_source_type,
                                enabled=True,
                                success=True,
                                retrieve_ms=timings["retrieve_ms"],
                                retrieved_count=int(aggregate_row.entity.get("filtered_record_count") or 0),
                                used_count=1,
                                message="answered_by_agentic_aggregate",
                            )
                        )
                    elif source_enabled_map[source_type]:
                        source_status.append(
                            _build_skipped_source_status(
                                source_type,
                                reason="skipped_by_agentic_aggregate",
                            )
                        )
                    else:
                        source_status.append(_build_disabled_source_status(source_type))

                route_plan = UnifiedRoutePlan(
                    route_with_llm=request.route_with_llm,
                    selected_sources=[aggregate_source_type],
                    skipped_sources=[
                        source_type
                        for source_type in ("document", "table", "adela", "public_cloud")
                        if source_type != aggregate_source_type
                    ],
                    fallback_used=False,
                    reason=str(aggregate_row.entity.get("planner_reason") or "agentic aggregate plan"),
                )

                response = UnifiedQueryResponse(
                    query=request.query,
                    fused_evidences=fused_evidences,
                    reference=[],
                    source_status=source_status,
                    route_plan=route_plan,
                    answer=answer,
                    timings=UnifiedQueryTimings(**timings),
                    message="answered_by_agentic_aggregate",
                )

                logger.info(
                    "Unified aggregate query completed: retrieve_ms=%.3f, fuse_ms=%.3f, total_ms=%.3f",
                    timings["retrieve_ms"],
                    timings["fuse_ms"],
                    timings["total_ms"],
                )

                if route_task is not None and not route_task.done():
                    route_task.cancel()
                    try:
                        await route_task
                    except asyncio.CancelledError:
                        pass

                if request.stream:
                    async def generate_aggregate():
                        payload = _model_dump_compatible(response)
                        yield f"data: {json.dumps({'content': answer}, ensure_ascii=False)}\n\n"
                        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                        yield "data: [DONE]\n\n"

                    return StreamingResponse(
                        generate_aggregate(),
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                            "X-Accel-Buffering": "no",
                        },
                    )

                return response

        if route_task is not None:
            selected_sources, fallback_used, route_reason = await route_task
        selected_source_set = set(selected_sources)
        timings["route_ms"] = _elapsed_ms(route_start)

        async def _retrieve_documents() -> Tuple[List[UnifiedEvidenceItem], UnifiedSourceStatus]:
            cfg = request.document_config
            if not cfg.enabled:
                return [], _build_disabled_source_status("document")
            if "document" not in selected_source_set:
                return [], _build_skipped_source_status("document")
            start = time.perf_counter()
            try:
                retrieving_request = RetrievingRequest(
                    query=request.query,
                    retrieval_method=cfg.retrieval_method,
                    top_k=cfg.top_k,
                    similarity_threshold=cfg.similarity_threshold,
                    filter=cfg.filter,
                    uri=cfg.uri,
                    collection_name=cfg.collection_name,
                    embedding_models=cfg.embedding_models,
                    vector_store_configs=cfg.vector_store_configs,
                )
                chunks = await asyncio.to_thread(rag_service.retrieving, retrieving_request)
                evidences = _build_document_evidences(chunks)
                return (
                    evidences,
                    UnifiedSourceStatus(
                        source_type="document",
                        enabled=True,
                        success=True,
                        retrieve_ms=_elapsed_ms(start),
                        retrieved_count=len(chunks),
                        used_count=0,
                        message=None,
                    ),
                )
            except Exception as exc:
                return (
                    [],
                    UnifiedSourceStatus(
                        source_type="document",
                        enabled=True,
                        success=False,
                        retrieve_ms=_elapsed_ms(start),
                        retrieved_count=0,
                        used_count=0,
                        message=str(exc),
                    ),
                )

        async def _retrieve_tables() -> Tuple[List[UnifiedEvidenceItem], UnifiedSourceStatus]:
            cfg = request.table_config
            if not cfg.enabled:
                return [], _build_disabled_source_status("table")
            if "table" not in selected_source_set:
                return [], _build_skipped_source_status("table")
            start = time.perf_counter()
            try:
                table_request = TableChatRequest(
                    query=request.query,
                    retrieval_method=cfg.retrieval_method,
                    top_k=cfg.top_k,
                    similarity_threshold=cfg.similarity_threshold,
                    data_path=cfg.data_path,
                    searchable_fields=cfg.searchable_fields,
                    return_fields=cfg.return_fields,
                    embedding_models=cfg.embedding_models,
                )
                rows = await asyncio.to_thread(rag_service.table_chat_retrieving, table_request)
                evidences = _build_structured_evidences(rows, source_type="table")
                return (
                    evidences,
                    UnifiedSourceStatus(
                        source_type="table",
                        enabled=True,
                        success=True,
                        retrieve_ms=_elapsed_ms(start),
                        retrieved_count=len(rows),
                        used_count=0,
                        message=None,
                    ),
                )
            except Exception as exc:
                return (
                    [],
                    UnifiedSourceStatus(
                        source_type="table",
                        enabled=True,
                        success=False,
                        retrieve_ms=_elapsed_ms(start),
                        retrieved_count=0,
                        used_count=0,
                        message=str(exc),
                    ),
                )

        async def _retrieve_adela() -> Tuple[List[UnifiedEvidenceItem], UnifiedSourceStatus]:
            cfg = request.adela_config
            if not cfg.enabled:
                return [], _build_disabled_source_status("adela")
            if "adela" not in selected_source_set:
                return [], _build_skipped_source_status("adela")
            start = time.perf_counter()
            try:
                adela_request = AdelaChatRequest(
                    query=request.query,
                    retrieval_method=cfg.retrieval_method,
                    top_k=cfg.top_k,
                    similarity_threshold=cfg.similarity_threshold,
                    data_path=cfg.data_path,
                    source_dir=cfg.source_dir,
                    searchable_fields=cfg.searchable_fields,
                    return_fields=cfg.return_fields,
                    embedding_models=cfg.embedding_models,
                )
                rows = await asyncio.to_thread(rag_service.adela_chat_retrieving, adela_request)
                evidences = _build_structured_evidences(rows, source_type="adela")
                return (
                    evidences,
                    UnifiedSourceStatus(
                        source_type="adela",
                        enabled=True,
                        success=True,
                        retrieve_ms=_elapsed_ms(start),
                        retrieved_count=len(rows),
                        used_count=0,
                        message=None,
                    ),
                )
            except Exception as exc:
                return (
                    [],
                    UnifiedSourceStatus(
                        source_type="adela",
                        enabled=True,
                        success=False,
                        retrieve_ms=_elapsed_ms(start),
                        retrieved_count=0,
                        used_count=0,
                        message=str(exc),
                    ),
                )

        async def _retrieve_public_cloud() -> Tuple[List[UnifiedEvidenceItem], UnifiedSourceStatus]:
            cfg = request.public_cloud_config
            if not cfg.enabled:
                return [], _build_disabled_source_status("public_cloud")
            if "public_cloud" not in selected_source_set:
                return [], _build_skipped_source_status("public_cloud")
            start = time.perf_counter()
            try:
                rows = await asyncio.to_thread(
                    _retrieve_public_cloud_rows,
                    request.query,
                    cfg.api_url,
                    cfg.api_token,
                    int(cfg.top_k or 20),
                )
                evidences = _build_structured_evidences(rows, source_type="public_cloud")
                return (
                    evidences,
                    UnifiedSourceStatus(
                        source_type="public_cloud",
                        enabled=True,
                        success=True,
                        retrieve_ms=_elapsed_ms(start),
                        retrieved_count=len(rows),
                        used_count=0,
                        message=None,
                    ),
                )
            except Exception as exc:
                return (
                    [],
                    UnifiedSourceStatus(
                        source_type="public_cloud",
                        enabled=True,
                        success=False,
                        retrieve_ms=_elapsed_ms(start),
                        retrieved_count=0,
                        used_count=0,
                        message=str(exc),
                    ),
                )

        retrieve_start = time.perf_counter()
        (
            (doc_evidences, doc_status),
            (table_evidences, table_status),
            (adela_evidences, adela_status),
            (public_cloud_evidences, public_cloud_status),
        ) = await asyncio.gather(
            _retrieve_documents(),
            _retrieve_tables(),
            _retrieve_adela(),
            _retrieve_public_cloud(),
        )
        timings["retrieve_ms"] = _elapsed_ms(retrieve_start)

        fuse_start = time.perf_counter()
        fused_top_k = max(1, int(request.fused_top_k or 12))
        rrf_k = max(1, int(request.rrf_k or 60))
        fused_evidences = _rrf_fuse_evidences(
            source_lists=[doc_evidences, table_evidences, adela_evidences, public_cloud_evidences],
            fused_top_k=fused_top_k,
            rrf_k=rrf_k,
        )
        references = _resolve_unified_references(fused_evidences)
        timings["fuse_ms"] = _elapsed_ms(fuse_start)

        status_list = [doc_status, table_status, adela_status, public_cloud_status]
        for status in status_list:
            status.used_count = sum(
                1 for evidence in fused_evidences if evidence.source_type == status.source_type
            )
        route_plan = UnifiedRoutePlan(
            route_with_llm=request.route_with_llm,
            selected_sources=selected_sources,
            skipped_sources=[
                source_type
                for source_type in ("document", "table", "adela", "public_cloud")
                if source_type not in selected_source_set
            ],
            fallback_used=fallback_used,
            reason=route_reason,
        )

        if request.stream:
            async def generate():
                answer_start = time.perf_counter()
                output_parts: List[str] = []
                try:
                    async for chunk in answer_unified_question_stream(
                        query=request.query,
                        fused_evidences=fused_evidences,
                        llm_config=request.llm_config,
                    ):
                        if chunk:
                            output_parts.append(chunk)
                            yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"

                    answer = "".join(output_parts)
                    timings["answer_ms"] = _elapsed_ms(answer_start)
                    timings["total_ms"] = _elapsed_ms(total_start)
                    final_output = {
                        "content": "",
                        "query": request.query,
                        "fused_evidences": [_model_dump_compatible(item) for item in fused_evidences],
                        "reference": [_model_dump_compatible(item) for item in references],
                        "source_status": [_model_dump_compatible(item) for item in status_list],
                        "route_plan": _model_dump_compatible(route_plan),
                        "answer": answer,
                        "timings": _model_dump_compatible(UnifiedQueryTimings(**timings)),
                        "success": True,
                        "message": None,
                    }
                    yield f"data: {json.dumps(final_output, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                except asyncio.CancelledError:
                    logger.info("Unified query stream cancelled by client: query='%s'", request.query)
                    raise
                except Exception as exc:
                    if not output_parts:
                        logger.warning(
                            "Unified query stream failed before first chunk, falling back to non-stream answer: query='%s', error=%s",
                            request.query,
                            exc,
                            exc_info=True,
                        )
                        try:
                            fallback_answer = await answer_unified_question(
                                query=request.query,
                                fused_evidences=fused_evidences,
                                llm_config=request.llm_config,
                            )
                            if fallback_answer:
                                output_parts.append(fallback_answer)
                                yield f"data: {json.dumps({'content': fallback_answer}, ensure_ascii=False)}\n\n"

                            timings["answer_ms"] = _elapsed_ms(answer_start)
                            timings["total_ms"] = _elapsed_ms(total_start)
                            final_output = {
                                "content": "",
                                "query": request.query,
                                "fused_evidences": [_model_dump_compatible(item) for item in fused_evidences],
                                "reference": [_model_dump_compatible(item) for item in references],
                                "source_status": [_model_dump_compatible(item) for item in status_list],
                                "route_plan": _model_dump_compatible(route_plan),
                                "answer": fallback_answer,
                                "timings": _model_dump_compatible(UnifiedQueryTimings(**timings)),
                                "success": True,
                                "message": "stream answer failed before first chunk, fell back to non-stream answer",
                            }
                            yield f"data: {json.dumps(final_output, ensure_ascii=False)}\n\n"
                            yield "data: [DONE]\n\n"
                            return
                        except Exception as fallback_exc:
                            exc = fallback_exc

                    answer = "".join(output_parts)
                    timings["answer_ms"] = _elapsed_ms(answer_start)
                    timings["total_ms"] = _elapsed_ms(total_start)
                    message = f"RAG 流式回答生成失败: {str(exc)}"
                    logger.error(
                        "Unified query stream generation error: query='%s', partial_answer_chars=%s, error=%s",
                        request.query,
                        len(answer),
                        exc,
                        exc_info=True,
                    )
                    final_output = {
                        "content": "",
                        "query": request.query,
                        "fused_evidences": [_model_dump_compatible(item) for item in fused_evidences],
                        "reference": [_model_dump_compatible(item) for item in references],
                        "source_status": [_model_dump_compatible(item) for item in status_list],
                        "route_plan": _model_dump_compatible(route_plan),
                        "answer": answer,
                        "timings": _model_dump_compatible(UnifiedQueryTimings(**timings)),
                        "success": False,
                        "message": message,
                    }
                    yield f"data: {json.dumps(final_output, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        answer_start = time.perf_counter()
        answer = await answer_unified_question(
            query=request.query,
            fused_evidences=fused_evidences,
            llm_config=request.llm_config,
        )
        timings["answer_ms"] = _elapsed_ms(answer_start)
        timings["total_ms"] = _elapsed_ms(total_start)

        logger.info(
            "Unified query completed: fused=%s, retrieve_ms=%.3f, fuse_ms=%.3f, answer_ms=%.3f, total_ms=%.3f",
            len(fused_evidences),
            timings["retrieve_ms"],
            timings["fuse_ms"],
            timings["answer_ms"],
            timings["total_ms"],
        )
        return UnifiedQueryResponse(
            query=request.query,
            fused_evidences=fused_evidences,
            reference=references,
            source_status=status_list,
            route_plan=route_plan,
            answer=answer,
            timings=UnifiedQueryTimings(**timings),
        )
    except ValueError as e:
        timings["total_ms"] = _elapsed_ms(total_start)
        logger.warning("Invalid unified query request: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        timings["total_ms"] = _elapsed_ms(total_start)
        logger.error("Unified query processing error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"统一检索问答处理过程中发生错误: {str(e)}")


@router.post(
    "/embedding",
    response_model=EmbeddingList,
    tags=["Embedding"],
)
async def process_embedding(request: EmbeddingRequest):
    """
    处理文本向量化请求
    """
    logger.info(f"Received embedding request: model={request.model}")
    try:
        rag_service = get_rag_service()
        start_time = time.time()

        # 检查模型是否存在
        if request.model not in rag_service.embedding_models:
            logger.warning(f"Embedding model not found: {request.model}")
            raise HTTPException(status_code=400, detail=f"模型 '{request.model}' 不存在")

        # 处理输入文本
        inputs = request.input if isinstance(request.input, list) else [request.input]

        # 计算token数量（简化版本，实际应使用模型的tokenizer）
        prompt_tokens = 0

        # 获取向量嵌入
        embeddings = rag_service.get_text_embedding_batch(inputs, request.model)
        embeddings = [Embedding(index=idx, embedding=embedding, object="embedding") for idx, embedding in enumerate(embeddings)]
        # embeddings = []
        # for idx, text in enumerate(inputs):
        #     embedding = rag_service.get_text_embedding(text, request.model)
        #     embeddings.append(Embedding(index=idx, embedding=embedding, object="embedding"))

        logger.info(f"Embedding generation successful: model={request.model}, time={time.time() - start_time:.2f}s")

        # 构建响应
        return EmbeddingList(
            object="list",
            data=embeddings,
            model=request.model,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                total_tokens=prompt_tokens
            )
        )

    except ValueError as e:
        logger.warning(f"Invalid embedding request: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Embedding processing error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"向量化处理过程中发生错误: {str(e)}")
