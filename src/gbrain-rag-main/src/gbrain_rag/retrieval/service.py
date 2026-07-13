from __future__ import annotations

import re
import time
from typing import Any
import logging

import numpy as np

from gbrain_rag.core.config import Settings, get_settings
from gbrain_rag.core.types import Chunk, RoutePlan, ScoredChunk
from gbrain_rag.retrieval.embeddings import EmbeddingManager, get_embedding_manager
from gbrain_rag.retrieval.aspects import (
    ACCURACY_ASPECT,
    DEPLOYMENT_ASPECT,
    GENERAL_ASPECT,
    INPUT_OUTPUT_ASPECT,
    LABEL_ASPECT,
    LIMITATION_ASPECT,
    MODEL_ARTIFACT_ASPECT,
    PERFORMANCE_ASPECT,
    RELEASE_CHANGE_ASPECT,
    answerability_score,
    classify_chunk_aspects,
    chunk_section_type,
)
from gbrain_rag.retrieval.entities import entity_names, entity_overlap_score
from gbrain_rag.retrieval.query_understanding import (
    build_query_intent,
    expand_query_terms,
    semantic_query_terms,
    score_structured_row,
)
from gbrain_rag.retrieval.ranking import bm25_scores, reciprocal_rank_fusion
from gbrain_rag.retrieval.store import BrainStore


logger = logging.getLogger(__name__)


REQUEST_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "model_family": ("模型族",),
    "model_name": ("模型名称", "model_name", "模型文件", "模型"),
    "component_type": ("组件类型", "component_type", "组件"),
    "oid": ("OID", "oid"),
    "platform": ("平台", "platform", "推荐配置", "supported_device", "支持设备"),
    "feature_dim": ("特征维度", "维度", "feature_dim"),
    "owner": ("负责人", "owner"),
    "recommended_config": ("推荐配置", "recommended_config"),
    "supported_device": ("支持设备", "supported_device"),
    "last_updated": ("更新时间", "最近更新时间", "last_updated"),
    "did": ("did", "DID"),
    "rid": ("rid", "RID"),
}

TABLE_FIELD_LABELS: dict[str, str] = {
    "model_family": "模型族",
    "model_name": "模型名称",
    "component_type": "组件类型",
    "oid": "OID",
    "platform": "平台",
    "feature_dim": "特征维度",
    "owner": "负责人",
    "recommended_config": "推荐配置",
    "supported_device": "支持设备",
    "last_updated": "最近更新时间",
    "did": "did",
    "rid": "rid",
}


def _requested_fields(query: str) -> list[str]:
    normalized = str(query or "").lower()
    fields = []
    for field, aliases in REQUEST_FIELD_ALIASES.items():
        if any(alias.lower() in normalized for alias in aliases):
            fields.append(field)
    if not fields and any(phrase in normalized for phrase in ("是多少", "是什么", "是谁", "哪个", "哪一个", "有哪些")):
        fields.extend(["model_name", "oid", "platform", "feature_dim", "owner"])
    return list(dict.fromkeys(fields))


def _parse_structured_table_rows(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = [part.strip() for part in line.split(":", 1)]
        field = None
        for candidate, label in TABLE_FIELD_LABELS.items():
            if key == label:
                field = candidate
                break
        if field is None:
            continue
        if field == "model_family" and current and any(name in current for name in ("model_name", "oid", "feature_dim")):
            rows.append(current)
            current = {}
        elif field == "model_name" and current.get("model_name"):
            rows.append(current)
            current = {"model_family": current.get("model_family", "")}
        current[field] = value
    if current:
        rows.append(current)
    return [row for row in rows if any(value for value in row.values())]


def _field_summary_from_rows(query: str, rows: list[dict[str, str]], max_rows: int = 12) -> str | None:
    requested = _requested_fields(query)
    if not requested:
        return None
    display_fields = list(dict.fromkeys(["model_family", "model_name", *requested]))
    summaries = []
    for row in rows:
        parts = []
        for field in display_fields:
            value = row.get(field)
            if not value:
                continue
            label = TABLE_FIELD_LABELS.get(field, field)
            if field == "model_name" and len(value) > 120:
                value = value[:117] + "..."
            elif len(value) > 180:
                value = value[:177] + "..."
            parts.append(f"{label}={value}")
        if parts:
            summaries.append(" | ".join(parts))
        if len(summaries) >= max_rows:
            break
    if not summaries:
        return None
    return "字段抽取:\n" + "\n".join(f"- {summary}" for summary in summaries)


def _lexical_boost(query: str, chunk: Chunk) -> float:
    q = query.lower()
    text = "\n".join([chunk.doc_name, chunk.title or "", chunk.text, chunk.index_text]).lower()
    boost = 0.0
    if "特征维度" in query and "特征维度" in text:
        boost += 0.32
    if "模型文件列表" in text and ("特征维度" in query or "oid" in q or "平台" in query):
        boost += 0.12
    if chunk.source_type == "table" and "特征维度" in query and "特征维度" not in text:
        boost -= 0.18
    return boost


ENTITY_TERM_STOPWORDS = {
    "模型",
    "算法",
    "检测",
    "识别",
    "分类",
    "属性",
    "精度",
    "指标",
    "如何",
    "怎样",
    "什么",
    "哪些",
    "情况",
    "无法",
    "保证",
    "输入",
    "输出",
    "版本",
    "当前",
    "测试",
    "结果",
}


def _entity_constraint_terms(intent: Any) -> tuple[str, ...]:
    frame = getattr(intent, "query_frame", {}) or {}
    candidates: list[str] = []
    candidates.extend(str(term) for term in frame.get("normalized_entities") or [])
    for mention in frame.get("entity_mentions") or []:
        text = str(mention or "").strip()
        if not text:
            continue
        candidates.append(text)
        stripped = re.sub(r"(检测|识别|分类|属性|特征)?(模型|算法)?$", "", text).strip()
        if stripped and stripped != text:
            candidates.append(stripped)
    candidates.extend(str(term) for term in getattr(intent, "target_terms", ()) or ())
    candidates.extend(str(term) for term in getattr(intent, "algorithm_terms", ()) or ())
    candidates.extend(str(term) for term in getattr(intent, "model_terms", ()) or ())
    if getattr(intent, "exact_model", None):
        candidates.append(str(intent.exact_model))
    for term in getattr(intent, "semantic_terms", ()) or ():
        value = str(term or "").strip().lower()
        if re.search(r"[A-Za-z0-9_./+-]", value) and len(value) >= 3 and value not in ENTITY_TERM_STOPWORDS:
            candidates.append(value)

    cleaned: list[str] = []
    seen: set[str] = set()
    for term in candidates:
        value = str(term or "").strip().lower()
        if not value:
            continue
        compact = re.sub(r"\s+", "", value)
        if len(compact) < 3 or compact in ENTITY_TERM_STOPWORDS:
            continue
        if compact in seen:
            continue
        seen.add(compact)
        cleaned.append(compact)
    return tuple(cleaned[:24])


def _entity_match_score(intent: Any, chunk: Chunk) -> float:
    terms = _entity_constraint_terms(intent)
    if not terms:
        return 0.0
    text = "\n".join(
        [
            chunk.doc_name,
            chunk.title or "",
            chunk.text,
            chunk.index_text,
            "\n".join(str(value) for value in (chunk.metadata or {}).values() if value not in (None, "", [], {})),
        ]
    ).lower()
    compact_text = re.sub(r"\s+", "", text)
    score = 0.0
    for term in terms:
        if term in text or term in compact_text:
            if "_" in term or "." in term or "-" in term:
                score += 1.0
            elif len(term) >= 5:
                score += 0.75
            else:
                score += 0.45
    return min(score, 2.0)


def _aspect_lexical_boost(query: str, chunk: Chunk, intent: Any) -> float:
    """Boost chunks that can directly answer the query aspect.

    This is a generic answerability signal. It distinguishes metric chunks from
    boundary/label/artifact chunks for any model entity, without naming a
    specific document or product.
    """

    query_aspect = getattr(intent, "aspect", GENERAL_ASPECT)
    if not query_aspect or query_aspect == GENERAL_ASPECT:
        return 0.0
    answerability, _metadata = answerability_score(query_aspect, chunk)
    entity_terms = _entity_constraint_terms(intent)
    entity_score = _entity_match_score(intent, chunk)
    if entity_terms and entity_score <= 0:
        boost = answerability * 0.02 - 0.65
    else:
        boost = answerability * 0.38 + min(entity_score, 1.0) * 0.18

    text = "\n".join([chunk.doc_name, chunk.title or "", chunk.text, chunk.index_text]).lower()
    if query_aspect == ACCURACY_ASPECT:
        if any(token in text for token in ("acc", "precision", "recall", "f1", "map", "top1", "m/prec", "mprec", "ma")):
            boost += 0.1
        if "无法保证" in text and not any(token in text for token in ("acc", "precision", "recall", "f1", "map", "top1")):
            boost -= 0.24
    elif query_aspect == LIMITATION_ASPECT:
        if any(token in text for token in ("无法保证", "算法边界", "前提条件", "场景要求", "目标要求")):
            boost += 0.12
    elif query_aspect == MODEL_ARTIFACT_ASPECT and "模型文件" in text:
        boost += 0.08
    elif query_aspect == LABEL_ASPECT and any(token in text for token in ("标签解释", "label", "类别标签")):
        boost += 0.08
    return boost


def _structured_lexical_boost(query: str, chunk: Chunk) -> float:
    """Small deterministic boost for exact field overlap in table/adela rows."""

    if chunk.source_type not in {"table", "adela"}:
        return 0.0
    terms = [term for term in semantic_query_terms(query, max_terms=36) if term not in {"t4", "p4", "l4"}]
    if not terms:
        return 0.0
    metadata_text = "\n".join(
        str(chunk.metadata.get(key) or "")
        for key in (
            "target_name",
            "algorithm_type",
            "algorithm_name",
            "application_scene",
            "model_name",
            "name",
            "label_list",
            "labels",
            "platform",
            "supported_device",
            "recommended_config",
        )
    ).lower()
    if not metadata_text.strip():
        return 0.0
    hits = [term for term in terms if term and term.lower() in metadata_text]
    if not hits:
        return 0.0
    boost = min(0.18, 0.035 * len(hits))
    platform = "\n".join(
        str(chunk.metadata.get(key) or "")
        for key in ("platform", "supported_device", "recommended_config", "model_name")
    )
    if any(token in query.lower() for token in ("t4", "p4", "l4")) and any(token in platform.lower() for token in ("t4", "p4", "l4")):
        boost += 0.04
    return boost


def _extract_field_summary(query: str, chunk: Chunk) -> str | None:
    requested = _requested_fields(query)
    if query and not requested:
        return None
    text = "\n".join([chunk.text, chunk.index_text])
    if "表格结构化行" in text or "模型族:" in text or "模型名称:" in text:
        rows = _parse_structured_table_rows(text)
        summary = _field_summary_from_rows(query, rows)
        if summary:
            return summary

    if chunk.source_type in {"table", "adela"}:
        parts = []
        for field in requested:
            aliases = REQUEST_FIELD_ALIASES.get(field, (field,))
            for key in aliases:
                value = chunk.metadata.get(key)
                if value not in (None, "", [], {}):
                    parts.append(f"{TABLE_FIELD_LABELS.get(field, field)}={str(value)}")
                    break
        if parts:
            return "字段抽取:\n- " + " | ".join(dict.fromkeys(parts))
    return None


CANONICAL_METADATA_FIELDS: dict[str, tuple[str, ...]] = {
    "row_id": ("row_id",),
    "target_name": ("target_name", "目标名称"),
    "algorithm_type": ("algorithm_type", "算法类型"),
    "algorithm_name": ("algorithm_name", "算法名称"),
    "application_scene": ("application_scene", "应用场景"),
    "owner": ("owner", "负责人(人员)", "负责人"),
    "model_name": ("model_name", "模型名称", "name"),
    "oid": ("oid", "OID"),
    "supported_device": ("supported_device", "支持设备"),
    "recommended_config": ("recommended_config", "推荐配置"),
    "last_updated": ("last_updated", "最近更新时间"),
    "last_updated_month": ("last_updated_month", "最近更新时间-提取年月"),
    "did": ("did",),
    "rid": ("rid",),
    "platform": ("platform",),
    "status": ("status",),
    "version": ("version",),
    "label_list": ("label_list", "labels"),
    "sheet_name": ("sheet_name",),
    "source_row_number": ("source_row_number",),
}


def _metadata_value(metadata: dict[str, Any], field: str) -> str:
    for key in CANONICAL_METADATA_FIELDS.get(field, (field,)):
        value = metadata.get(key)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, list):
            return " ".join(str(item) for item in value if item not in (None, ""))
        return str(value)
    return ""


def _legacy_table_row_id(metadata: dict[str, Any]) -> str:
    row_id = _metadata_value(metadata, "row_id")
    if row_id:
        return row_id
    sheet_name = _metadata_value(metadata, "sheet_name")
    source_row = _metadata_value(metadata, "source_row_number")
    if sheet_name and source_row:
        try:
            row_number = int(float(source_row))
            return f"{sheet_name}-{row_number:04d}"
        except ValueError:
            return f"{sheet_name}-{source_row}"
    return ""


def _legacy_evidence_id(chunk: Chunk) -> str | None:
    metadata = chunk.metadata
    if chunk.source_type == "table":
        row_id = _legacy_table_row_id(metadata)
        if row_id:
            return f"table::{row_id}"
    if chunk.source_type == "adela":
        row_id = _metadata_value(metadata, "row_id")
        if not row_id:
            model_name = _metadata_value(metadata, "model_name")
            platform = _metadata_value(metadata, "platform")
            rid = _metadata_value(metadata, "rid")
            did = _metadata_value(metadata, "did")
            if model_name and platform and rid and did:
                row_id = f"{model_name}-{platform}_{rid}_{did}"
        if row_id:
            return f"adela::{row_id}"
    if chunk.source_type == "document" and chunk.chunk_id.startswith("doc__"):
        return f"document::{chunk.chunk_id}"
    return None


def _canonical_metadata(metadata: dict[str, Any], source_type: str) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field in CANONICAL_METADATA_FIELDS:
        value = _metadata_value(metadata, field)
        if value:
            payload[field] = value
    if source_type == "table":
        row_id = _legacy_table_row_id(metadata)
        if row_id:
            payload["row_id"] = row_id
    elif source_type == "adela":
        row_id = payload.get("row_id")
        if not row_id and payload.get("model_name") and payload.get("platform") and payload.get("rid") and payload.get("did"):
            payload["row_id"] = f"{payload['model_name']}-{payload['platform']}_{payload['rid']}_{payload['did']}"
    return payload


class RetrievalService:
    def __init__(
        self,
        store: BrainStore | None = None,
        embeddings: EmbeddingManager | None = None,
        settings: Settings | None = None,
    ):
        self.settings = settings or get_settings()
        self.store = store or BrainStore(self.settings.INDEX_DB_PATH)
        self.embeddings = embeddings or get_embedding_manager()

    def route_sources(self, query: str, enabled_sources: list[str] | None = None) -> RoutePlan:
        enabled = set(enabled_sources or ["document", "table", "adela"])
        selected = [source for source in ["document", "table", "adela"] if source in enabled]

        return RoutePlan(
            document="document" in selected,
            table="table" in selected,
            adela="adela" in selected,
            reason="fallback broad retrieval across all enabled sources",
        )

    def retrieve(
        self,
        *,
        query: str,
        source_types: list[str] | None = None,
        retrieval_method: str = "hybrid",
        top_k: int | None = None,
        candidate_limit: int | None = None,
        embedding_model: str | None = None,
        embedding_models: list[str] | None = None,
        embedding_backend: str | None = None,
        similarity_threshold: float | None = None,
        query_expansion_terms: list[str] | None = None,
    ) -> tuple[list[ScoredChunk], dict[str, float]]:
        start = time.perf_counter()
        method = retrieval_method or self.settings.DEFAULT_RETRIEVAL_METHOD
        top = top_k or self.settings.DEFAULT_TOP_K
        limit = max(candidate_limit or self.settings.DEFAULT_CANDIDATE_LIMIT, top)
        source_filter = source_types or None
        intent = build_query_intent(query, extra_terms=query_expansion_terms)
        expanded_query = "\n".join(expand_query_terms(query, extra_terms=query_expansion_terms))

        chunks = self.store.load_chunks(source_filter)
        keyword_ranked: list[tuple[str, float]] = []
        vector_ranked: list[tuple[str, float]] = []
        structured_ranked: list[tuple[str, float]] = []

        keyword_start = time.perf_counter()
        if method in {"keyword", "bm25", "hybrid"}:
            # Try SQLite FTS first, then merge with a Python BM25 scan that is
            # Chinese/model-name friendly.
            fts_ranked = self.store.fts_search(expanded_query, source_types=source_filter, limit=limit)
            fts_ids = [chunk_id for chunk_id, _ in fts_ranked]
            bm25_scope = fts_ids if len(fts_ids) >= top else list(chunks.keys())
            docs = {
                chunk_id: "\n".join(
                    [
                        chunks[chunk_id].doc_name,
                        chunks[chunk_id].title or "",
                        chunks[chunk_id].text,
                        chunks[chunk_id].index_text,
                    ]
                )
                for chunk_id in bm25_scope
                if chunk_id in chunks
            }
            bm25 = bm25_scores(expanded_query, docs)
            merged: dict[str, float] = {}
            for chunk_id, score in fts_ranked:
                merged[chunk_id] = max(merged.get(chunk_id, 0.0), score)
            for chunk_id, score in bm25.items():
                merged[chunk_id] = max(merged.get(chunk_id, 0.0), score)
            if merged:
                max_score = max(merged.values()) or 1.0
                keyword_ranked = sorted(
                    ((chunk_id, score / max_score) for chunk_id, score in merged.items()),
                    key=lambda item: item[1],
                    reverse=True,
                )[:limit]
        keyword_ms = (time.perf_counter() - keyword_start) * 1000

        structured_start = time.perf_counter()
        if method in {"keyword", "bm25", "hybrid"} and intent.structured:
            raw_structured = {
                chunk_id: score_structured_row(chunk.metadata, intent, chunk.source_type)
                for chunk_id, chunk in chunks.items()
                if chunk.source_type in {"table", "adela"}
            }
            raw_structured = {chunk_id: score for chunk_id, score in raw_structured.items() if score > 0}
            if raw_structured:
                max_score = max(raw_structured.values()) or 1.0
                structured_ranked = sorted(
                    ((chunk_id, score / max_score) for chunk_id, score in raw_structured.items()),
                    key=lambda item: item[1],
                    reverse=True,
                )[:limit]
        structured_ms = (time.perf_counter() - structured_start) * 1000

        vector_start = time.perf_counter()
        vector_errors: list[str] = []
        if method in {"vector", "hybrid"}:
            models = embedding_models or [embedding_model or self.settings.EMBEDDING_MODEL]
            vector_ranked_lists: list[list[tuple[str, float]]] = []
            for model in dict.fromkeys(models):
                try:
                    chunk_ids, matrix = self.store.load_embedding_matrix(
                        model_name=model,
                        source_types=source_filter,
                    )
                    if not chunk_ids or not matrix.size:
                        continue
                    backend = self.embeddings.get(model, embedding_backend)
                    query_vector = backend.encode_one(query)
                    if query_vector.shape[0] != matrix.shape[1]:
                        continue
                    scores = matrix @ query_vector
                    order = np.argsort(-scores)[:limit]
                    vector_ranked_lists.append([(chunk_ids[idx], float(scores[idx])) for idx in order])
                except Exception as exc:
                    message = f"{model}: {exc}"
                    vector_errors.append(message)
                    logger.warning("Vector retrieval skipped for %s: %s", model, exc)
                    continue
            if len(vector_ranked_lists) == 1:
                vector_ranked = vector_ranked_lists[0]
            elif vector_ranked_lists:
                fused_vectors = reciprocal_rank_fusion(vector_ranked_lists, k=self.settings.RRF_K)
                vector_ranked = sorted(fused_vectors.items(), key=lambda item: item[1], reverse=True)[:limit]
            elif method == "vector" and vector_errors:
                logger.warning("Vector-only retrieval failed; returning empty results. errors=%s", vector_errors)
        vector_ms = (time.perf_counter() - vector_start) * 1000

        graph_start = time.perf_counter()
        query_entities = entity_names(query)
        candidate_ids = list(
            dict.fromkeys(
                [chunk_id for chunk_id, _ in vector_ranked]
                + [chunk_id for chunk_id, _ in keyword_ranked]
            )
        )
        if not candidate_ids and method == "vector":
            candidate_ids = list(chunks.keys())[:limit]
        entity_map = self.store.chunk_entities(candidate_ids)
        graph_scores = {
            chunk_id: entity_overlap_score(query_entities, entity_map.get(chunk_id, []))
            for chunk_id in candidate_ids
        }
        graph_ranked = sorted(graph_scores.items(), key=lambda item: item[1], reverse=True)
        graph_ranked = [(chunk_id, score) for chunk_id, score in graph_ranked if score > 0]
        graph_ms = (time.perf_counter() - graph_start) * 1000

        if method in {"keyword", "bm25"}:
            fused = {chunk_id: score for chunk_id, score in keyword_ranked}
        elif method == "vector":
            fused = {chunk_id: score for chunk_id, score in vector_ranked}
        else:
            fused = reciprocal_rank_fusion(
                [vector_ranked, keyword_ranked, graph_ranked],
                weights=[
                    self.settings.VECTOR_WEIGHT,
                    self.settings.KEYWORD_WEIGHT,
                    self.settings.GRAPH_WEIGHT,
                ],
                k=self.settings.RRF_K,
            )
            structured_fused = reciprocal_rank_fusion(
                [structured_ranked],
                weights=[self.settings.STRUCTURED_WEIGHT],
                k=max(8, int(self.settings.RRF_K / 3)),
            )
            for chunk_id, score in structured_fused.items():
                fused[chunk_id] = fused.get(chunk_id, 0.0) + score

        for chunk_id, graph_score in graph_scores.items():
            if chunk_id in fused and graph_score:
                fused[chunk_id] += graph_score * self.settings.GRAPH_WEIGHT

        for chunk_id in list(fused):
            chunk = chunks.get(chunk_id)
            if chunk is not None:
                fused[chunk_id] += _lexical_boost(query, chunk)
                fused[chunk_id] += _structured_lexical_boost(query, chunk)
                fused[chunk_id] += _aspect_lexical_boost(query, chunk, intent)

        ranked_ids = sorted(fused.items(), key=lambda item: item[1], reverse=True)
        if similarity_threshold is not None:
            ranked_ids = [(chunk_id, score) for chunk_id, score in ranked_ids if score >= similarity_threshold]

        results: list[ScoredChunk] = []
        source_scores = {chunk_id: score for chunk_id, score in vector_ranked + keyword_ranked}
        for rank, (chunk_id, score) in enumerate(ranked_ids[:top], start=1):
            chunk = chunks.get(chunk_id) or self.store.get_chunk(chunk_id)
            if chunk is None:
                continue
            results.append(
                ScoredChunk(
                    chunk=chunk,
                    score=round(float(score), 6),
                    source_rank=rank,
                    source_score=round(float(source_scores.get(chunk_id, score)), 6),
                    matched_entities=entity_map.get(chunk_id, []),
                    retrieval_signals={
                        "graph": round(float(graph_scores.get(chunk_id, 0.0)), 6),
                        "structured": round(float(dict(structured_ranked).get(chunk_id, 0.0)), 6),
                        **_aspect_retrieval_signals(intent.aspect, chunk),
                    },
                )
            )

        timings = {
            "retrieve_ms": round((time.perf_counter() - start) * 1000, 3),
            "keyword_ms": round(keyword_ms, 3),
            "structured_ms": round(structured_ms, 3),
            "vector_ms": round(vector_ms, 3),
            "graph_ms": round(graph_ms, 3),
        }
        if vector_errors:
            timings["vector_errors"] = round(float(len(vector_errors)), 3)
        return results, timings


def evidence_payload(scored: ScoredChunk, query: str = "") -> dict[str, Any]:
    chunk = scored.chunk
    snippet = chunk.text.strip()
    if len(snippet) > 900:
        snippet = snippet[:897] + "..."
    canonical_metadata = _canonical_metadata(chunk.metadata, chunk.source_type)
    legacy_id = _legacy_evidence_id(chunk)
    reference_url = (
        str(chunk.metadata.get("reference") or chunk.metadata.get("ones_release_link") or "").strip()
        or None
    )
    payload: dict[str, Any] = {
        "doc_name": chunk.doc_name,
        "page_label": chunk.page_label,
        "block_type": chunk.block_type,
        "source_path": chunk.source_path,
        "reference_url": reference_url,
        "index_text": chunk.index_text[:2400],
        "chunk_id": chunk.chunk_id,
        "canonical_metadata": canonical_metadata,
    }
    if legacy_id:
        payload["legacy_evidence_id"] = legacy_id
    if canonical_metadata.get("row_id"):
        payload["row_id"] = canonical_metadata["row_id"]
    field_summary = _extract_field_summary(query, chunk)
    if field_summary:
        payload["field_summary"] = field_summary
    aspect_payload = _aspect_payload(chunk, query)
    payload.update(aspect_payload)
    if chunk.source_type == "table":
        fields = scored.chunk.metadata.keys() & set(get_settings().TABLE_RETURN_FIELDS)
    elif chunk.source_type == "adela":
        fields = scored.chunk.metadata.keys() & set(get_settings().ADELA_RETURN_FIELDS)
    else:
        fields = []
    for key in fields:
        value = chunk.metadata.get(key)
        if value in (None, "", [], {}):
            continue
        value_text = str(value)
        payload[key] = value_text[:1200] + "..." if len(value_text) > 1200 else value
    for key, value in canonical_metadata.items():
        payload.setdefault(key, value)
    metadata = dict(chunk.metadata)
    metadata.update({f"canonical_{key}": value for key, value in canonical_metadata.items()})
    metadata.update(
        {
            "section_type": aspect_payload["section_type"],
            "aspects": aspect_payload["aspects"],
            "evidence_role": aspect_payload["evidence_role"],
            "query_aspect": aspect_payload["query_aspect"],
        }
    )
    if legacy_id:
        metadata["legacy_evidence_id"] = legacy_id
    return {
        "evidence_id": chunk.chunk_id,
        "legacy_evidence_id": legacy_id,
        "source_type": chunk.source_type,
        "score": scored.score,
        "source_rank": scored.source_rank,
        "source_score": scored.source_score,
        "title": chunk.title or chunk.doc_name,
        "snippet": snippet,
        "doc_id": chunk.doc_id,
        "doc_name": chunk.doc_name,
        "page_label": chunk.page_label,
        "block_type": chunk.block_type,
        "source_path": chunk.source_path,
        "reference_url": reference_url,
        "metadata": metadata,
        "matched_entities": scored.matched_entities[:12],
        "retrieval_signals": scored.retrieval_signals,
        "payload": payload,
    }


def _aspect_retrieval_signals(query_aspect: str, chunk: Chunk) -> dict[str, float]:
    score, _metadata = answerability_score(query_aspect, chunk)
    return {"answerability": round(float(score), 6)}


def _aspect_payload(chunk: Chunk, query: str) -> dict[str, Any]:
    intent = build_query_intent(query) if query else None
    query_aspect = intent.aspect if intent is not None else GENERAL_ASPECT
    score, metadata = answerability_score(query_aspect, chunk)
    aspects = tuple(metadata.get("chunk_aspects") or classify_chunk_aspects(chunk))
    section_type = str(metadata.get("section_type") or chunk_section_type(chunk, aspects))
    role = str(metadata.get("evidence_role") or "supporting")
    return {
        "query_frame": intent.query_frame if intent is not None else {},
        "query_aspect": query_aspect,
        "answer_type": intent.answer_type if intent is not None else "general_answer",
        "aspects": list(aspects),
        "section_type": section_type,
        "evidence_role": role,
        "answerability_score": round(float(score), 6),
    }
