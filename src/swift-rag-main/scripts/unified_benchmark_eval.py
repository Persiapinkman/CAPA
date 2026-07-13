#!/usr/bin/env python3
"""Unified benchmark evaluator for the unified_query endpoint.

Outputs:
- config.json
- requests.jsonl
- requests.csv
- summary.json
- failures.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import time
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import requests


SOURCE_TYPES = ("document", "table", "adela")


def _default_unified_query_url() -> str:
    api_base = os.getenv("SWIFT_RAG_API_BASE_URL")
    if api_base:
        return f"{api_base.rstrip('/')}/rag/chat_engine/unified_query"

    try:
        from src.core.config import get_settings

        settings = get_settings()
        return (
            f"http://127.0.0.1:{settings.PORT}{settings.API_V1_STR}"
            "/rag/chat_engine/unified_query"
        )
    except Exception:
        return "http://127.0.0.1:6060/api/v1/rag/chat_engine/unified_query"


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(v) for v in values)
    rank = (len(ordered) - 1) * q
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _stats(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {
            "count": 0,
            "avg": 0.0,
            "min": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    data = [float(v) for v in values]
    return {
        "count": len(data),
        "avg": round(sum(data) / len(data), 3),
        "min": round(min(data), 3),
        "p50": round(_percentile(data, 0.5), 3),
        "p90": round(_percentile(data, 0.9), 3),
        "p95": round(_percentile(data, 0.95), 3),
        "max": round(max(data), 3),
    }


def _normalize_text(text: Any) -> str:
    if text is None:
        return ""
    value = str(text).strip().lower()
    value = re.sub(r"\s+", "", value)
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", value)


def _normalize_key(text: Any) -> str:
    return _normalize_text(text)


def _set_metrics(expected: set[str], predicted: set[str]) -> Dict[str, float]:
    if not expected and not predicted:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "hit": 1.0}
    if not expected:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0, "hit": 1.0}
    if not predicted:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "hit": 0.0}

    inter = len(expected.intersection(predicted))
    precision = inter / max(len(predicted), 1)
    recall = inter / max(len(expected), 1)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0.0
        else 0.0
    )
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "hit": 1.0 if inter > 0 else 0.0,
    }


def _char_f1(reference: str, answer: str) -> float:
    ref = _normalize_text(reference)
    pred = _normalize_text(answer)
    if not ref and not pred:
        return 1.0
    if not ref or not pred:
        return 0.0
    ref_counter = Counter(ref)
    pred_counter = Counter(pred)
    common = 0
    for k, c in ref_counter.items():
        common += min(c, pred_counter.get(k, 0))
    precision = common / len(pred)
    recall = common / len(ref)
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _fallback_keywords(reference_answer: str) -> List[str]:
    parts = re.split(r"[\s,，。.;；:：/\\|()\[\]{}<>《》\"'`]+", reference_answer or "")
    out: List[str] = []
    seen: set[str] = set()
    for token in parts:
        token = token.strip()
        if not token:
            continue
        if len(_normalize_text(token)) < 2:
            continue
        if token in seen:
            continue
        out.append(token)
        seen.add(token)
    return out[:12]


def _extract_expected_source_types(item: Dict[str, Any]) -> List[str]:
    sources = item.get("retrieval_source_types")
    output: List[str] = []
    if isinstance(sources, list):
        for s in sources:
            value = str(s).strip().lower()
            if value in SOURCE_TYPES and value not in output:
                output.append(value)
    if not output:
        source_type = str(item.get("source_type", "")).strip().lower()
        if source_type in SOURCE_TYPES:
            output.append(source_type)
    return output


def _extract_expected_evidence_keys(item: Dict[str, Any]) -> List[str]:
    keys: List[str] = []

    def push_many(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, (list, tuple, set)):
            for one in value:
                push_many(one)
            return
        normalized = _normalize_key(value)
        if normalized and normalized not in keys:
            keys.append(normalized)

    for candidate in (item.get("source_id"), item.get("source_doc")):
        push_many(candidate)
    return keys


def _extract_evidence_keys(evidence: Dict[str, Any]) -> List[str]:
    keys: List[str] = []

    def push(value: Any) -> None:
        normalized = _normalize_key(value)
        if normalized and normalized not in keys:
            keys.append(normalized)

    push(evidence.get("evidence_id"))
    push(evidence.get("title"))
    payload = evidence.get("payload") or {}
    if isinstance(payload, dict):
        push(payload.get("row_id"))
        push(payload.get("doc_name"))
        push(payload.get("doc_id"))
        push(payload.get("index_id"))
        entity = payload.get("entity")
        if isinstance(entity, dict):
            push(entity.get("model_name"))
            push(entity.get("name"))
            push(entity.get("rid"))
            push(entity.get("did"))
            push(entity.get("source_file"))
    return keys


def _key_match(expected: str, observed: str) -> bool:
    if not expected or not observed:
        return False
    if expected == observed:
        return True
    # Substring checks are used for ids with additional prefixes/suffixes.
    min_len = min(len(expected), len(observed))
    if min_len >= 8 and (expected in observed or observed in expected):
        return True
    return False


def _evaluate_answer(
    benchmark_item: Dict[str, Any],
    answer: str,
) -> Dict[str, Any]:
    reference_answer = str(benchmark_item.get("reference_answer", "") or "")
    answer = answer or ""

    reference_norm = _normalize_text(reference_answer)
    answer_norm = _normalize_text(answer)

    exact_match = bool(reference_norm) and reference_norm == answer_norm
    containment = False
    if reference_norm and answer_norm:
        if (
            len(reference_norm) >= 6
            and reference_norm in answer_norm
        ) or (
            len(answer_norm) >= 6
            and answer_norm in reference_norm
        ):
            containment = True

    expected_keywords = benchmark_item.get("expected_keywords")
    keywords: List[str]
    if isinstance(expected_keywords, list):
        keywords = [str(k) for k in expected_keywords if str(k).strip()]
    else:
        keywords = []
    if not keywords:
        keywords = _fallback_keywords(reference_answer)

    keyword_hits: List[str] = []
    for kw in keywords:
        if _normalize_text(kw) and _normalize_text(kw) in answer_norm:
            keyword_hits.append(kw)
    keyword_recall = (
        len(keyword_hits) / len(keywords)
        if keywords
        else 0.0
    )

    char_f1 = _char_f1(reference_answer, answer)
    score = max(
        1.0 if exact_match else 0.0,
        0.95 if containment else 0.0,
        keyword_recall,
        char_f1,
    )
    answer_correct = score >= 0.8

    return {
        "reference_answer": reference_answer,
        "answer_length": len(answer),
        "exact_match": exact_match,
        "reference_containment": containment,
        "keyword_total": len(keywords),
        "keyword_hit": len(keyword_hits),
        "keyword_recall": round(keyword_recall, 6),
        "keyword_hits": keyword_hits,
        "char_f1": round(char_f1, 6),
        "answer_score": round(score, 6),
        "answer_correct": answer_correct,
    }


def _evaluate_retrieval(
    benchmark_item: Dict[str, Any],
    response_data: Dict[str, Any],
) -> Dict[str, Any]:
    expected_source_types = set(_extract_expected_source_types(benchmark_item))
    expected_keys = _extract_expected_evidence_keys(benchmark_item)

    route_plan = response_data.get("route_plan") or {}
    selected_sources = route_plan.get("selected_sources") or []
    route_source_set = {
        str(s).strip().lower()
        for s in selected_sources
        if str(s).strip().lower() in SOURCE_TYPES
    }
    route_metrics = _set_metrics(expected_source_types, route_source_set)

    fused_evidences = response_data.get("fused_evidences") or []
    if not isinstance(fused_evidences, list):
        fused_evidences = []
    fused_source_set = {
        str(item.get("source_type", "")).strip().lower()
        for item in fused_evidences
        if str(item.get("source_type", "")).strip().lower() in SOURCE_TYPES
    }
    fused_source_metrics = _set_metrics(expected_source_types, fused_source_set)

    matched_ranks: List[int] = []
    for rank, evidence in enumerate(fused_evidences, start=1):
        evidence_keys = _extract_evidence_keys(evidence if isinstance(evidence, dict) else {})
        matched = False
        for expected in expected_keys:
            for observed in evidence_keys:
                if _key_match(expected, observed):
                    matched = True
                    break
            if matched:
                break
        if matched:
            matched_ranks.append(rank)

    fused_k = len(fused_evidences)
    evidence_hit = bool(matched_ranks)
    evidence_precision = (len(matched_ranks) / fused_k) if fused_k > 0 else 0.0
    evidence_recall = 1.0 if evidence_hit else 0.0
    evidence_mrr = (1.0 / matched_ranks[0]) if matched_ranks else 0.0

    return {
        "expected_source_types": sorted(expected_source_types),
        "expected_evidence_keys": expected_keys,
        "route_selected_sources": sorted(route_source_set),
        "route_precision": route_metrics["precision"],
        "route_recall": route_metrics["recall"],
        "route_f1": route_metrics["f1"],
        "route_hit": route_metrics["hit"],
        "fused_source_types": sorted(fused_source_set),
        "fused_source_precision": fused_source_metrics["precision"],
        "fused_source_recall": fused_source_metrics["recall"],
        "fused_source_f1": fused_source_metrics["f1"],
        "fused_source_hit": fused_source_metrics["hit"],
        "fused_count": fused_k,
        "evidence_match_count": len(matched_ranks),
        "evidence_first_hit_rank": matched_ranks[0] if matched_ranks else None,
        "evidence_precision_at_k": round(evidence_precision, 6),
        "evidence_recall_at_k": round(evidence_recall, 6),
        "evidence_mrr": round(evidence_mrr, 6),
    }


def _looks_like_html_response(content_type: str, text: str) -> bool:
    content_type = (content_type or "").lower()
    return "text/html" in content_type or text.lstrip().lower().startswith("<!doctype html")


def _build_endpoint_hint(url: str, response_text: str) -> str:
    if "Jupyter Server" in response_text or "_xsrf" in response_text:
        return (
            f"request reached Jupyter instead of RAG API: {url}. "
            "Please check host/port."
        )
    return f"unexpected non-json response from endpoint: {url}"


def _load_benchmark(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    if not rows:
        raise ValueError(f"empty benchmark file: {path}")
    return rows


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "benchmark_index",
        "benchmark_id",
        "run_index",
        "request_id",
        "query",
        "success",
        "http_status",
        "error_message",
        "client_total_ms",
        "server_total_ms",
        "route_ms",
        "retrieve_ms",
        "fuse_ms",
        "answer_ms",
        "fused_count",
        "route_reason",
        "route_fallback_used",
        "route_selected_sources",
        "fused_source_types",
        "expected_source_types",
        "route_precision",
        "route_recall",
        "route_f1",
        "route_hit",
        "fused_source_precision",
        "fused_source_recall",
        "fused_source_f1",
        "fused_source_hit",
        "evidence_match_count",
        "evidence_first_hit_rank",
        "evidence_precision_at_k",
        "evidence_recall_at_k",
        "evidence_mrr",
        "exact_match",
        "reference_containment",
        "keyword_total",
        "keyword_hit",
        "keyword_recall",
        "char_f1",
        "answer_score",
        "answer_correct",
        "reference_answer",
        "answer_text",
        "answer_preview",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "benchmark_index": row.get("benchmark_index"),
                    "benchmark_id": row.get("benchmark_id"),
                    "run_index": row.get("run_index"),
                    "request_id": row.get("request_id"),
                    "query": row.get("query"),
                    "success": row.get("success"),
                    "http_status": row.get("http_status"),
                    "error_message": row.get("error_message"),
                    "client_total_ms": row.get("client_total_ms"),
                    "server_total_ms": row.get("server_total_ms"),
                    "route_ms": row.get("route_ms"),
                    "retrieve_ms": row.get("retrieve_ms"),
                    "fuse_ms": row.get("fuse_ms"),
                    "answer_ms": row.get("answer_ms"),
                    "fused_count": row.get("fused_count"),
                    "route_reason": row.get("route_reason"),
                    "route_fallback_used": row.get("route_fallback_used"),
                    "route_selected_sources": ",".join(row.get("route_selected_sources", [])),
                    "fused_source_types": ",".join(row.get("fused_source_types", [])),
                    "expected_source_types": ",".join(row.get("expected_source_types", [])),
                    "route_precision": row.get("route_precision"),
                    "route_recall": row.get("route_recall"),
                    "route_f1": row.get("route_f1"),
                    "route_hit": row.get("route_hit"),
                    "fused_source_precision": row.get("fused_source_precision"),
                    "fused_source_recall": row.get("fused_source_recall"),
                    "fused_source_f1": row.get("fused_source_f1"),
                    "fused_source_hit": row.get("fused_source_hit"),
                    "evidence_match_count": row.get("evidence_match_count"),
                    "evidence_first_hit_rank": row.get("evidence_first_hit_rank"),
                    "evidence_precision_at_k": row.get("evidence_precision_at_k"),
                    "evidence_recall_at_k": row.get("evidence_recall_at_k"),
                    "evidence_mrr": row.get("evidence_mrr"),
                    "exact_match": row.get("exact_match"),
                    "reference_containment": row.get("reference_containment"),
                    "keyword_total": row.get("keyword_total"),
                    "keyword_hit": row.get("keyword_hit"),
                    "keyword_recall": row.get("keyword_recall"),
                    "char_f1": row.get("char_f1"),
                    "answer_score": row.get("answer_score"),
                    "answer_correct": row.get("answer_correct"),
                    "reference_answer": row.get("reference_answer"),
                    "answer_text": row.get("answer_text"),
                    "answer_preview": row.get("answer_preview"),
                }
            )


def _build_payload(args: argparse.Namespace, query: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "query": query,
        "fused_top_k": args.fused_top_k,
        "rrf_k": args.rrf_k,
        "stream": False,
        "route_with_llm": args.route_with_llm,
        "document_config": {"enabled": args.enable_document},
        "table_config": {"enabled": args.enable_table},
        "adela_config": {"enabled": args.enable_adela},
    }
    if args.llm_base_url:
        llm_config: Dict[str, Any] = {
            "base_url": args.llm_base_url,
        }
        if args.llm_model:
            llm_config["model"] = args.llm_model
        if args.llm_api_key:
            llm_config["api_key"] = args.llm_api_key
        if args.llm_max_tokens is not None:
            llm_config["max_tokens"] = args.llm_max_tokens
        if args.llm_temperature is not None:
            llm_config["temperature"] = args.llm_temperature
        if args.llm_top_p is not None:
            llm_config["top_p"] = args.llm_top_p
        payload["llm_config"] = llm_config
    return payload


def _resolve_run_dir(output_root: Path, run_name: str | None) -> Path:
    if run_name:
        return output_root / run_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_root / f"unified_benchmark_{timestamp}"


def _build_summary(
    args: argparse.Namespace,
    benchmark_path: Path,
    run_dir: Path,
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    success_rows = [row for row in rows if row.get("success")]
    failed_rows = [row for row in rows if not row.get("success")]

    def avg_of(key: str) -> float:
        values = [_safe_float(row.get(key)) for row in rows]
        cleaned = [v for v in values if v is not None]
        if not cleaned:
            return 0.0
        return round(sum(cleaned) / len(cleaned), 6)

    latency = {
        "client_total_ms": _stats([row["client_total_ms"] for row in rows if row.get("client_total_ms") is not None]),
        "server_total_ms": _stats([row["server_total_ms"] for row in rows if row.get("server_total_ms") is not None]),
        "route_ms": _stats([row["route_ms"] for row in rows if row.get("route_ms") is not None]),
        "retrieve_ms": _stats([row["retrieve_ms"] for row in rows if row.get("retrieve_ms") is not None]),
        "fuse_ms": _stats([row["fuse_ms"] for row in rows if row.get("fuse_ms") is not None]),
        "answer_ms": _stats([row["answer_ms"] for row in rows if row.get("answer_ms") is not None]),
    }

    source_breakdown: Dict[str, Dict[str, Any]] = {}
    for source_type in SOURCE_TYPES:
        subset = [
            row for row in rows
            if source_type in (row.get("expected_source_types") or [])
        ]
        if not subset:
            continue
        source_breakdown[source_type] = {
            "count": len(subset),
            "fused_source_hit_rate": round(
                sum(float(row.get("fused_source_hit", 0.0)) for row in subset) / len(subset),
                6,
            ),
            "evidence_recall_at_k": round(
                sum(float(row.get("evidence_recall_at_k", 0.0)) for row in subset) / len(subset),
                6,
            ),
            "answer_correct_rate": round(
                sum(1.0 for row in subset if row.get("answer_correct")) / len(subset),
                6,
            ),
            "avg_char_f1": round(
                sum(float(row.get("char_f1", 0.0)) for row in subset) / len(subset),
                6,
            ),
        }

    route_combo_counter = Counter(
        "+".join(row.get("route_selected_sources") or ["none"]) for row in rows
    )

    summary = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "benchmark_path": str(benchmark_path),
            "api_url": args.api_url,
            "output_dir": str(run_dir),
            "repeat": args.repeat,
            "limit": args.limit,
            "fused_top_k": args.fused_top_k,
            "rrf_k": args.rrf_k,
            "route_with_llm": args.route_with_llm,
            "enable_document": args.enable_document,
            "enable_table": args.enable_table,
            "enable_adela": args.enable_adela,
            "timeout": args.timeout,
            "sleep_ms": args.sleep_ms,
        },
        "counts": {
            "total_requests": len(rows),
            "success_requests": len(success_rows),
            "failed_requests": len(failed_rows),
            "success_rate": round(len(success_rows) / len(rows), 6) if rows else 0.0,
        },
        "latency_ms": latency,
        "retrieval_metrics": {
            "avg_route_precision": avg_of("route_precision"),
            "avg_route_recall": avg_of("route_recall"),
            "avg_route_f1": avg_of("route_f1"),
            "route_hit_rate": avg_of("route_hit"),
            "avg_fused_source_precision": avg_of("fused_source_precision"),
            "avg_fused_source_recall": avg_of("fused_source_recall"),
            "avg_fused_source_f1": avg_of("fused_source_f1"),
            "fused_source_hit_rate": avg_of("fused_source_hit"),
            "avg_evidence_precision_at_k": avg_of("evidence_precision_at_k"),
            "avg_evidence_recall_at_k": avg_of("evidence_recall_at_k"),
            "avg_evidence_mrr": avg_of("evidence_mrr"),
        },
        "answer_metrics": {
            "exact_match_rate": round(
                sum(1.0 for row in rows if row.get("exact_match")) / len(rows), 6
            )
            if rows
            else 0.0,
            "reference_containment_rate": round(
                sum(1.0 for row in rows if row.get("reference_containment")) / len(rows), 6
            )
            if rows
            else 0.0,
            "avg_keyword_recall": avg_of("keyword_recall"),
            "avg_char_f1": avg_of("char_f1"),
            "avg_answer_score": avg_of("answer_score"),
            "answer_correct_rate": round(
                sum(1.0 for row in rows if row.get("answer_correct")) / len(rows), 6
            )
            if rows
            else 0.0,
        },
        "source_type_breakdown": source_breakdown,
        "route_selected_distribution": [
            {"sources": key, "count": count}
            for key, count in route_combo_counter.most_common()
        ],
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run benchmark against unified_query API and evaluate latency, "
            "retrieval precision/recall, and answer quality metrics."
        )
    )
    parser.add_argument(
        "--benchmark-jsonl",
        default="benchmark/single_step_rag_100_v1/benchmark_100.jsonl",
        help="benchmark JSONL path",
    )
    parser.add_argument(
        "--api-url",
        default=os.getenv("SWIFT_RAG_UNIFIED_QUERY_URL", _default_unified_query_url()),
        help="unified_query endpoint URL",
    )
    parser.add_argument(
        "--output-root",
        default="results/unified_benchmark",
        help="parent output directory",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="optional run directory name under output-root",
    )
    parser.add_argument("--repeat", type=int, default=1, help="repeat count per benchmark row")
    parser.add_argument("--limit", type=int, default=0, help="max benchmark rows to run (0 = all)")
    parser.add_argument("--timeout", type=float, default=180.0, help="request timeout seconds")
    parser.add_argument("--fused-top-k", type=int, default=12, help="fused_top_k in API payload")
    parser.add_argument("--rrf-k", type=int, default=60, help="rrf_k in API payload")
    parser.add_argument("--route-with-llm", action="store_true", default=False)
    parser.add_argument("--enable-document", action="store_true", default=True)
    parser.add_argument("--disable-document", action="store_false", dest="enable_document")
    parser.add_argument("--enable-table", action="store_true", default=True)
    parser.add_argument("--disable-table", action="store_false", dest="enable_table")
    parser.add_argument("--enable-adela", action="store_true", default=True)
    parser.add_argument("--disable-adela", action="store_false", dest="enable_adela")
    parser.add_argument("--llm-model", default=None, help="override llm_config.model")
    parser.add_argument("--llm-base-url", default=None, help="override llm_config.base_url")
    parser.add_argument("--llm-api-key", default=None, help="override llm_config.api_key")
    parser.add_argument("--llm-max-tokens", type=int, default=None, help="override llm max_tokens")
    parser.add_argument("--llm-temperature", type=float, default=None, help="override llm temperature")
    parser.add_argument("--llm-top-p", type=float, default=None, help="override llm top_p")
    parser.add_argument("--sleep-ms", type=float, default=0.0, help="sleep between requests")
    parser.add_argument(
        "--include-raw-response",
        action="store_true",
        default=False,
        help="include full API response in requests.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1")
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")

    benchmark_path = Path(args.benchmark_jsonl)
    if not benchmark_path.exists():
        raise FileNotFoundError(f"benchmark file not found: {benchmark_path}")
    benchmark_rows = _load_benchmark(benchmark_path)
    if args.limit:
        benchmark_rows = benchmark_rows[: args.limit]

    output_root = Path(args.output_root)
    run_dir = _resolve_run_dir(output_root, args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    config_payload = {
        "args": vars(args),
        "benchmark_count": len(benchmark_rows),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(run_dir / "config.json", config_payload)

    print(f"unified benchmark: {benchmark_path}")
    print(f"api endpoint: {args.api_url}")
    print(f"questions: {len(benchmark_rows)} | repeat: {args.repeat}")
    print(f"output dir: {run_dir}")

    session = requests.Session()
    session.headers.update(
        {
            "Content-Type": "application/json",
            "User-Agent": "swift-rag-unified-benchmark-eval/1.0",
        }
    )

    records: List[Dict[str, Any]] = []
    for benchmark_index, item in enumerate(benchmark_rows, start=1):
        question = str(item.get("question", "")).strip()
        benchmark_id = str(item.get("id") or f"q{benchmark_index}")
        for run_index in range(1, args.repeat + 1):
            request_id = uuid.uuid4().hex[:12]
            payload = _build_payload(args, question)
            payload["client_request_id"] = request_id

            client_total_ms: float | None = None
            status_code: int | None = None
            response_data: Dict[str, Any] = {}
            error_message: str | None = None

            started = time.perf_counter()
            try:
                response = session.post(
                    args.api_url,
                    json=payload,
                    timeout=args.timeout,
                )
                client_total_ms = round((time.perf_counter() - started) * 1000, 3)
                status_code = response.status_code
                content_type = response.headers.get("Content-Type", "")
                if response.status_code != 200:
                    detail = response.text[:2000]
                    if _looks_like_html_response(content_type, detail):
                        detail = _build_endpoint_hint(args.api_url, detail)
                    raise RuntimeError(f"http={response.status_code}: {detail}")
                try:
                    response_data = response.json()
                except ValueError as exc:
                    detail = response.text[:2000]
                    if _looks_like_html_response(content_type, detail):
                        detail = _build_endpoint_hint(args.api_url, detail)
                    raise RuntimeError(f"invalid json response: {detail}") from exc
            except Exception as exc:
                client_total_ms = round((time.perf_counter() - started) * 1000, 3)
                error_message = str(exc)

            timings = response_data.get("timings") or {}
            retrieval_metrics = _evaluate_retrieval(item, response_data)
            answer_metrics = _evaluate_answer(item, str(response_data.get("answer", "") or ""))

            record: Dict[str, Any] = {
                "benchmark_index": benchmark_index,
                "benchmark_id": benchmark_id,
                "run_index": run_index,
                "request_id": request_id,
                "query": question,
                "success": bool(response_data.get("success", False)) and not error_message,
                "http_status": status_code,
                "error_message": error_message or response_data.get("message"),
                "client_total_ms": client_total_ms,
                "server_total_ms": _safe_float(timings.get("total_ms")),
                "route_ms": _safe_float(timings.get("route_ms")),
                "retrieve_ms": _safe_float(timings.get("retrieve_ms")),
                "fuse_ms": _safe_float(timings.get("fuse_ms")),
                "answer_ms": _safe_float(timings.get("answer_ms")),
                "source_status": response_data.get("source_status") or [],
                "route_reason": ((response_data.get("route_plan") or {}).get("reason")),
                "route_fallback_used": ((response_data.get("route_plan") or {}).get("fallback_used")),
                "answer_text": str(response_data.get("answer", "") or ""),
                "answer_preview": str(response_data.get("answer", "") or "")[:240],
                **retrieval_metrics,
                **answer_metrics,
            }
            if args.include_raw_response:
                record["raw_response"] = response_data
                record["request_payload"] = payload

            records.append(record)

            print(
                f"[{benchmark_index:03d}/{len(benchmark_rows):03d}] run={run_index} "
                f"id={benchmark_id} success={record['success']} "
                f"latency={record['client_total_ms']}ms "
                f"retr_hit={record['evidence_recall_at_k']:.0f} "
                f"ans_ok={record['answer_correct']}"
            )

            if args.sleep_ms > 0:
                time.sleep(args.sleep_ms / 1000.0)

    summary = _build_summary(
        args=args,
        benchmark_path=benchmark_path,
        run_dir=run_dir,
        rows=records,
    )

    failures = [
        row for row in records
        if (not row.get("success"))
        or (float(row.get("evidence_recall_at_k", 0.0)) < 1.0)
        or (not row.get("answer_correct"))
    ]

    _write_jsonl(run_dir / "requests.jsonl", records)
    _write_csv(run_dir / "requests.csv", records)
    _write_json(run_dir / "summary.json", summary)
    _write_jsonl(run_dir / "failures.jsonl", failures)

    print("")
    print("done")
    print(f"requests: {run_dir / 'requests.jsonl'}")
    print(f"csv:      {run_dir / 'requests.csv'}")
    print(f"summary:  {run_dir / 'summary.json'}")
    print(f"failures: {run_dir / 'failures.jsonl'}")
    print("")
    print(
        "headline metrics: "
        f"success_rate={summary['counts']['success_rate']:.2%}, "
        f"evidence_recall@k={summary['retrieval_metrics']['avg_evidence_recall_at_k']:.2%}, "
        f"answer_correct={summary['answer_metrics']['answer_correct_rate']:.2%}"
    )


if __name__ == "__main__":
    main()
