import argparse
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import requests


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


def _load_queries(query: str, queries_file: str | None) -> list[str]:
    if not queries_file:
        return [query]
    p = Path(queries_file)
    if not p.exists():
        raise ValueError(f"queries_file 不存在: {queries_file}")
    rows = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"queries_file 中没有有效问题: {queries_file}")
    return rows


def _iter_query_runs(queries: list[str], repeat: int) -> Iterable[tuple[int, int, str]]:
    for query_index, q in enumerate(queries, start=1):
        for run_index in range(1, repeat + 1):
            yield query_index, run_index, q


def _build_payload(args: argparse.Namespace, query: str) -> dict[str, Any]:
    return {
        "query": query,
        "fused_top_k": args.fused_top_k,
        "rrf_k": args.rrf_k,
        "stream": False,
        "route_with_llm": args.route_with_llm,
        "document_config": {"enabled": args.enable_document},
        "table_config": {"enabled": args.enable_table},
        "adela_config": {"enabled": args.enable_adela},
    }


def _format_ms(v: float | None) -> str:
    return "N/A" if v is None else f"{v:.3f} ms"


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    w = rank - low
    return ordered[low] * (1 - w) + ordered[high] * w


def append_jsonl(path: str, records: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def print_run_summary(r: dict[str, Any]) -> None:
    print("=" * 80)
    print(f"请求 {r['request_id']} | 问题#{r['query_index']} 第{r['run_index']}次")
    print(f"问题: {r['query']}")
    print(f"请求状态: {'成功' if r['success'] else '失败'}")
    print(f"客户端总耗时: {_format_ms(r['client_total_ms'])}")
    print(f"服务端总耗时: {_format_ms(r['server_total_ms'])}")
    print(f"路由耗时: {_format_ms(r['route_ms'])} | 检索耗时: {_format_ms(r['retrieve_ms'])} | 融合耗时: {_format_ms(r['fuse_ms'])} | 回答耗时: {_format_ms(r['answer_ms'])}")
    print(f"融合证据数: {r['fused_count']} | 参与来源: {', '.join(r.get('selected_sources') or []) or 'N/A'}")
    print(f"答案预览: {r['answer_preview']}")
    if r.get("error_message"):
        print(f"错误信息: {r['error_message']}")


def print_aggregate_summary(results: list[dict[str, Any]]) -> None:
    def vals(k: str) -> list[float]:
        return [x[k] for x in results if x.get(k) is not None]

    metrics = {
        "客户端总耗时": vals("client_total_ms"),
        "服务端总耗时": vals("server_total_ms"),
        "路由耗时": vals("route_ms"),
        "检索耗时": vals("retrieve_ms"),
        "融合耗时": vals("fuse_ms"),
        "回答耗时": vals("answer_ms"),
    }

    print("\n" + "=" * 80)
    print(f"汇总统计: 共 {len(results)} 次请求")
    ok = sum(1 for x in results if x.get("success"))
    print(f"成功次数: {ok} | 失败次数: {len(results) - ok}")
    for n, v in metrics.items():
        if not v:
            print(f"{n}: N/A")
            continue
        print(
            f"{n}: avg={_format_ms(sum(v)/len(v))}, min={_format_ms(min(v))}, "
            f"p50={_format_ms(_percentile(v,0.5))}, p95={_format_ms(_percentile(v,0.95))}, max={_format_ms(max(v))}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="统一三源接口 benchmark：调用 unified_query 并统计耗时")
    p.add_argument("--query", default="请问 RD 现有跟人脸相关的模型有多少个")
    p.add_argument("--queries-file", help="问题文件路径，每行一个问题")
    p.add_argument("--repeat", type=int, default=1, help="每个问题重复次数")
    p.add_argument("--chat-api-url", default=os.getenv("SWIFT_RAG_UNIFIED_QUERY_URL", _default_unified_query_url()))
    p.add_argument("--fused-top-k", type=int, default=12)
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--route-with-llm", action="store_true", default=False)
    p.add_argument("--enable-document", action="store_true", default=True)
    p.add_argument("--disable-document", action="store_false", dest="enable_document")
    p.add_argument("--enable-table", action="store_true", default=True)
    p.add_argument("--disable-table", action="store_false", dest="enable_table")
    p.add_argument("--enable-adela", action="store_true", default=True)
    p.add_argument("--disable-adela", action="store_false", dest="enable_adela")
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--output-jsonl", help="结果输出 JSONL")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeat < 1:
        raise ValueError("--repeat 必须 >= 1")
    queries = _load_queries(args.query, args.queries_file)

    print(f"使用 unified 接口: {args.chat_api_url}")
    print(f"问题数量: {len(queries)} | 每题重复次数: {args.repeat}")

    session = requests.Session()
    session.headers.update({"Content-Type": "application/json", "User-Agent": "swift-rag-unified-benchmark/1.0"})

    results: list[dict[str, Any]] = []
    for query_index, run_index, q in _iter_query_runs(queries, args.repeat):
        rid = uuid.uuid4().hex[:12]
        payload = _build_payload(args, q)
        payload["client_request_id"] = rid
        start = time.perf_counter()
        try:
            resp = session.post(args.chat_api_url, json=payload, timeout=args.timeout)
            client_total_ms = (time.perf_counter() - start) * 1000
            resp.raise_for_status()
            data = resp.json()
            t = data.get("timings") or {}
            result = {
                "request_id": rid,
                "query_index": query_index,
                "run_index": run_index,
                "query": q,
                "client_total_ms": round(client_total_ms, 3),
                "server_total_ms": t.get("total_ms"),
                "route_ms": t.get("route_ms"),
                "retrieve_ms": t.get("retrieve_ms"),
                "fuse_ms": t.get("fuse_ms"),
                "answer_ms": t.get("answer_ms"),
                "fused_count": len(data.get("fused_evidences", [])),
                "selected_sources": data.get("route_plan", {}).get("selected_sources", []),
                "answer": data.get("answer", ""),
                "answer_preview": (data.get("answer", "") or "")[:160],
                "success": data.get("success", True),
                "error_message": None,
            }
        except Exception as exc:
            result = {
                "request_id": rid,
                "query_index": query_index,
                "run_index": run_index,
                "query": q,
                "client_total_ms": round((time.perf_counter() - start) * 1000, 3),
                "server_total_ms": None,
                "route_ms": None,
                "retrieve_ms": None,
                "fuse_ms": None,
                "answer_ms": None,
                "fused_count": 0,
                "selected_sources": [],
                "answer": "",
                "answer_preview": "",
                "success": False,
                "error_message": str(exc),
            }
        results.append(result)
        print_run_summary(result)

    print_aggregate_summary(results)
    if args.output_jsonl:
        append_jsonl(args.output_jsonl, results)
        print(f"\n结果已写入: {args.output_jsonl}")


if __name__ == "__main__":
    main()
