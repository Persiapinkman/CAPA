import argparse
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import requests


def _default_chat_api_url() -> str:
    api_base = os.getenv("SWIFT_RAG_API_BASE_URL")
    if api_base:
        return f"{api_base.rstrip('/')}/rag/chat_engine/query"

    try:
        from src.core.config import get_settings

        settings = get_settings()
        return (
            f"http://127.0.0.1:{settings.PORT}{settings.API_V1_STR}"
            "/rag/chat_engine/query"
        )
    except Exception:
        return "http://127.0.0.1:6060/api/v1/rag/chat_engine/query"


def _default_milvus_uri() -> str:
    try:
        from src.core.config import get_settings

        return get_settings().DATA_SOURCE_VECTOR_DB_URI
    except Exception:
        return str(
            Path(__file__).resolve().parents[1]
            / "data_source"
            / "embedding_artifacts"
            / "documents"
            / "milvus_data_source_evoqwen_3b.db"
        )


def _default_llm_config() -> dict[str, Any]:
    try:
        from src.rag.search_agent.config import DEFAULT_LLM_API_CONFIG

        return dict(DEFAULT_LLM_API_CONFIG)
    except Exception:
        return {
            "model": "Qwen3.5-4B",
            "base_url": "http://10.111.32.253:8000/v1",
            "api_key": "token.sdc@2026",
            "max_tokens": 2048,
        }


def _looks_like_html_response(content_type: str, text: str) -> bool:
    return "text/html" in content_type.lower() or text.lstrip().lower().startswith(
        "<!doctype html"
    )


def _build_endpoint_hint(url: str, response_text: str) -> str:
    if "Jupyter Server" in response_text or "_xsrf" in response_text:
        return (
            f"当前请求打到了 Jupyter 服务而不是 RAG API: {url}\n"
            "请检查端口是否写错了。RAG 服务默认应为 "
            "`http://127.0.0.1:6060/api/v1/rag/chat_engine/query`。"
        )
    return f"接口返回了非预期响应，请检查服务地址是否正确: {url}"


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def _format_ms(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f} ms"


def _build_payload(args: argparse.Namespace, query: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": query,
        "top_k": args.top_k,
        "similarity_threshold": args.similarity_threshold,
        "uri": args.milvus_uri,
        "collection_name": args.collection_name,
        "embedding_models": args.embedding_models,
    }
    if args.filter_expr:
        payload["filter"] = args.filter_expr
    if args.llm_model or args.llm_base_url or args.llm_api_key or args.llm_max_tokens:
        llm_defaults = _default_llm_config()
        payload["llm_config"] = {
            "model": args.llm_model or llm_defaults["model"],
            "base_url": args.llm_base_url or llm_defaults["base_url"],
            "api_key": args.llm_api_key or llm_defaults["api_key"],
            "max_tokens": args.llm_max_tokens or llm_defaults["max_tokens"],
        }
    return payload


def _load_queries(args: argparse.Namespace) -> list[str]:
    if args.queries_file:
        queries_path = Path(args.queries_file)
        if not queries_path.exists():
            raise ValueError(
                "queries_file 不存在: "
                f"{args.queries_file}\n"
                "请传入真实的问题文件路径，或直接使用示例文件 "
                "`sample_code/questions.txt`。"
            )
        lines = queries_path.read_text(encoding="utf-8").splitlines()
        queries = [line.strip() for line in lines if line.strip()]
        if not queries:
            raise ValueError(f"queries_file 中没有有效问题: {args.queries_file}")
        return queries

    return [args.query]


def _iter_query_runs(queries: list[str], repeat: int) -> Iterable[tuple[int, int, str]]:
    for query_index, query in enumerate(queries, start=1):
        for run_index in range(1, repeat + 1):
            yield query_index, run_index, query


def invoke_rag_chat(
    session: requests.Session,
    url: str,
    payload: dict[str, Any],
    timeout: float,
) -> tuple[dict[str, Any], float]:
    start = time.perf_counter()
    try:
        response = session.post(url, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(
            f"问答接口请求失败，无法连接到 RAG API: {url}\n"
            f"原始错误: {exc}"
        ) from exc
    client_total_ms = (time.perf_counter() - start) * 1000

    content_type = response.headers.get("Content-Type", "")
    if response.status_code != 200:
        detail = response.text[:2000]
        if _looks_like_html_response(content_type, detail):
            detail = _build_endpoint_hint(url, detail)
        raise RuntimeError(
            f"问答接口请求失败，状态码: {response.status_code}\n{detail}"
        )

    try:
        return response.json(), client_total_ms
    except ValueError as exc:
        detail = response.text[:2000]
        if _looks_like_html_response(content_type, detail):
            detail = _build_endpoint_hint(url, detail)
        raise RuntimeError(
            "问答接口返回的不是合法 JSON。\n"
            f"{detail}"
        ) from exc


def print_run_summary(result: dict[str, Any]) -> None:
    print("=" * 80)
    print(
        f"请求 {result['request_id']} | 问题#{result['query_index']} 第{result['run_index']}次"
    )
    print(f"问题: {result['query']}")
    print(f"请求状态: {'成功' if result['success'] else '失败'}")
    print(f"客户端总耗时: {_format_ms(result['client_total_ms'])}")
    print(f"服务端总耗时: {_format_ms(result['server_total_ms'])}")
    print(f"客户端额外开销: {_format_ms(result['client_overhead_ms'])}")
    print(f"检索耗时: {_format_ms(result['retrieve_ms'])}")
    print(f"回答耗时: {_format_ms(result['answer_ms'])}")
    print(f"reference耗时: {_format_ms(result['reference_ms'])}")
    print(
        f"返回块数: {result['retrieved_count']} | 来源数: {result['reference_count']} | 答案长度: {result['answer_length']}"
    )
    if result["error_message"]:
        print(f"错误信息: {result['error_message']}")
    print(f"答案预览: {result['answer_preview']}")


def print_aggregate_summary(results: list[dict[str, Any]]) -> None:
    def values(key: str) -> list[float]:
        return [item[key] for item in results if item[key] is not None]

    metric_map = {
        "客户端总耗时": values("client_total_ms"),
        "服务端总耗时": values("server_total_ms"),
        "客户端额外开销": values("client_overhead_ms"),
        "检索耗时": values("retrieve_ms"),
        "回答耗时": values("answer_ms"),
        "reference耗时": values("reference_ms"),
    }

    print("\n" + "=" * 80)
    print(f"汇总统计: 共 {len(results)} 次请求")
    success_count = sum(1 for item in results if item["success"])
    print(f"成功次数: {success_count} | 失败次数: {len(results) - success_count}")
    for metric_name, metric_values in metric_map.items():
        if not metric_values:
            print(f"{metric_name}: N/A")
            continue
        print(
            f"{metric_name}: "
            f"avg={_format_ms(sum(metric_values) / len(metric_values))}, "
            f"min={_format_ms(min(metric_values))}, "
            f"p50={_format_ms(_percentile(metric_values, 0.5))}, "
            f"p95={_format_ms(_percentile(metric_values, 0.95))}, "
            f"max={_format_ms(max(metric_values))}"
        )


def append_jsonl(path: str, records: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="模拟真实场景调用 document 模型发版文档正文问答 API，并输出端到端耗时统计。"
    )
    parser.add_argument(
        "--query",
        default=os.getenv(
            "RAG_SAMPLE_QUERY",
            "safety_rope v0.2.1 追加了什么数据，标签有哪些？",
        ),
        help="单次测试的问题文本",
    )
    parser.add_argument(
        "--queries-file",
        help="问题文件路径，每行一个问题；传入后会忽略 --query",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="每个问题重复调用次数",
    )
    parser.add_argument(
        "--chat-api-url",
        default=os.getenv("SWIFT_RAG_CHAT_API_URL", _default_chat_api_url()),
        help="document 模型发版文档正文问答 API 地址",
    )
    parser.add_argument(
        "--milvus-uri",
        default=os.getenv("SWIFT_RAG_MILVUS_URI", _default_milvus_uri()),
        help="向量库地址",
    )
    parser.add_argument(
        "--collection-name",
        default=os.getenv("SWIFT_RAG_COLLECTION_NAME", "llamacollection"),
        help="collection 名称",
    )
    parser.add_argument(
        "--embedding-models",
        nargs="+",
        default=["EvoQwen2.5-VL-Retriever-3B-v1"],
        help="检索使用的 embedding 模型列表",
    )
    parser.add_argument("--top-k", type=int, default=5, help="返回块数")
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.5,
        help="相似度阈值",
    )
    parser.add_argument(
        "--filter-expr",
        help="检索 filter 表达式",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="HTTP 请求超时时间，单位秒",
    )
    parser.add_argument("--llm-model", help="覆盖默认 LLM model")
    parser.add_argument("--llm-base-url", help="覆盖默认 LLM base_url")
    parser.add_argument("--llm-api-key", help="覆盖默认 LLM api_key")
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        help="覆盖默认 LLM max_tokens",
    )
    parser.add_argument(
        "--output-jsonl",
        help="将每次调用结果附加写入指定 JSONL 文件",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeat < 1:
        raise ValueError("--repeat 必须大于等于 1")

    queries = _load_queries(args)
    results: list[dict[str, Any]] = []

    print(f"使用问答接口: {args.chat_api_url}")
    print(f"使用向量库: {args.milvus_uri}")
    print(f"问题数量: {len(queries)} | 每题重复次数: {args.repeat}")

    session = requests.Session()
    session.headers.update(
        {
            "Content-Type": "application/json",
            "User-Agent": "swift-rag-benchmark/1.0",
        }
    )

    for query_index, run_index, query in _iter_query_runs(queries, args.repeat):
        request_id = uuid.uuid4().hex[:12]
        payload = _build_payload(args, query)
        payload["client_request_id"] = request_id

        request_start = time.perf_counter()
        try:
            data, client_total_ms = invoke_rag_chat(
                session=session,
                url=args.chat_api_url,
                payload=payload,
                timeout=args.timeout,
            )
            timings = data.get("timings") or {}
            server_total_ms = timings.get("total_ms")
            result = {
                "request_id": request_id,
                "query_index": query_index,
                "run_index": run_index,
                "query": query,
                "client_total_ms": round(client_total_ms, 3),
                "server_total_ms": server_total_ms,
                "client_overhead_ms": (
                    round(client_total_ms - server_total_ms, 3)
                    if server_total_ms is not None
                    else None
                ),
                "retrieve_ms": timings.get("retrieve_ms"),
                "answer_ms": timings.get("answer_ms"),
                "reference_ms": timings.get("reference_ms"),
                "retrieved_count": len(data.get("retrieved_chunks", [])),
                "reference_count": len(data.get("reference", [])),
                "answer_length": len(data.get("answer", "")),
                "answer": data.get("answer", ""),
                "answer_preview": (data.get("answer", "") or "")[:160],
                "success": data.get("success", True),
                "error_message": None,
            }
        except Exception as exc:
            result = {
                "request_id": request_id,
                "query_index": query_index,
                "run_index": run_index,
                "query": query,
                "client_total_ms": round((time.perf_counter() - request_start) * 1000, 3),
                "server_total_ms": None,
                "client_overhead_ms": None,
                "retrieve_ms": None,
                "answer_ms": None,
                "reference_ms": None,
                "retrieved_count": 0,
                "reference_count": 0,
                "answer_length": 0,
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
