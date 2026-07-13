import json
import os
from typing import Any

import requests


def _default_unified_stream_api_url() -> str:
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


def _looks_like_html_response(content_type: str, text: str) -> bool:
    return "text/html" in content_type.lower() or text.lstrip().lower().startswith(
        "<!doctype html"
    )


def _build_endpoint_hint(url: str, response_text: str) -> str:
    if "Jupyter Server" in response_text or "_xsrf" in response_text:
        return (
            f"当前请求打到了 Jupyter 服务而不是 RAG API: {url}\n"
            "请检查端口是否写错了。RAG 服务默认应为 "
            "`http://127.0.0.1:6060/api/v1/rag/chat_engine/unified_query`。"
        )
    return f"接口返回了非预期响应，请检查服务地址是否正确: {url}"


unified_stream_api_url = os.getenv(
    "SWIFT_RAG_UNIFIED_STREAM_API_URL",
    _default_unified_stream_api_url(),
)


def query_unified_stream(
    query: str,
    fused_top_k: int = 12,
    rrf_k: int = 60,
    route_with_llm: bool = True,
    timeout: int = 180,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": query,
        "fused_top_k": fused_top_k,
        "rrf_k": rrf_k,
        "stream": True,
        "route_with_llm": route_with_llm,
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(
            unified_stream_api_url,
            json=payload,
            headers=headers,
            timeout=timeout,
            stream=True,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"统一检索流式接口请求失败，无法连接到 RAG API: {unified_stream_api_url}\n"
            f"原始错误: {exc}"
        ) from exc

    content_type = response.headers.get("Content-Type", "")
    if response.status_code != 200:
        detail = response.text[:2000]
        if _looks_like_html_response(content_type, detail):
            detail = _build_endpoint_hint(unified_stream_api_url, detail)
        raise RuntimeError(
            f"统一检索流式接口请求失败，状态码: {response.status_code}\n{detail}"
        )

    output_parts: list[str] = []
    final_stat: dict[str, Any] = {}

    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue

        if "[DONE]" in line:
            break

        data = json.loads(line[6:])
        chunk = data.get("content", "")
        if chunk:
            output_parts.append(chunk)
            print(chunk, end="", flush=True)

        extra_stat = {k: v for k, v in data.items() if k != "content"}
        if extra_stat:
            final_stat.update(extra_stat)

    print()
    full_answer = "".join(output_parts)
    return {
        "full_answer": full_answer,
        "stat": final_stat,
    }


if __name__ == "__main__":
    query = os.getenv(
        "RAG_SAMPLE_QUERY",
        "Shikra-Embedding-V3.7.0 相比旧版本主要优化了哪些能力？",
    )
    fused_top_k = int(os.getenv("RAG_FUSED_TOP_K", "12"))
    rrf_k = int(os.getenv("RAG_RRF_K", "60"))
    route_with_llm = os.getenv("RAG_ROUTE_WITH_LLM", "true").lower() in {"1", "true", "yes"}

    print(f"使用 unified 流式接口: {unified_stream_api_url}")
    print(f"问题: {query}")
    print("流式结果:")

    result = query_unified_stream(
        query=query,
        fused_top_k=fused_top_k,
        rrf_k=rrf_k,
        route_with_llm=route_with_llm,
    )

    print("\n统计信息:")
    print(json.dumps(result.get("stat", {}), ensure_ascii=False, indent=2))
