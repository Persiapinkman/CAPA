import json
import os
from typing import Any

import requests


def _default_unified_chat_api_url() -> str:
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


unified_chat_api_url = os.getenv(
    "SWIFT_RAG_UNIFIED_CHAT_API_URL",
    _default_unified_chat_api_url(),
)


def _resolve_unified_chat_api_url(api_base_url: str | None) -> str:
    if api_base_url:
        return f"{api_base_url.rstrip('/')}/rag/chat_engine/unified_query"
    return unified_chat_api_url


def _invoke_unified_chat(
    payload: dict[str, Any],
    url: str,
    timeout: int,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"统一检索问答接口请求失败，无法连接到 RAG API: {url}\n"
            f"原始错误: {exc}"
        ) from exc

    content_type = response.headers.get("Content-Type", "")
    if response.status_code != 200:
        detail = response.text[:2000]
        if _looks_like_html_response(content_type, detail):
            detail = _build_endpoint_hint(url, detail)
        raise RuntimeError(
            f"统一检索问答接口请求失败，状态码: {response.status_code}\n{detail}"
        )

    try:
        return response.json()
    except ValueError as exc:
        detail = response.text[:2000]
        if _looks_like_html_response(content_type, detail):
            detail = _build_endpoint_hint(url, detail)
        raise RuntimeError(
            "统一检索问答接口返回的不是合法 JSON。\n"
            f"{detail}"
        ) from exc


def query_unified_chat(
    query: str,
    fused_top_k: int = 12,
    rrf_k: int = 60,
    stream: bool = False,
    route_with_llm: bool = True,
    api_base_url: str | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": query,
        "fused_top_k": fused_top_k,
        "rrf_k": rrf_k,
        "stream": stream,
        "route_with_llm": route_with_llm,
    }
    url = _resolve_unified_chat_api_url(api_base_url)
    return _invoke_unified_chat(payload=payload, url=url, timeout=timeout)


if __name__ == "__main__":
    query = os.getenv(
        "RAG_SAMPLE_QUERY",
        "文搜图特征维度是多少",
    )

    print(f"使用统一检索问答接口: {unified_chat_api_url}")
    result = query_unified_chat(query=query)

    print("路由计划:")
    print(json.dumps(result.get("route_plan", {}), ensure_ascii=False, indent=2))

    print("融合证据:")
    print(json.dumps(result.get("fused_evidences", []), ensure_ascii=False, indent=2))

    print("来源状态:")
    print(json.dumps(result.get("source_status", []), ensure_ascii=False, indent=2))

    print("问答结果:")
    print(result.get("answer", ""))

    print("耗时信息:")
    print(json.dumps(result.get("timings", {}), ensure_ascii=False, indent=2))
