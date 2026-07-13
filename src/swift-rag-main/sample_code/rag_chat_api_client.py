import json
import os
from typing import Any, Literal

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
            (os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            + "/data_source/embedding_artifacts/documents/milvus_data_source_evoqwen_3b.db"
        )


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


chat_api_url = os.getenv("SWIFT_RAG_CHAT_API_URL", _default_chat_api_url())
milvus_uri = os.getenv("SWIFT_RAG_MILVUS_URI", _default_milvus_uri())
collection_name = os.getenv("SWIFT_RAG_COLLECTION_NAME", "llamacollection")


def _resolve_chat_api_url(api_base_url: str | None) -> str:
    if api_base_url:
        return f"{api_base_url.rstrip('/')}/rag/chat_engine/query"
    return chat_api_url


def _invoke_rag_chat(
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
            f"问答接口请求失败，无法连接到 RAG API: {url}\n"
            f"原始错误: {exc}"
        ) from exc

    content_type = response.headers.get("Content-Type", "")
    if response.status_code != 200:
        detail = response.text[:2000]
        if _looks_like_html_response(content_type, detail):
            detail = _build_endpoint_hint(url, detail)
        raise RuntimeError(
            f"问答接口请求失败，状态码: {response.status_code}\n{detail}"
        )

    try:
        return response.json()
    except ValueError as exc:
        detail = response.text[:2000]
        if _looks_like_html_response(content_type, detail):
            detail = _build_endpoint_hint(url, detail)
        raise RuntimeError(
            "问答接口返回的不是合法 JSON。\n"
            f"{detail}"
        ) from exc


def query_rag_chat(
    query: str,
    top_k: int = 5,
    similarity_threshold: float = 0.5,
    retrieval_method: Literal["vector", "bm25", "hybrid"] = "hybrid",
    embedding_models: list[str] | None = None,
    filter_expr: str | None = None,
    api_base_url: str | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    if embedding_models is None:
        embedding_models = ["bge_m3", "EvoQwen2.5-VL-Retriever-3B-v1"]

    payload = {
        "query": query,
        "retrieval_method": retrieval_method,
        "top_k": top_k,
        "similarity_threshold": similarity_threshold,
        "uri": milvus_uri,
        "collection_name": collection_name,
        "embedding_models": embedding_models,
    }
    if filter_expr:
        payload["filter"] = filter_expr

    url = _resolve_chat_api_url(api_base_url)
    return _invoke_rag_chat(payload=payload, url=url, timeout=timeout)


def single_step_rag_tool(
    query: str,
    retrieval_method: Literal["vector", "bm25", "hybrid"] = "hybrid",
    top_k: int = 5,
    similarity_threshold: float = 0.5,
    api_base_url: str | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    """
    单步 RAG 工具函数（兼容原 single_step_rag_tool.py）。
    """
    data = query_rag_chat(
        query=query,
        retrieval_method=retrieval_method,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        api_base_url=api_base_url,
        timeout=timeout,
    )
    return {
        "success": bool(data.get("success", True)),
        "query": data.get("query", query),
        "answer": data.get("answer", ""),
        "reference": data.get("reference", []),
        "timings": data.get("timings", {}),
        "retrieved_chunks": data.get("retrieved_chunks", []),
        "message": data.get("message"),
    }


if __name__ == "__main__":
    query = os.getenv(
        "RAG_SAMPLE_QUERY",
        "安全绳检测用到了哪个模型？",
    )

    print(f"使用问答接口: {chat_api_url}")
    print(f"使用向量库: {milvus_uri}")

    result = query_rag_chat(query)

    print("检索结果:")
    print(
        json.dumps(
            result.get("retrieved_chunks", []),
            ensure_ascii=False,
            indent=2,
        )
    )

    print("问答结果:")
    print(result.get("answer", ""))

    print("来源链接:")
    print(
        json.dumps(
            result.get("reference", []),
            ensure_ascii=False,
            indent=2,
        )
    )
