import json
import os
from typing import Any, Literal

import requests


def _default_adela_chat_api_url() -> str:
    api_base = os.getenv("SWIFT_RAG_API_BASE_URL")
    if api_base:
        return f"{api_base.rstrip('/')}/rag/chat_engine/adela_query"

    try:
        from src.core.config import get_settings

        settings = get_settings()
        return (
            f"http://127.0.0.1:{settings.PORT}{settings.API_V1_STR}"
            "/rag/chat_engine/adela_query"
        )
    except Exception:
        return "http://127.0.0.1:6060/api/v1/rag/chat_engine/adela_query"


def _default_adela_data_path() -> str:
    try:
        from src.core.config import get_settings

        return get_settings().ADELA_DATA_JSONL_PATH
    except Exception:
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data_source",
            "adela",
            "adela_release_records.jsonl",
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
            "`http://127.0.0.1:6060/api/v1/rag/chat_engine/adela_query`。"
        )
    return f"接口返回了非预期响应，请检查服务地址是否正确: {url}"


adela_chat_api_url = os.getenv(
    "SWIFT_RAG_ADELA_CHAT_API_URL",
    _default_adela_chat_api_url(),
)
adela_data_path = os.getenv(
    "SWIFT_RAG_ADELA_DATA_PATH",
    _default_adela_data_path(),
)


def _resolve_adela_chat_api_url(api_base_url: str | None) -> str:
    if api_base_url:
        return f"{api_base_url.rstrip('/')}/rag/chat_engine/adela_query"
    return adela_chat_api_url


def _invoke_adela_chat(
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
            f"adela 问答接口请求失败，无法连接到 RAG API: {url}\n"
            f"原始错误: {exc}"
        ) from exc

    content_type = response.headers.get("Content-Type", "")
    if response.status_code != 200:
        detail = response.text[:2000]
        if _looks_like_html_response(content_type, detail):
            detail = _build_endpoint_hint(url, detail)
        raise RuntimeError(
            f"adela 问答接口请求失败，状态码: {response.status_code}\n{detail}"
        )

    try:
        return response.json()
    except ValueError as exc:
        detail = response.text[:2000]
        if _looks_like_html_response(content_type, detail):
            detail = _build_endpoint_hint(url, detail)
        raise RuntimeError(
            "adela 问答接口返回的不是合法 JSON。\n"
            f"{detail}"
        ) from exc


def query_adela_chat(
    query: str,
    top_k: int = 20,
    similarity_threshold: float = 0.15,
    retrieval_method: Literal["keyword", "vector", "hybrid"] = "hybrid",
    embedding_models: list[str] | None = None,
    searchable_fields: list[str] | None = None,
    return_fields: list[str] | None = None,
    api_base_url: str | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    if embedding_models is None:
        embedding_models = ["bge_m3"]

    payload: dict[str, Any] = {
        "query": query,
        "retrieval_method": retrieval_method,
        "top_k": top_k,
        "similarity_threshold": similarity_threshold,
        "data_path": adela_data_path,
        "embedding_models": embedding_models,
    }
    if searchable_fields:
        payload["searchable_fields"] = searchable_fields
    if return_fields:
        payload["return_fields"] = return_fields

    url = _resolve_adela_chat_api_url(api_base_url)
    return _invoke_adela_chat(payload=payload, url=url, timeout=timeout)


if __name__ == "__main__":
    query = os.getenv(
        "RAG_SAMPLE_QUERY",
        "人脸识别场景，adela上部署的模型记录能达到什么性能和精度?",
    )

    print(f"使用 adela 问答接口: {adela_chat_api_url}")
    print(f"使用 adela 数据: {adela_data_path}")

    result = query_adela_chat(query=query)

    print("命中记录:")
    print(
        json.dumps(
            result.get("matched_records", []),
            ensure_ascii=False,
            indent=2,
        )
    )

    print("reference:")
    print(
        json.dumps(
            result.get("reference", []),
            ensure_ascii=False,
            indent=2,
        )
    )

    print("问答结果:")
    print(result.get("answer", ""))

    print("耗时信息:")
    print(
        json.dumps(
            result.get("timings", {}),
            ensure_ascii=False,
            indent=2,
        )
    )
