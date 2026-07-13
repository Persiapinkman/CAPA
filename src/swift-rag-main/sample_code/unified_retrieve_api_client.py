import json
import os
from pathlib import Path
from typing import Any

import requests


def _default_unified_retrieve_api_url() -> str:
    api_base = os.getenv("SWIFT_RAG_API_BASE_URL")
    if api_base:
        return f"{api_base.rstrip('/')}/rag/chat_engine/unified_retrieve"

    try:
        from src.core.config import get_settings

        settings = get_settings()
        return (
            f"http://127.0.0.1:{settings.PORT}{settings.API_V1_STR}"
            "/rag/chat_engine/unified_retrieve"
        )
    except Exception:
        return "http://127.0.0.1:6060/api/v1/rag/chat_engine/unified_retrieve"


def _looks_like_html_response(content_type: str, text: str) -> bool:
    return "text/html" in content_type.lower() or text.lstrip().lower().startswith(
        "<!doctype html"
    )


def _build_endpoint_hint(url: str, response_text: str) -> str:
    if "Jupyter Server" in response_text or "_xsrf" in response_text:
        return (
            f"当前请求打到了 Jupyter 服务而不是 RAG API: {url}\n"
            "请检查端口是否写错了。RAG 服务默认应为 "
            "`http://127.0.0.1:6060/api/v1/rag/chat_engine/unified_retrieve`。"
        )
    return f"接口返回了非预期响应，请检查服务地址是否正确: {url}"


unified_retrieve_api_url = os.getenv(
    "SWIFT_RAG_UNIFIED_RETRIEVE_API_URL",
    _default_unified_retrieve_api_url(),
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_unified_retrieve_api_url(api_base_url: str | None) -> str:
    if api_base_url:
        return f"{api_base_url.rstrip('/')}/rag/chat_engine/unified_retrieve"
    return unified_retrieve_api_url


def _resolve_local_json_path(entity: dict[str, Any]) -> Path | None:
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
        candidate = PROJECT_ROOT / "data_source" / "adela" / path
        if candidate.exists():
            return candidate.resolve()

    return None


def _attach_adela_json_links(result: dict[str, Any]) -> None:
    fused_evidences = result.get("fused_evidences", [])
    if not isinstance(fused_evidences, list):
        return

    for evidence in fused_evidences:
        if not isinstance(evidence, dict):
            continue
        if evidence.get("source_type") != "adela":
            continue

        payload = evidence.get("payload")
        if not isinstance(payload, dict):
            continue

        if payload.get("json_link"):
            continue

        entity = payload.get("entity")
        if not isinstance(entity, dict):
            continue

        resolved = _resolve_local_json_path(entity)
        if resolved is None:
            continue
        payload["json_path"] = str(resolved)
        payload["json_link"] = resolved.as_uri()


def _invoke_unified_retrieve(
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
            f"统一检索接口请求失败，无法连接到 RAG API: {url}\n"
            f"原始错误: {exc}"
        ) from exc

    content_type = response.headers.get("Content-Type", "")
    if response.status_code != 200:
        detail = response.text[:2000]
        if _looks_like_html_response(content_type, detail):
            detail = _build_endpoint_hint(url, detail)
        raise RuntimeError(
            f"统一检索接口请求失败，状态码: {response.status_code}\n{detail}"
        )

    try:
        return response.json()
    except ValueError as exc:
        detail = response.text[:2000]
        if _looks_like_html_response(content_type, detail):
            detail = _build_endpoint_hint(url, detail)
        raise RuntimeError(
            "统一检索接口返回的不是合法 JSON。\n"
            f"{detail}"
        ) from exc


def query_unified_retrieve(
    query: str,
    source_types: list[str] | None = None,
    fused_top_k: int = 12,
    rrf_k: int = 60,
    api_base_url: str | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": query,
        "fused_top_k": fused_top_k,
        "rrf_k": rrf_k,
    }
    if source_types is not None:
        payload["source_types"] = source_types
    url = _resolve_unified_retrieve_api_url(api_base_url)
    result = _invoke_unified_retrieve(payload=payload, url=url, timeout=timeout)
    _attach_adela_json_links(result)
    return result


if __name__ == "__main__":
    query = os.getenv(
        "RAG_SAMPLE_QUERY",
        "人脸识别模型1:N和1:n的测试精度是怎样",
    )
    source_types_env = os.getenv("RAG_SOURCE_TYPES", "table,adela").strip()
    source_types = [item.strip() for item in source_types_env.split(",") if item.strip()]

    print(f"使用统一检索接口: {unified_retrieve_api_url}")
    print(f"source_types: {source_types}")
    result = query_unified_retrieve(
        query=query,
        source_types=source_types or None,
    )

    print("实际检索数据源:")
    print(json.dumps(result.get("selected_sources", []), ensure_ascii=False, indent=2))

    print("融合证据:")
    print(json.dumps(result.get("fused_evidences", []), ensure_ascii=False, indent=2))

    print("adela JSON 链接:")
    for evidence in result.get("fused_evidences", []):
        if evidence.get("source_type") != "adela":
            continue
        payload = evidence.get("payload", {}) or {}
        json_link = payload.get("json_link")
        if not json_link:
            continue
        title = evidence.get("title") or evidence.get("evidence_id")
        print(f"- {title}: {json_link}")

    print("来源状态:")
    print(json.dumps(result.get("source_status", []), ensure_ascii=False, indent=2))

    print("耗时信息:")
    print(json.dumps(result.get("timings", {}), ensure_ascii=False, indent=2))
