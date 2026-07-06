"""GBrain RAG HTTP 客户端：支持 Playbook 兼容接口与 unified API（见 skills/rag-retrieve-answer/GBRAIN-RAG-API-USAGE.md）。

配置优先级（高 → 低）：
1. `configure_rag(...)` 显式传入的参数
2. 环境变量（如 `DEMO_RAG_API_MODE`）
3. 模块内置默认值
"""

from __future__ import annotations

import os
from typing import Any

import requests

RAG_API_MODE_PLAYBOOK = "playbook"
RAG_API_MODE_UNIFIED = "unified"

_DEFAULT_PLAYBOOK_QUERY_URL = "http://127.0.0.1:6062/api/v1/playbook/query"
_DEFAULT_GBRAIN_BASE_URL = "http://127.0.0.1:6061/api/v1/rag"
_DEFAULT_RAG_API_MODE = RAG_API_MODE_PLAYBOOK

# 由 configure_rag() 写入；未设置时回退到环境变量/默认值
_runtime: dict[str, Any] = {}


def configure_rag(
    *,
    api_mode: str | None = None,
    playbook_query_url: str | None = None,
    gbrain_base_url: str | None = None,
    unified_query_url: str | None = None,
    unified_retrieve_url: str | None = None,
    include_full_documents: bool | None = None,
    top_k: int | None = None,
    playbook_top_k: int | None = None,
    retrieval_method: str | None = None,
    sync_to_environ: bool = False,
) -> None:
    """在代码中设置 RAG 参数（推荐在 demo_server.py 启动前调用一次）。

    sync_to_environ=True 时，同时写入 os.environ，便于子进程或其它模块读取。
    """
    mapping: dict[str, Any] = {}
    if api_mode is not None:
        mapping["api_mode"] = str(api_mode).strip().lower()
    if playbook_query_url is not None:
        mapping["playbook_query_url"] = str(playbook_query_url).strip()
    if gbrain_base_url is not None:
        mapping["gbrain_base_url"] = str(gbrain_base_url).strip().rstrip("/")
    if unified_query_url is not None:
        mapping["unified_query_url"] = str(unified_query_url).strip().rstrip("/")
    if unified_retrieve_url is not None:
        mapping["unified_retrieve_url"] = str(unified_retrieve_url).strip().rstrip("/")
    if include_full_documents is not None:
        mapping["include_full_documents"] = bool(include_full_documents)
    if top_k is not None:
        mapping["top_k"] = max(1, int(top_k))
    if playbook_top_k is not None:
        mapping["playbook_top_k"] = max(1, int(playbook_top_k))
    if retrieval_method is not None:
        mapping["retrieval_method"] = str(retrieval_method).strip() or "hybrid"
    _runtime.update(mapping)

    if not sync_to_environ:
        return
    env_pairs = {
        "api_mode": ("DEMO_RAG_API_MODE", mapping.get("api_mode")),
        "playbook_query_url": ("RAG_BASE_URL", mapping.get("playbook_query_url")),
        "gbrain_base_url": ("GBRAIN_RAG_BASE_URL", mapping.get("gbrain_base_url")),
        "unified_query_url": ("RAG_UNIFIED_QUERY_URL", mapping.get("unified_query_url")),
        "unified_retrieve_url": ("RAG_UNIFIED_RETRIEVE_URL", mapping.get("unified_retrieve_url")),
        "include_full_documents": (
            "DEMO_RAG_INCLUDE_FULL_DOCUMENTS",
            "1" if mapping.get("include_full_documents") else "0"
            if "include_full_documents" in mapping
            else None,
        ),
        "top_k": ("RAG_TOP_K", str(mapping["top_k"]) if "top_k" in mapping else None),
        "playbook_top_k": (
            "RAG_PLAYBOOK_TOP_K",
            str(mapping["playbook_top_k"]) if "playbook_top_k" in mapping else None,
        ),
        "retrieval_method": ("RAG_RETRIEVAL_METHOD", mapping.get("retrieval_method")),
    }
    for _key, (env_name, value) in env_pairs.items():
        if value is not None:
            os.environ[env_name] = str(value)


def get_rag_config() -> dict[str, Any]:
    """返回当前生效的 RAG 配置快照（便于日志/调试）。"""
    return {
        "api_mode": rag_api_mode(),
        "playbook_query_url": _playbook_query_url_value(),
        "gbrain_base_url": _gbrain_base_url_value(),
        "unified_query_url": unified_query_url(),
        "unified_retrieve_url": unified_retrieve_url(),
        "include_full_documents": include_full_documents_enabled(),
        "top_k": rag_top_k(),
        "playbook_top_k": playbook_top_k(),
        "retrieval_method": retrieval_method(),
        "runtime_overrides": dict(_runtime),
    }


def _cfg(key: str) -> Any:
    return _runtime.get(key)


def _playbook_query_url_value() -> str:
    explicit = _cfg("playbook_query_url")
    if explicit:
        return str(explicit).strip()
    return (
        os.environ.get("RAG_BASE_URL")
        or os.environ.get("RAG_QUERY_URL")
        or _DEFAULT_PLAYBOOK_QUERY_URL
    ).strip()


def _gbrain_base_url_value() -> str:
    explicit = _cfg("gbrain_base_url")
    if explicit:
        return str(explicit).strip().rstrip("/")
    return (
        os.environ.get("GBRAIN_RAG_BASE_URL")
        or os.environ.get("RAG_UNIFIED_BASE_URL")
        or _DEFAULT_GBRAIN_BASE_URL
    ).strip().rstrip("/")


# 兼容旧代码：导入时快照（configure 后请用 get_rag_config() 或下方函数）
RAG_PLAYBOOK_QUERY_URL = _playbook_query_url_value()
GBRAIN_RAG_BASE_URL = _gbrain_base_url_value()


def rag_api_mode() -> str:
    explicit = _cfg("api_mode")
    raw = str(explicit or os.environ.get("DEMO_RAG_API_MODE") or _DEFAULT_RAG_API_MODE).strip().lower()
    if raw in {RAG_API_MODE_UNIFIED, "gbrain", "unified_retrieve"}:
        return RAG_API_MODE_UNIFIED
    return RAG_API_MODE_PLAYBOOK


def include_full_documents_enabled() -> bool:
    explicit = _cfg("include_full_documents")
    if explicit is not None:
        return bool(explicit)
    return str(os.environ.get("DEMO_RAG_INCLUDE_FULL_DOCUMENTS", "1")).strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def rag_top_k() -> int:
    explicit = _cfg("top_k")
    if explicit is not None:
        return max(1, int(explicit))
    try:
        return max(1, int(os.environ.get("RAG_TOP_K", "12")))
    except (TypeError, ValueError):
        return 12


def playbook_top_k() -> int:
    explicit = _cfg("playbook_top_k")
    if explicit is not None:
        return max(1, int(explicit))
    try:
        return max(1, int(os.environ.get("RAG_PLAYBOOK_TOP_K", "8")))
    except (TypeError, ValueError):
        return 8


def retrieval_method() -> str:
    explicit = _cfg("retrieval_method")
    if explicit:
        return str(explicit).strip() or "hybrid"
    return str(os.environ.get("RAG_RETRIEVAL_METHOD", "hybrid")).strip() or "hybrid"


def _playbook_base() -> str:
    base = _playbook_query_url_value().rstrip("/")
    if base.endswith("/query"):
        return base[: -len("/query")]
    if base.endswith("/retrieve"):
        return base[: -len("/retrieve")]
    if base.endswith("/feedback"):
        return base[: -len("/feedback")]
    return base


def playbook_query_url() -> str:
    base = _playbook_base()
    query = _playbook_query_url_value().rstrip("/")
    if query.endswith("/query"):
        return query
    return f"{base}/query"


def playbook_retrieve_url() -> str:
    base = _playbook_base()
    return f"{base}/retrieve"


def playbook_feedback_url() -> str:
    base = _playbook_base()
    return f"{base}/feedback"


def unified_retrieve_url() -> str:
    explicit = _cfg("unified_retrieve_url") or str(os.environ.get("RAG_UNIFIED_RETRIEVE_URL") or "").strip()
    if explicit:
        return explicit.rstrip("/")
    return f"{_gbrain_base_url_value()}/chat_engine/unified_retrieve"


def unified_query_url() -> str:
    explicit = _cfg("unified_query_url") or str(os.environ.get("RAG_UNIFIED_QUERY_URL") or "").strip()
    if explicit:
        return explicit.rstrip("/")
    return f"{_gbrain_base_url_value()}/chat_engine/unified_query"


def build_playbook_query_payload(query: str, *, stream: bool = False) -> dict[str, Any]:
    return {
        "query": str(query or "").strip(),
        "stream": bool(stream),
        "top_k": rag_top_k(),
        "use_playbook": True,
        "playbook_top_k": playbook_top_k(),
    }


def build_playbook_retrieve_payload(query: str) -> dict[str, Any]:
    payload = build_playbook_query_payload(query, stream=False)
    payload["include_full_documents"] = include_full_documents_enabled()
    return payload


def build_unified_retrieve_payload(
    query: str,
    *,
    top_k: int | None = None,
    include_full_documents: bool | None = None,
) -> dict[str, Any]:
    return {
        "query": str(query or "").strip(),
        "top_k": rag_top_k() if top_k is None else max(1, int(top_k)),
        "retrieval_method": retrieval_method(),
        "include_full_documents": (
            include_full_documents_enabled()
            if include_full_documents is None
            else bool(include_full_documents)
        ),
    }


def build_unified_query_payload(
    query: str,
    *,
    stream: bool = False,
    top_k: int | None = None,
    include_full_documents: bool | None = None,
) -> dict[str, Any]:
    payload = build_unified_retrieve_payload(
        query,
        top_k=top_k,
        include_full_documents=include_full_documents,
    )
    payload["stream"] = bool(stream)
    return payload


def post_rag_json(
    url: str,
    payload: dict[str, Any],
    *,
    stream: bool = False,
    timeout: tuple[int, int] = (10, 300),
) -> dict[str, Any]:
    resp = requests.post(
        str(url or "").strip(),
        json=payload,
        headers={"Content-Type": "application/json"},
        stream=bool(stream),
        timeout=timeout,
    )
    if resp.status_code != 200:
        body = (resp.text or "")[:2000]
        raise requests.HTTPError(f"HTTP {resp.status_code}: {body}", response=resp)
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError("RAG response is not a JSON object")
    return data


def extract_evidences(data: dict) -> list[dict]:
    for key in ("evidences", "retrieved_chunks", "fused_evidences"):
        value = data.get(key)
        if isinstance(value, list):
            chunks = [item for item in value if isinstance(item, dict)]
            if chunks:
                return chunks
    return []


def extract_full_documents(data: dict) -> list[dict]:
    value = data.get("full_documents")
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def extract_answer(data: dict) -> str:
    for key in ("answer", "content", "text"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def rag_request_label(*, for_query: bool = False) -> str:
    mode = rag_api_mode()
    action = "query" if for_query else "retrieve"
    if mode == RAG_API_MODE_UNIFIED:
        url = unified_query_url() if for_query else unified_retrieve_url()
        return f"unified_{action}:{url}"
    url = playbook_query_url() if for_query else playbook_retrieve_url()
    return f"playbook_{action}:{url}"


def debug_request_blob(
    *,
    for_query: bool = False,
    query: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mode = rag_api_mode()
    if mode == RAG_API_MODE_UNIFIED:
        payload = (
            build_unified_query_payload(query, stream=False)
            if for_query
            else build_unified_retrieve_payload(query)
        )
        url = unified_query_url() if for_query else unified_retrieve_url()
    else:
        payload = (
            build_playbook_query_payload(query, stream=False)
            if for_query
            else build_playbook_retrieve_payload(query)
        )
        url = playbook_query_url() if for_query else playbook_retrieve_url()
    blob: dict[str, Any] = {"api_mode": mode, "url": url, "payload": payload}
    if extra:
        blob.update(extra)
    return blob
