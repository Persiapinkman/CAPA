"""Non-destructive health probes for services used by the demo capabilities."""

from __future__ import annotations

import os
import time
from pathlib import Path
from urllib.parse import urlparse

import requests


DEFAULT_MODEL_GATEWAY = "http://10.111.32.253:8000/v1"
DEFAULT_QWEN_DETECTION = "http://127.0.0.1:9012/v1"
DEFAULT_PLAYBOOK_QUERY = "http://127.0.0.1:6062/api/v1/playbook/query"
DEFAULT_GBRAIN_BASE = "http://127.0.0.1:6061/api/v1/rag"
DEFAULT_FLUX_API = "https://api.apiyi.com/v1"


def _is_loopback(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1"}


def _session_for(url: str) -> requests.Session:
    session = requests.Session()
    if _is_loopback(url):
        session.trust_env = False
    return session


def _safe_error(exc: Exception, api_key: str = "") -> str:
    text = str(exc or "").strip().replace("\n", " ")[:300]
    if api_key:
        text = text.replace(api_key, "***")
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _probe_json(
    url: str,
    *,
    api_key: str = "",
    timeout: float = 5.0,
) -> tuple[dict, float]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    started = time.perf_counter()
    response = _session_for(url).get(url, headers=headers, timeout=timeout)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("health response is not a JSON object")
    return payload, elapsed_ms


def probe_openai_models(
    base_url: str,
    *,
    api_key: str,
    required_models: tuple[str, ...] = (),
    timeout: float = 5.0,
) -> dict:
    base = str(base_url or "").strip().rstrip("/")
    url = f"{base}/models"
    try:
        payload, elapsed_ms = _probe_json(url, api_key=api_key, timeout=timeout)
        rows = payload.get("data") if isinstance(payload.get("data"), list) else []
        models = sorted(
            str(item.get("id") or "").strip()
            for item in rows
            if isinstance(item, dict) and str(item.get("id") or "").strip()
        )
        missing = sorted(set(required_models) - set(models))
        return {
            "status": "online" if not missing else "degraded",
            "url": url,
            "elapsed_ms": elapsed_ms,
            "models": models,
            "missing_required_models": missing,
        }
    except Exception as exc:
        return {
            "status": "offline",
            "url": url,
            "error": _safe_error(exc, api_key),
        }


def probe_http_health(url: str, *, timeout: float = 5.0) -> dict:
    target = str(url or "").strip()
    try:
        payload, elapsed_ms = _probe_json(target, timeout=timeout)
        payload_status = str(payload.get("status") or "").strip().lower()
        nested_v2 = payload.get("v2") if isinstance(payload.get("v2"), dict) else {}
        nested_status = str(nested_v2.get("status") or "").strip().lower()
        status = (
            "degraded"
            if payload_status == "degraded"
            or nested_status == "degraded"
            or nested_v2.get("reachable") is False
            else "online"
        )
        return {
            "status": status,
            "url": target,
            "elapsed_ms": elapsed_ms,
            "response": payload,
        }
    except Exception as exc:
        return {
            "status": "offline",
            "url": target,
            "error": _safe_error(exc),
        }


def _playbook_health_url(query_url: str) -> str:
    base = str(query_url or "").strip().rstrip("/")
    for suffix in ("/query", "/retrieve", "/feedback", "/health"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return f"{base}/health"


def _gbrain_health_url(base_url: str) -> str:
    base = str(base_url or "").strip().rstrip("/")
    if base.endswith("/health"):
        return base
    return f"{base}/health"


def _read_api_key(root: Path) -> str:
    path = root / "api_key.txt"
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def probe_demo_services(
    root: Path,
    *,
    timeout: float = 5.0,
    include_flux: bool = False,
) -> dict:
    workspace = Path(root).resolve()
    llm_key = os.environ.get("DEMO_LLM_API_KEY", "token.sdc@2026")
    model_gateway = os.environ.get("DEMO_LLM_API_BASE", DEFAULT_MODEL_GATEWAY)
    qwen_base = os.environ.get("DEMO_QWEN_DETECTION_URL", DEFAULT_QWEN_DETECTION)
    playbook_query = os.environ.get(
        "RAG_BASE_URL", os.environ.get("RAG_QUERY_URL", DEFAULT_PLAYBOOK_QUERY)
    )
    gbrain_base = os.environ.get(
        "GBRAIN_RAG_BASE_URL",
        os.environ.get("RAG_UNIFIED_BASE_URL", DEFAULT_GBRAIN_BASE),
    )
    probes = {
        "model_gateway": probe_openai_models(
            model_gateway,
            api_key=llm_key,
            required_models=("Qwen3.5-4B", "Qwen3.5-35B-A3B", "Rex-Omni"),
            timeout=timeout,
        ),
        "qwen_detection": probe_openai_models(
            qwen_base,
            api_key=llm_key,
            timeout=timeout,
        ),
        "rag_playbook": probe_http_health(
            _playbook_health_url(playbook_query), timeout=timeout
        ),
        "rag_unified": probe_http_health(
            _gbrain_health_url(gbrain_base), timeout=timeout
        ),
    }
    if include_flux:
        flux_key = _read_api_key(workspace)
        if flux_key:
            probes["flux_api"] = probe_openai_models(
                os.environ.get("DEMO_API_BASE", DEFAULT_FLUX_API),
                api_key=flux_key,
                timeout=timeout,
            )
        else:
            probes["flux_api"] = {
                "status": "unconfigured",
                "error": "api_key.txt is missing or empty",
            }
    else:
        probes["flux_api"] = {
            "status": "not_probed",
            "reason": "enable include_flux for a credentialed read-only model-list probe",
        }
    return probes


def service_summary(probes: dict) -> dict[str, int]:
    counts = {"online": 0, "degraded": 0, "offline": 0, "other": 0}
    for probe in probes.values():
        status = str(probe.get("status") or "") if isinstance(probe, dict) else ""
        if status in counts:
            counts[status] += 1
        else:
            counts["other"] += 1
    return counts
