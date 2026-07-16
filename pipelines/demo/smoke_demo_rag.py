#!/usr/bin/env python3
"""Start GBrain/ACE RAG with isolated empty stores and verify their HTTP contracts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gbrain-port", type=int, default=6061)
    parser.add_argument("--ace-port", type=int, default=6062)
    parser.add_argument("--startup-timeout", type=float, default=20.0)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("/raid/zkq/artifacts/CAPA/capability_reproduction"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/demo_rag_smoke.json",
    )
    return parser.parse_args()


def _local_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    return session


def _get_json(url: str, timeout: float) -> dict[str, Any]:
    response = _local_session().get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"non-object response from {url}")
    return payload


def _post_json(url: str, body: dict, timeout: float) -> dict[str, Any]:
    response = _local_session().post(url, json=body, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"non-object response from {url}")
    return payload


def _wait_for_health(
    url: str,
    timeout: float,
    *,
    process: subprocess.Popen | None = None,
    log_path: Path | None = None,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            log_tail = ""
            if log_path is not None and log_path.exists():
                log_tail = log_path.read_text(
                    encoding="utf-8", errors="replace"
                )[-2000:]
            raise RuntimeError(
                f"service exited before becoming healthy: {url}; "
                f"exit_code={process.returncode}; log_tail={log_tail}"
            )
        try:
            return _get_json(url, timeout=min(2.0, timeout))
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {str(exc)[:240]}"
            time.sleep(0.2)
    raise TimeoutError(f"service did not become healthy: {url}; last_error={last_error}")


def _start_service(
    *,
    python: str,
    app: str,
    port: int,
    pythonpath: Path,
    env_updates: dict[str, str],
    log_path: Path,
) -> tuple[subprocess.Popen, Any]:
    env = os.environ.copy()
    env.update(env_updates)
    env["PYTHONPATH"] = str(pythonpath)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            python,
            "-m",
            "uvicorn",
            app,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=str(ROOT),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return process, log_handle


def _stop(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _score(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    args = parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.artifact_root.resolve() / f"rag-smoke-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    # Preserve the venv entry point; resolving its symlink escapes the venv.
    python = sys.executable
    gbrain_process: subprocess.Popen | None = None
    ace_process: subprocess.Popen | None = None
    handles: list[Any] = []
    try:
        gbrain_process, handle = _start_service(
            python=python,
            app="gbrain_rag.main:app",
            port=args.gbrain_port,
            pythonpath=ROOT / "src/gbrain-rag-main/src",
            env_updates={
                "GBRAIN_RAG_INDEX_DB_PATH": str(run_dir / "gbrain.sqlite3"),
                "GBRAIN_RAG_ARTIFACTS_DIR": str(run_dir / "gbrain-artifacts"),
                "GBRAIN_RAG_EMBEDDING_BACKEND": "hashing",
                "GBRAIN_RAG_EMBEDDING_MODEL": "hashing",
                "GBRAIN_RAG_EMBEDDING_MODELS": '["hashing"]',
                "GBRAIN_RAG_ENABLE_LLM_QUERY_EXPANSION": "false",
            },
            log_path=run_dir / "gbrain.log",
        )
        handles.append(handle)
        gbrain_health_url = (
            f"http://127.0.0.1:{args.gbrain_port}/api/v1/rag/health"
        )
        gbrain_health = _wait_for_health(
            gbrain_health_url,
            args.startup_timeout,
            process=gbrain_process,
            log_path=run_dir / "gbrain.log",
        )

        ace_process, handle = _start_service(
            python=python,
            app="ace_rag.main:app",
            port=args.ace_port,
            pythonpath=ROOT / "src/ace-rag-main/src",
            env_updates={
                "ACE_RAG_PLAYBOOK_DB_PATH": str(run_dir / "playbook.sqlite3"),
                "ACE_RAG_V2_BASE_URL": f"http://127.0.0.1:{args.gbrain_port}/api/v1/rag",
                "ACE_RAG_AUTO_IMPORT_SEED": "false",
            },
            log_path=run_dir / "ace.log",
        )
        handles.append(handle)
        ace_health_url = (
            f"http://127.0.0.1:{args.ace_port}/api/v1/playbook/health"
        )
        ace_health = _wait_for_health(
            ace_health_url,
            args.startup_timeout,
            process=ace_process,
            log_path=run_dir / "ace.log",
        )

        retrieve = _post_json(
            f"http://127.0.0.1:{args.gbrain_port}/api/v1/rag/chat_engine/unified_retrieve",
            {
                "query": "能力复现烟雾测试",
                "top_k": 3,
                "retrieval_method": "hybrid",
                "include_full_documents": True,
            },
            args.request_timeout,
        )
        answer = _post_json(
            f"http://127.0.0.1:{args.ace_port}/api/v1/playbook/query",
            {
                "query": "能力复现烟雾测试",
                "stream": False,
                "top_k": 3,
                "use_playbook": True,
                "playbook_top_k": 3,
            },
            args.request_timeout,
        )
        retrieved_count = int(answer.get("retrieved_count") or 0)
        kb_score = _score(answer.get("knowledge_base_fully_answered"))
        http_contract_passed = (
            isinstance(gbrain_health.get("chunks"), int)
            and isinstance(ace_health.get("playbook"), dict)
            and isinstance(retrieve.get("evidences"), list)
            and isinstance(answer.get("answer"), str)
        )
        content_ready = (
            int(gbrain_health.get("chunks") or 0) > 0
            and int((ace_health.get("playbook") or {}).get("active_items") or 0) > 0
        )
        miss_contract_passed = (
            retrieved_count == 0
            and kb_score < 0.97
            and bool(str(answer.get("answer") or "").strip())
        )
        payload = {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": (
                "content_ready"
                if http_contract_passed and content_ready
                else "contract_passed_content_missing"
                if http_contract_passed
                else "failed"
            ),
            "http_contract_passed": http_contract_passed,
            "content_ready": content_ready,
            "miss_contract_passed": miss_contract_passed,
            "gbrain": {
                "health_status": gbrain_health.get("status"),
                "chunks": gbrain_health.get("chunks", 0),
                "sources": gbrain_health.get("sources", {}),
                "retrieved_count": retrieve.get("retrieved_count", 0),
            },
            "ace": {
                "health_status": ace_health.get("status"),
                "v2_reachable": (ace_health.get("v2") or {}).get("reachable"),
                "active_playbook_items": (ace_health.get("playbook") or {}).get(
                    "active_items", 0
                ),
                "retrieved_count": retrieved_count,
                "knowledge_base_fully_answered": kb_score,
                "answer_present": bool(str(answer.get("answer") or "").strip()),
            },
            "artifacts": {
                "run_dir": str(run_dir),
                "gbrain_log": str(run_dir / "gbrain.log"),
                "ace_log": str(run_dir / "ace.log"),
            },
            "boundary": (
                "This smoke verifies HTTP and empty-index miss behavior only. "
                "The repository does not contain the production corpus or indexes."
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if not http_contract_passed or not miss_contract_passed:
            raise SystemExit(1)
    finally:
        _stop(ace_process)
        _stop(gbrain_process)
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()
