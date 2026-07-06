#!/usr/bin/env python3
"""RAG 查询：默认调用 Playbook 增强问答接口，兼容旧 SSE RAG 响应。"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from typing import Any

import requests
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import RequestsDependencyWarning
from requests.exceptions import RequestException
from requests.exceptions import Timeout as RequestsTimeout

warnings.filterwarnings("ignore", category=RequestsDependencyWarning)

DEFAULT_BASE_URL = "http://127.0.0.1:6062/api/v1/playbook/query"


def _normalize_kb_score(value: Any) -> float:
    """将 knowledge_base_fully_answered 规范为 [0, 1]；与 demo.agent 判定逻辑一致。"""
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, score))
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"true", "1", "yes", "y"}:
            return 1.0
        if raw in {"false", "0", "no", "n", ""}:
            return 0.0
        try:
            return _normalize_kb_score(float(raw))
        except ValueError:
            return 0.0
    return 0.0


def _build_payload(query: str, *, stream: bool = True) -> dict[str, Any]:
    return {
        "query": query.strip(),
        "stream": bool(stream),
        "use_playbook": True,
        "playbook_top_k": int(os.environ.get("RAG_PLAYBOOK_TOP_K", "8")),
        "top_k": int(os.environ.get("RAG_TOP_K", "12")),
    }


def _extract_history_assistant_text(obj: Any) -> list[str]:
    out: list[str] = []
    if not isinstance(obj, dict):
        return out
    history = obj.get("history")
    if not isinstance(history, list):
        return out
    for item in history:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        if role != "assistant":
            continue
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            out.append(content)
    return out


def _normalize_refs(maybe_refs: Any) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if not isinstance(maybe_refs, list):
        return out
    seen: set[str] = set()
    for item in maybe_refs:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(
            {
                "doc_name": str(item.get("doc_name") or "").strip(),
                "url": url,
            }
        )
    return out


def _normalize_unified_refs(maybe_evidences: Any) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if not isinstance(maybe_evidences, list):
        return out
    seen: set[str] = set()
    for item in maybe_evidences:
        if not isinstance(item, dict):
            continue
        payload = item.get("payload")
        payload_dict = payload if isinstance(payload, dict) else {}
        url = str(
            payload_dict.get("url")
            or payload_dict.get("reference")
            or payload_dict.get("link")
            or ""
        ).strip()
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(
            {
                "doc_name": str(
                    item.get("title")
                    or payload_dict.get("doc_name")
                    or payload_dict.get("title")
                    or ""
                ).strip(),
                "url": url,
            }
        )
    return out


def _extract_text_piece(packet: dict[str, Any], current_answer: str) -> str:
    for key in ("delta", "token", "chunk", "content", "text", "answer_delta", "answer_chunk"):
        v = packet.get(key)
        if isinstance(v, str) and v:
            return v
    answer = packet.get("answer")
    if isinstance(answer, str) and answer:
        if answer.startswith(current_answer):
            return answer[len(current_answer) :]
        return answer
    return ""


def _consume_sse_response(resp: requests.Response) -> dict[str, Any]:
    answer_parts: list[str] = []
    final_packet: dict[str, Any] = {}
    for raw_line in resp.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue
        line = str(raw_line).strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            packet = json.loads(payload)
        except json.JSONDecodeError:
            answer_parts.append(payload)
            continue
        if not isinstance(packet, dict):
            continue
        final_packet = packet
        piece = _extract_text_piece(packet, "".join(answer_parts))
        if piece:
            answer_parts.append(piece)

    answer = "".join(answer_parts).strip()
    if not answer and isinstance(final_packet.get("answer"), str):
        answer = str(final_packet.get("answer") or "").strip()

    refs = _normalize_refs(final_packet.get("reference"))
    if not refs:
        refs = _normalize_unified_refs(final_packet.get("fused_evidences"))

    retrieved_chunks = (
        [x for x in final_packet.get("retrieved_chunks", []) if isinstance(x, dict)]
        if isinstance(final_packet.get("retrieved_chunks"), list)
        else []
    )
    return {
        "success": bool(answer),
        "answer": answer,
        "reference": refs,
        "retrieved_chunks": retrieved_chunks,
        "knowledge_base_fully_answered": _normalize_kb_score(
            final_packet.get("knowledge_base_fully_answered")
        ),
        "raw_last_packet": final_packet,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Query RAG service; print answer to stdout")
    parser.add_argument("--query", required=True, help="User question text")
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to write merged JSON response (UTF-8)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("RAG_QUERY_URL", DEFAULT_BASE_URL),
        help=(
            "RAG query endpoint 地址 "
            "(默认: 环境变量 RAG_QUERY_URL，否则为内置地址)"
        ),
    )
    parser.add_argument(
        "--url",
        default=None,
        help="已弃用；若指定则覆盖 --base-url",
    )
    args = parser.parse_args()

    endpoint = (args.url or args.base_url or "").strip()
    if not endpoint:
        print("Error: empty --base-url / --url", file=sys.stderr)
        sys.exit(2)
    if not args.query.strip():
        print("Error: empty --query", file=sys.stderr)
        sys.exit(2)

    wants_stream = not endpoint.rstrip("/").endswith("/playbook/query")
    payload = _build_payload(args.query, stream=wants_stream)
    try:
        resp = requests.post(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=wants_stream,
            timeout=(10, 300),
        )
    except RequestsConnectionError as e:
        print(f"RAG connection failed: {e}", file=sys.stderr)
        print(
            "\n常见原因：目标地址上没有进程在监听（服务未启动、端口不对或防火墙拦截）。\n"
            f"  当前请求 URL: {endpoint}\n"
            "  处理办法：启动 RAG HTTP 服务后重试；若地址不同，请设置环境变量\n"
            "    export RAG_QUERY_URL='http://<host>:6062/api/v1/playbook/query'\n"
            "  或传入: python3 .../run_rag.py --query '...' --base-url 'http://...'",
            file=sys.stderr,
        )
        sys.exit(1)
    except RequestsTimeout as e:
        print(f"RAG request timed out: {e}", file=sys.stderr)
        sys.exit(1)
    except RequestException as e:
        print(f"RAG request failed: {e}", file=sys.stderr)
        sys.exit(1)

    if resp.status_code != 200:
        body = resp.text[:2000]
        print(f"RAG HTTP {resp.status_code}: {body}", file=sys.stderr)
        sys.exit(1)

    ctype = str(resp.headers.get("Content-Type") or "").lower()
    if "text/event-stream" in ctype:
        result = _consume_sse_response(resp)
    else:
        try:
            result = resp.json()
        except json.JSONDecodeError:
            print(f"Invalid JSON response: {resp.text[:2000]}", file=sys.stderr)
            sys.exit(1)
        if not isinstance(result, dict):
            print("RAG response is not JSON object", file=sys.stderr)
            sys.exit(1)

    if args.out:
        out_path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    if not result.get("success", True):
        print(result.get("answer") or json.dumps(result, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)

    answer = str(result.get("answer") or "").strip()
    if not answer:
        hist = _extract_history_assistant_text(result)
        if hist:
            answer = "\n".join(hist).strip()
    if not answer:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        sys.exit(0)

    print(answer)
    refs = _normalize_refs(result.get("reference"))
    if not refs:
        refs = _normalize_unified_refs(result.get("fused_evidences"))
    if refs:
        print("")
        print("更多详情请参考以下链接：")
        for i, item in enumerate(refs, start=1):
            print(f"{i}. {item['url']}")


if __name__ == "__main__":
    main()
