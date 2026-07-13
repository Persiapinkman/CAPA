#!/usr/bin/env python3
"""RAG 查询：调用 unified_query，仅发送 query，其他参数使用接口默认值。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import requests
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import RequestException
from requests.exceptions import Timeout as RequestsTimeout

DEFAULT_BASE_URL = "http://10.111.32.254:6060/api/v1/rag/chat_engine/unified_query"


def _build_payload(query: str) -> dict[str, Any]:
    return {
        "query": query.strip(),
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

    payload = _build_payload(args.query)
    try:
        resp = requests.post(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=300,
        )
    except RequestsConnectionError as e:
        print(f"RAG connection failed: {e}", file=sys.stderr)
        print(
            "\n常见原因：目标地址上没有进程在监听（服务未启动、端口不对或防火墙拦截）。\n"
            f"  当前请求 URL: {endpoint}\n"
            "  处理办法：启动 RAG HTTP 服务后重试；若地址不同，请设置环境变量\n"
            "    export RAG_QUERY_URL='http://<host>:<port>/api/v1/rag/chat_engine/unified_query'\n"
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

    try:
        result: Any = resp.json()
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
