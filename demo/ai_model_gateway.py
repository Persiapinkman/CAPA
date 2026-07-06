#!/usr/bin/env python3
"""AI Model Gateway 0.1.0.

OpenAI-compatible gateway in front of a local vLLM OpenAI server.
It accepts OpenAI multimodal content lists and normalizes them for text-only
planner models such as Qwen3.5-4B.
"""

from __future__ import annotations

import argparse
import time
import uuid
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

VERSION = "0.1.0"


def _trim_data_url(value: str, limit: int = 80) -> str:
    text = str(value or "")
    if text.startswith("data:"):
        head = text.split(",", 1)[0]
        return f"{head},<base64 omitted>"
    if len(text) > limit:
        return text[:limit].rstrip() + "..."
    return text


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if not isinstance(content, list):
        return str(content)

    parts: list[str] = []
    image_count = 0
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            parts.append(str(item))
            continue

        item_type = str(item.get("type") or "").strip()
        if item_type == "text":
            parts.append(str(item.get("text") or ""))
            continue
        if item_type in {"image_url", "input_image", "image"} or "image_url" in item:
            image_count += 1
            image_url = item.get("image_url")
            if isinstance(image_url, dict):
                url = _trim_data_url(str(image_url.get("url") or ""))
            else:
                url = _trim_data_url(str(image_url or item.get("image") or ""))
            parts.append(f"[image {image_count} provided to gateway; url={url}]")
            continue

        # Preserve unknown structured parts as text so clients using future
        # OpenAI content blocks are accepted instead of rejected by validation.
        parts.append(str(item))

    return "\n".join(part for part in parts if part).strip()


def _normalize_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="messages must be a list")
    normalized: list[dict[str, Any]] = []
    for raw in messages:
        if not isinstance(raw, dict):
            raise HTTPException(status_code=400, detail="each message must be an object")
        msg = dict(raw)
        msg["content"] = _normalize_content(msg.get("content"))
        normalized.append(msg)
    if not normalized:
        raise HTTPException(status_code=400, detail="messages must not be empty")
    return normalized


def _request_payload(
    body: dict[str, Any],
    *,
    served_model_name: str,
    default_temperature: float,
    default_top_p: float,
    default_seed: int | None,
) -> dict[str, Any]:
    payload = dict(body)
    payload["model"] = str(payload.get("model") or served_model_name)
    payload["messages"] = _normalize_messages(payload.get("messages"))
    do_sample = payload.pop("do_sample", None)
    if do_sample is False:
        payload["temperature"] = 0.0
    if "temperature" not in payload:
        payload["temperature"] = default_temperature
    if "top_p" not in payload:
        payload["top_p"] = default_top_p
    if default_seed is not None and "seed" not in payload:
        payload["seed"] = default_seed
    return payload


def build_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="AI Model Gateway", version=VERSION)
    upstream_base = str(args.upstream_base_url).rstrip("/")
    timeout = httpx.Timeout(float(args.timeout_seconds), connect=20.0)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        upstream_health: dict[str, Any] | None = None
        try:
            async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
                resp = await client.get(f"{upstream_base}/health")
            if resp.status_code == 200:
                upstream_health = resp.json()
        except Exception:
            upstream_health = None
        return {
            "status": "ok",
            "name": "AI Model Gateway",
            "version": VERSION,
            "served_model_name": args.served_model_name,
            "upstream_base_url": upstream_base,
            "upstream_health": upstream_health,
        }

    @app.get("/v1/models")
    async def models() -> Response:
        try:
            async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
                resp = await client.get(f"{upstream_base}/v1/models")
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type", "application/json"),
            )
        except httpx.HTTPError:
            return JSONResponse(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": args.served_model_name,
                            "object": "model",
                            "created": 0,
                            "owned_by": "local",
                        }
                    ],
                }
            )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid JSON body: {exc}") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="request body must be an object")

        payload = _request_payload(
            body,
            served_model_name=args.served_model_name,
            default_temperature=float(args.default_temperature),
            default_top_p=float(args.default_top_p),
            default_seed=args.default_seed,
        )
        headers = {"Authorization": request.headers.get("authorization", "Bearer dummy")}

        if bool(payload.get("stream")):
            client = httpx.AsyncClient(timeout=None, trust_env=False)
            upstream_req = client.build_request(
                "POST",
                f"{upstream_base}/v1/chat/completions",
                json=payload,
                headers=headers,
            )
            try:
                upstream_resp = await client.send(upstream_req, stream=True)
            except httpx.HTTPError as exc:
                await client.aclose()
                raise HTTPException(status_code=502, detail=f"upstream request failed: {exc}") from exc

            async def iterator():
                try:
                    async for chunk in upstream_resp.aiter_bytes():
                        yield chunk
                finally:
                    await upstream_resp.aclose()
                    await client.aclose()

            return StreamingResponse(
                iterator(),
                status_code=upstream_resp.status_code,
                media_type=upstream_resp.headers.get("content-type", "text/event-stream"),
            )

        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            try:
                resp = await client.post(
                    f"{upstream_base}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                )
            except httpx.HTTPError as exc:
                raise HTTPException(status_code=502, detail=f"upstream request failed: {exc}") from exc

        # Some vLLM versions do not support every OpenAI response_format variant.
        # Keep serving usable for planner evals by retrying once without it.
        if resp.status_code >= 400 and "response_format" in payload and bool(args.retry_without_response_format):
            retry_payload = dict(payload)
            retry_payload.pop("response_format", None)
            async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
                resp = await client.post(
                    f"{upstream_base}/v1/chat/completions",
                    json=retry_payload,
                    headers=headers,
                )

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
        )

    @app.get("/")
    async def root() -> dict[str, Any]:
        return {
            "id": f"aimodelgateway-{uuid.uuid4().hex[:8]}",
            "object": "service",
            "created": int(time.time()),
            "name": "AI Model Gateway",
            "version": VERSION,
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Model Gateway 0.1.0")
    parser.add_argument("--upstream-base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--served-model-name", default="qwen3.5-4b")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--default-temperature", type=float, default=0.0)
    parser.add_argument("--default-top-p", type=float, default=1.0)
    parser.add_argument("--default-seed", type=int, default=42)
    parser.add_argument("--retry-without-response-format", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = build_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
