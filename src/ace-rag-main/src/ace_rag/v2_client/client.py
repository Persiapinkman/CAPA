from __future__ import annotations

import asyncio
from typing import Any

import requests


class V2ClientError(RuntimeError):
    pass


class V2Client:
    def __init__(self, base_url: str, timeout_seconds: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def health(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._request_json, "GET", "/health", None)

    async def retrieve(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._request_json, "POST", "/retrieve", payload)

    async def query(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._request_json, "POST", "/query", payload)

    async def embedding(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._request_json, "POST", "/embedding", payload)

    def _request_json(self, method: str, path: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            if method == "GET":
                response = requests.get(url, timeout=self.timeout_seconds)
            else:
                response = requests.request(method, url, json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise V2ClientError(f"v2 returned non-object JSON from {path}")
            return data
        except requests.RequestException as exc:
            raise V2ClientError(f"v2 request failed: {method} {url}: {exc}") from exc
        except ValueError as exc:
            raise V2ClientError(f"v2 returned invalid JSON: {method} {url}: {exc}") from exc
