#!/usr/bin/env python3
"""Use the frozen benchmark driver against Transformers continuous batching.

The shared driver still controls condition randomization, trial boundaries,
telemetry, validation, and persistence.  This adapter changes only the request
schema: Transformers Serve accepts text prompts and a JSON GenerationConfig,
whereas vLLM accepts token-id prompts plus ignore_eos/min_tokens extensions.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp
from transformers import AutoTokenizer

import benchmark_openai_throughput as core


TOKENIZER: Any = None


async def one_transformers_completion(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    prompt_token_ids: list[int],
    output_length: int,
    request_id: str,
    seed: int,
) -> dict[str, Any]:
    prompt = TOKENIZER.decode(
        prompt_token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    round_trip = TOKENIZER.encode(prompt, add_special_tokens=False)
    if round_trip != prompt_token_ids:
        raise ValueError(f"{request_id}: decoded prompt does not round-trip to the source token IDs")

    generation_config = {
        "max_new_tokens": output_length,
        "min_new_tokens": output_length,
        "do_sample": False,
        "eos_token_id": None,
        "pad_token_id": TOKENIZER.pad_token_id,
    }
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": output_length,
        "stream": True,
        "seed": seed,
        "generation_config": json.dumps(generation_config, separators=(",", ":")),
    }
    started = time.perf_counter()
    first_content_at: float | None = None
    first_byte_at: float | None = None
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    text_parts: list[str] = []
    chunks = 0
    status: int | None = None

    try:
        async with session.post(
            f"{endpoint.rstrip('/')}/v1/completions",
            json=payload,
            headers={"X-Request-ID": request_id},
        ) as response:
            status = response.status
            if response.status != 200:
                body = await response.text()
                raise RuntimeError(f"HTTP {response.status}: {body[:1000]}")
            async for raw_line in response.content:
                now = time.perf_counter()
                if first_byte_at is None and raw_line:
                    first_byte_at = now
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                event = json.loads(data)
                if event.get("error") is not None:
                    raise RuntimeError(f"stream error: {event['error']}")
                if event.get("usage") is not None:
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                content = choice.get("text") or ""
                if content:
                    chunks += 1
                    text_parts.append(content)
                    if first_content_at is None:
                        first_content_at = now
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
        ended = time.perf_counter()
        if first_content_at is None:
            raise RuntimeError("stream completed without a non-empty content chunk")
        if usage is None:
            raise RuntimeError("stream completed without server usage statistics")
        completion_tokens = int(usage.get("completion_tokens", -1))
        prompt_tokens = int(usage.get("prompt_tokens", -1))
        latency = ended - started
        ttft = first_content_at - started
        tpot = (latency - ttft) / max(completion_tokens - 1, 1)
        return {
            "request_id": request_id,
            "success": True,
            "http_status": status,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": int(usage.get("total_tokens", prompt_tokens + completion_tokens)),
            "latency_s": latency,
            "ttfb_s": None if first_byte_at is None else first_byte_at - started,
            "ttft_s": ttft,
            "tpot_s": tpot,
            "stream_content_chunks": chunks,
            "finish_reason": finish_reason,
            "output_chars": sum(len(part) for part in text_parts),
            "output_sha256": hashlib.sha256("".join(text_parts).encode("utf-8")).hexdigest(),
            "error": None,
        }
    except Exception as exc:
        ended = time.perf_counter()
        return {
            "request_id": request_id,
            "success": False,
            "http_status": status,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "latency_s": ended - started,
            "ttfb_s": None if first_byte_at is None else first_byte_at - started,
            "ttft_s": None if first_content_at is None else first_content_at - started,
            "tpot_s": None,
            "stream_content_chunks": chunks,
            "finish_reason": finish_reason,
            "output_chars": sum(len(part) for part in text_parts),
            "output_sha256": hashlib.sha256("".join(text_parts).encode("utf-8")).hexdigest(),
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:18080")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt-pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--phase", default="formal")
    parser.add_argument("--gpu-indices", type=int, nargs="+", required=True)
    parser.add_argument("--conditions", default="256:1,256:4,256:16,2048:1,2048:4,2048:16")
    parser.add_argument("--output-length", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--waves-per-trial", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    global TOKENIZER
    TOKENIZER = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False, use_fast=True)
    if TOKENIZER.pad_token_id is None:
        TOKENIZER.pad_token = TOKENIZER.eos_token

    # Preserve the shared implementation while recording this adapter's exact hash.
    core.one_completion = one_transformers_completion
    core.__file__ = __file__
    asyncio.run(core.async_main(args))


if __name__ == "__main__":
    main()
