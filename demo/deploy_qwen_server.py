#!/usr/bin/env python3
"""Minimal OpenAI-compatible chat server for a local Qwen causal LM."""

from __future__ import annotations

import argparse
import time
import uuid
from threading import Lock
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.backends.cudnn.enabled = False

try:
    from peft import PeftModel
except Exception:  # pragma: no cover - optional runtime dependency
    PeftModel = None


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = Field(default=2048, ge=1, le=8192)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = False


def normalize_content(content: Any) -> str:
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
        elif item_type in {"image_url", "input_image", "image"} or "image_url" in item:
            image_count += 1
            parts.append(f"[image {image_count} provided]")
        else:
            parts.append(str(item))

    return "\n".join(part for part in parts if part).strip()


def build_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="Qwen local inference server")
    lock = Lock()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16 if args.dtype == "bfloat16" else "auto"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
    )
    if args.adapter_path:
        if PeftModel is None:
            raise RuntimeError("peft is required when --adapter-path is provided")
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": args.model_name,
            "model_path": args.model_path,
            "adapter_path": args.adapter_path,
            "cuda": torch.cuda.is_available(),
        }

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": args.model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
        if request.stream:
            raise HTTPException(status_code=400, detail="stream=true is not implemented by this lightweight server")
        if not request.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")

        messages = [
            {"role": message.role, "content": normalize_content(message.content)}
            for message in request.messages
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=bool(args.enable_thinking),
            )
        except Exception:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        do_sample = request.temperature > 0
        generation_kwargs = {
            "max_new_tokens": request.max_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = request.temperature
            generation_kwargs["top_p"] = request.top_p

        with lock:
            with torch.inference_mode():
                output_ids = model.generate(**inputs, **generation_kwargs)
        completion_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
        content = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        prompt_tokens = int(inputs["input_ids"].numel())
        completion_tokens = int(completion_ids.numel())

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model or args.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local Qwen model with OpenAI-compatible endpoints.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--model-name", default="qwen3.5-4b")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "auto"], default="float16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Qwen thinking mode in chat template. Disabled by default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = build_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
