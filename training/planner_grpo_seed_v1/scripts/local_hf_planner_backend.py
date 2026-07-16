#!/usr/bin/env python3
"""Local Transformers backend implementing the Planner VLMService interface."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class LocalHFPlannerBackend:
    """Reusable local model that can replace ``capa.agent.VLMService``."""

    def __init__(
        self,
        *,
        model_path: Path,
        adapter_path: Path | None = None,
        device: str = "cuda",
        attn_implementation: str = "sdpa",
    ) -> None:
        self.model_path = model_path.resolve()
        self.adapter_path = adapter_path.resolve() if adapter_path is not None else None
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=False,
            use_fast=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=torch.float16,
            attn_implementation=attn_implementation,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        ).to(device)
        self.model: Any = base
        if self.adapter_path is not None:
            self.model = PeftModel.from_pretrained(base, self.adapter_path).to(device)
        self.model.eval()
        self.last_response_metadata: dict[str, Any] = {}

    def service_factory(self, *args: Any, **kwargs: Any) -> "LocalHFPlannerBackend":
        """Return this loaded backend for each VLMService construction."""

        return self

    @staticmethod
    def _messages(prompt: str | None, messages: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        if isinstance(messages, list) and messages:
            return messages
        return [{"role": "user", "content": str(prompt or "")}]

    def generate_text(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        image_paths: list[str] | None = None,
        response_format: dict[str, Any] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        del image_paths, response_format
        request_messages = self._messages(prompt, messages)
        rendered = self.tokenizer.apply_chat_template(
            request_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(
            rendered,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.device)
        temperature = float(kwargs.get("temperature", os.environ.get("DEMO_OPENAI_TEMPERATURE", "0")))
        top_p = float(kwargs.get("top_p", os.environ.get("DEMO_OPENAI_TOP_P", "1")))
        max_new_tokens = int(
            kwargs.get("max_tokens", os.environ.get("DEMO_OPENAI_MAX_TOKENS", "320"))
        )
        do_sample = env_bool("DEMO_OPENAI_DO_SAMPLE", default=temperature > 0)
        if temperature <= 0:
            do_sample = False
        seed = int(kwargs.get("seed", os.environ.get("DEMO_OPENAI_SEED", "42")))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        generation: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "remove_invalid_values": True,
            "renormalize_logits": True,
            "use_cache": True,
        }
        if do_sample:
            generation.update({"temperature": temperature, "top_p": top_p})
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, **generation)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        completion_ids = output[0, inputs.input_ids.shape[1] :].detach().cpu().tolist()
        natural_eos = self.tokenizer.eos_token_id in completion_ids
        self.last_response_metadata = {
            "api_call_ms": elapsed_ms,
            "input_tokens": int(inputs.input_ids.shape[1]),
            "output_tokens": len(completion_ids),
            "total_tokens": int(inputs.input_ids.shape[1]) + len(completion_ids),
            "finish_reason": "stop" if natural_eos else "length",
            "model": model or str(self.model_path),
            "backend": "local_transformers",
        }
        return self.tokenizer.decode(completion_ids, skip_special_tokens=False)
