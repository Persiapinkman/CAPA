#!/usr/bin/env python3
"""Run the Qwen3.5-4B real-weight (G1) and LoRA backward (G2) gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from peft.tuners.tuners_utils import BaseTunerLayer
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = Path("/raid/zkq/models/Qwen3.5-4B")
DEFAULT_STEP_DATA = (
    ROOT
    / "training/planner_grpo_seed_v1/step_data/"
    "planner_multistep_grpo_value_v5_train_v1_qwen35_4b_nothinking_step2.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT
    / "experiments/studies/planner_multistep_tool_routing_grpo_qwen35_4b_v1/"
    "qwen35_grpo_g1_g2_20260715.json"
)
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
]
EXPECTED_MODEL_CLASS = "Qwen3_5ForCausalLM"
EXPECTED_LORA_MODULES = 152
EXPECTED_TRAINABLE_PARAMS = 14_376_960
EXPECTED_EOS_ID = 248046
EXPECTED_PAD_ID = 248044
GIB = 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--step-data", type=Path, default=DEFAULT_STEP_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--completion-tokens", type=int, default=320)
    parser.add_argument("--max-allocated-gib", type=float, default=28.0)
    parser.add_argument("--minimum-free-gib", type=float, default=2.0)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_step_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected an object")
        rows.append(value)
    if len(rows) != 480 or any(int(row.get("step_index") or 0) != 2 for row in rows):
        raise ValueError("G1/G2 requires the frozen 480-row target-step-2 dataset")
    return rows


def memory_snapshot(device: torch.device) -> dict[str, float]:
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    return {
        "allocated_gib": torch.cuda.memory_allocated(device) / GIB,
        "reserved_gib": torch.cuda.memory_reserved(device) / GIB,
        "max_allocated_gib": torch.cuda.max_memory_allocated(device) / GIB,
        "max_reserved_gib": torch.cuda.max_memory_reserved(device) / GIB,
        "device_free_gib": free / GIB,
        "device_total_gib": total / GIB,
    }


def load_model(model_path: Path) -> tuple[Any, dict[str, Any]]:
    loaded = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        output_loading_info=True,
    )
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise RuntimeError("Transformers did not return loading_info")
    model, loading_info = loaded
    if model.__class__.__name__ != EXPECTED_MODEL_CLASS:
        raise TypeError(f"expected {EXPECTED_MODEL_CLASS}, got {model.__class__.__name__}")
    missing = list(loading_info.get("missing_keys") or [])
    unexpected = list(loading_info.get("unexpected_keys") or [])
    mismatched = list(loading_info.get("mismatched_keys") or [])
    errors = list(loading_info.get("error_msgs") or [])
    if missing or unexpected or mismatched or errors:
        raise RuntimeError(
            f"weight loading contract failed: missing={missing[:5]}, unexpected={unexpected[:5]}, "
            f"mismatched={mismatched[:5]}, errors={errors[:3]}"
        )
    visual_modules = [name for name, _ in model.named_modules() if name == "visual" or name.startswith("visual.")]
    if visual_modules:
        raise RuntimeError(f"text policy unexpectedly instantiated visual modules: {visual_modules[:5]}")
    return model, {
        "class": model.__class__.__name__,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "mismatched_keys": mismatched,
        "error_msgs": errors,
        "visual_module_count": len(visual_modules),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def run_g1(model: Any, prompt_ids: list[int], device: torch.device) -> dict[str, Any]:
    model.to(device)
    model.eval()
    model.config.use_cache = False
    checks: list[dict[str, Any]] = []
    for length in (32, 512, 4096):
        if len(prompt_ids) < length:
            raise ValueError(f"longest prompt has only {len(prompt_ids)} tokens")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        input_ids = torch.tensor([prompt_ids[:length]], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        started = time.perf_counter()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=1,
            )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        finite = bool(torch.isfinite(outputs.logits).all().item())
        if not finite:
            raise FloatingPointError(f"G1 length={length} produced non-finite logits")
        checks.append(
            {
                "tokens": length,
                "finite": finite,
                "elapsed_seconds": elapsed,
                "tokens_per_second": length / elapsed,
                "memory": memory_snapshot(device),
            }
        )
        del outputs, input_ids, attention_mask
    return {"status": "pass", "forward_checks": checks, "final_memory": memory_snapshot(device)}


def lora_audit(model: Any) -> dict[str, Any]:
    modules = [(name, module) for name, module in model.named_modules() if isinstance(module, BaseTunerLayer)]
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    target_counts: dict[str, int] = {}
    for name, _ in modules:
        leaf = name.rsplit(".", 1)[-1]
        target_counts[leaf] = target_counts.get(leaf, 0) + 1
    if len(modules) != EXPECTED_LORA_MODULES or trainable_params != EXPECTED_TRAINABLE_PARAMS:
        raise RuntimeError(
            "LoRA coverage mismatch: "
            f"modules={len(modules)} expected={EXPECTED_LORA_MODULES}, "
            f"trainable={trainable_params} expected={EXPECTED_TRAINABLE_PARAMS}, targets={target_counts}"
        )
    return {
        "module_count": len(modules),
        "trainable_parameter_count": trainable_params,
        "target_counts": dict(sorted(target_counts.items())),
        "module_names": [name for name, _ in modules],
    }


def run_g2(
    base_model: Any,
    prompt_ids: list[int],
    device: torch.device,
    completion_tokens: int,
) -> dict[str, Any]:
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(base_model, peft_config)
    model.train()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    audit = lora_audit(model)

    filler_token = 1773  # A normal punctuation token; never used as an EOS/PAD control token.
    full_ids = prompt_ids + [filler_token] * completion_tokens
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    targets = input_ids[:, -completion_tokens:]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    scaler = torch.amp.GradScaler("cuda", init_scale=1.0, growth_interval=100000)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=completion_tokens + 1,
        )
        prediction_logits = outputs.logits[:, :-1, :].float()
        if prediction_logits.shape[1] != completion_tokens:
            raise RuntimeError(f"unexpected kept-logit shape: {tuple(prediction_logits.shape)}")
        loss = F.cross_entropy(
            prediction_logits.reshape(-1, prediction_logits.shape[-1]),
            targets.reshape(-1),
        )
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError(f"G2 produced non-finite loss: {loss.item()}")
    scaler.scale(loss).backward()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    gradient_tensors = 0
    missing_gradient_tensors: list[str] = []
    nonfinite_gradient_tensors: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing_gradient_tensors.append(name)
            continue
        gradient_tensors += 1
        if not bool(torch.isfinite(parameter.grad).all().item()):
            nonfinite_gradient_tensors.append(name)
    if missing_gradient_tensors or nonfinite_gradient_tensors:
        raise FloatingPointError(
            f"G2 gradient contract failed: missing={missing_gradient_tensors[:5]}, "
            f"nonfinite={nonfinite_gradient_tensors[:5]}"
        )
    memory = memory_snapshot(device)
    result = {
        "status": "pass",
        "optimizer_steps": 0,
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": completion_tokens,
        "sequence_tokens": len(full_ids),
        "loss": float(loss.detach().item()),
        "grad_scaler": {"init_scale": 1.0, "growth_interval": 100000},
        "gradient_tensors": gradient_tensors,
        "missing_gradient_tensors": missing_gradient_tensors,
        "nonfinite_gradient_tensors": nonfinite_gradient_tensors,
        "elapsed_seconds": elapsed,
        "memory": memory,
        "lora": audit,
    }
    model.zero_grad(set_to_none=True)
    if memory["max_allocated_gib"] > 28.0:
        raise RuntimeError(f"G2 peak allocation exceeded 28 GiB: {memory['max_allocated_gib']:.3f}")
    if memory["device_free_gib"] < 2.0:
        raise RuntimeError(f"G2 left less than 2 GiB device headroom: {memory['device_free_gib']:.3f}")
    return result


def main() -> None:
    args = parse_args()
    started_at = utc_now()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "started_at": started_at,
        "status": "running",
        "optimizer_steps": 0,
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.backends.cudnn.enabled = False
        device = torch.device("cuda:0")
        model_path = args.model_name_or_path.resolve()
        step_data = args.step_data if args.step_data.is_absolute() else ROOT / args.step_data
        rows = read_step_rows(step_data)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=False,
            use_fast=True,
            padding_side="left",
        )
        if tokenizer.eos_token_id != EXPECTED_EOS_ID or tokenizer.pad_token_id != EXPECTED_PAD_ID:
            raise RuntimeError(
                f"tokenizer contract mismatch: eos={tokenizer.eos_token_id}, pad={tokenizer.pad_token_id}"
            )
        longest = max(rows, key=lambda row: int(row["prompt_token_count"]))
        prompt_ids = tokenizer(longest["prompt"], add_special_tokens=False)["input_ids"]
        if len(prompt_ids) != int(longest["prompt_token_count"]):
            raise RuntimeError("frozen prompt token count changed")

        model, load_audit = load_model(model_path)
        payload.update(
            {
                "model_name_or_path": str(model_path),
                "step_data": str(step_data),
                "step_data_sha256": sha256_file(step_data),
                "longest_case_id": longest["case_id"],
                "longest_prompt_tokens": len(prompt_ids),
                "cuda_device": torch.cuda.get_device_name(device),
                "cuda_capability": list(torch.cuda.get_device_capability(device)),
                "cudnn_enabled": torch.backends.cudnn.enabled,
                "loading": load_audit,
            }
        )
        payload["g1"] = run_g1(model, prompt_ids, device)
        payload["g2"] = run_g2(model, prompt_ids, device, args.completion_tokens)
        peak = float(payload["g2"]["memory"]["max_allocated_gib"])
        free = float(payload["g2"]["memory"]["device_free_gib"])
        if peak > args.max_allocated_gib or free < args.minimum_free_gib:
            raise RuntimeError(
                f"G2 headroom gate failed: peak={peak:.3f} GiB, free={free:.3f} GiB, "
                f"limits={args.max_allocated_gib}/{args.minimum_free_gib}"
            )
        payload["status"] = "pass"
    except Exception as exc:
        payload["status"] = "fail"
        payload["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        payload["finished_at"] = utc_now()
        write_json(output, payload)
        raise
    payload["finished_at"] = utc_now()
    write_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
