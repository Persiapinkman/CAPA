#!/usr/bin/env python3
"""TRL GRPO training entrypoint for CAPA Planner routing.

This script is intentionally separate from the older prototype so the TRL path
can run in a clean environment without vLLM compatibility patches.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "demo"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from training.planner_grpo_seed_v1.scripts.train_planner_grpo import (  # noqa: E402
    DEFAULT_CASES,
    build_step_dataset,
    completion_text,
    first_json_span_text,
    first_json_text,
    load_jsonl,
    score_step_completion,
)
from util.path_resolver import resolve_model_name_or_path  # noqa: E402


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen2.5 Planner LoRA with TRL GRPO.")
    parser.add_argument("--model-name-or-path", default="/raid/zkq/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter-path", default="", help="Optional trainable PEFT adapter to continue from.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument(
        "--step-data",
        type=Path,
        default=None,
        help="Optional prebuilt step JSONL. This freezes prompts across ranks and runs.",
    )
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "planner-grpo-qwen25-7b-trl-lora"))
    parser.add_argument("--prompt-format", choices=["pseudo", "qwen_chatml"], default="pseudo")
    parser.add_argument("--max-prompt-tokens", type=int, default=6144)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--remove-invalid-values", type=parse_bool, default=True)
    parser.add_argument("--renormalize-logits", type=parse_bool, default=True)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", type=parse_bool, default=True)
    parser.add_argument("--bf16", type=parse_bool, default=False)
    parser.add_argument("--gradient-checkpointing", type=parse_bool, default=True)
    parser.add_argument("--trust-remote-code", type=parse_bool, default=True)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--use-lora", type=parse_bool, default=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--loss-type", default="dr_grpo")
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--mask-truncated-completions", type=parse_bool, default=False)
    parser.add_argument(
        "--score-first-json-only",
        type=parse_bool,
        default=True,
        help="Score only the first complete JSON object in each generation.",
    )
    parser.add_argument("--task-reward-weight", type=float, default=0.75)
    parser.add_argument("--format-reward-weight", type=float, default=0.25)
    parser.add_argument("--tail-penalty-tokens", type=int, default=64)
    parser.add_argument("--prefix-penalty-tokens", type=int, default=16)
    parser.add_argument("--penalize-truncated-completions", type=parse_bool, default=True)
    parser.add_argument("--ddp-find-unused-parameters", type=parse_bool, default=False)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--run-name", default="qwen25-7b-capa-planner-trl-grpo-lora")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Build dataset/reward/token stats, then exit.")
    return parser.parse_args()


def bounded_decay_score(token_count: int, budget: int) -> float:
    if token_count <= 0:
        return 1.0
    if budget <= 0:
        return 0.0
    return max(0.0, 1.0 - (float(token_count) / float(budget)))


def make_format_reward(
    *,
    tokenizer,
    max_completion_length: int,
    tail_penalty_tokens: int,
    prefix_penalty_tokens: int,
    penalize_truncated_completions: bool,
) -> Any:
    def format_reward(completion: Any) -> float:
        raw = completion_text(completion)
        first_json, start, end = first_json_span_text(raw)
        if not first_json:
            return 0.0

        prefix = raw[:start].strip() if start >= 0 else raw
        tail = raw[end:].strip() if end >= 0 else raw
        prefix_tokens = len(tokenizer(prefix, add_special_tokens=False)["input_ids"]) if prefix else 0
        tail_tokens = len(tokenizer(tail, add_special_tokens=False)["input_ids"]) if tail else 0
        raw_tokens = len(tokenizer(raw, add_special_tokens=False)["input_ids"]) if raw else 0

        prefix_score = bounded_decay_score(prefix_tokens, prefix_penalty_tokens)
        tail_score = bounded_decay_score(tail_tokens, tail_penalty_tokens)
        if penalize_truncated_completions:
            truncated_score = 0.0 if raw_tokens >= max_completion_length else 1.0
        else:
            truncated_score = 1.0

        return max(0.0, min(1.0, 0.20 * prefix_score + 0.55 * tail_score + 0.25 * truncated_score))

    return format_reward


def make_reward_func(
    *,
    tokenizer,
    score_first_json_only: bool = True,
    max_completion_length: int,
    task_reward_weight: float,
    format_reward_weight: float,
    tail_penalty_tokens: int,
    prefix_penalty_tokens: int,
    penalize_truncated_completions: bool,
):
    format_reward = make_format_reward(
        tokenizer=tokenizer,
        max_completion_length=max_completion_length,
        tail_penalty_tokens=tail_penalty_tokens,
        prefix_penalty_tokens=prefix_penalty_tokens,
        penalize_truncated_completions=penalize_truncated_completions,
    )
    denominator = max(1e-8, task_reward_weight + format_reward_weight)

    def reward_func(completions, **kwargs) -> list[float]:
        scores: list[float] = []
        expected_steps = kwargs.get("expected_step", [])
        forbidden_actions = kwargs.get("forbidden_actions", [])
        reward_specs = kwargs.get("reward_spec", [])
        previous_actions = kwargs.get("previous_action", [])
        full_expected_actions = kwargs.get("full_expected_actions", [])
        step_indexes = kwargs.get("step_index", [])
        for idx, completion in enumerate(completions):
            scored_completion = first_json_text(completion) if score_first_json_only else completion
            task_score = score_step_completion(
                completion=scored_completion,
                expected_step=expected_steps[idx],
                forbidden_actions=forbidden_actions[idx],
                reward_spec=reward_specs[idx],
                previous_action=previous_actions[idx],
                full_expected_actions=full_expected_actions[idx],
                step_index=int(step_indexes[idx]),
                first_json_only=score_first_json_only,
            )
            score = ((task_reward_weight * task_score) + (format_reward_weight * format_reward(completion))) / denominator
            scores.append(max(0.0, min(1.0, score)))
        return scores

    return reward_func


def pseudo_prompt_to_qwen_chatml(prompt: str) -> str:
    prefix = "<|system|>\n"
    user_sep = "\n<|user|>\n"
    assistant_suffix = "\n<|assistant|>\n"
    if not prompt.startswith(prefix) or user_sep not in prompt or not prompt.endswith(assistant_suffix):
        raise ValueError("prompt does not match expected pseudo chat format")
    body = prompt[len(prefix) : -len(assistant_suffix)]
    system, user = body.split(user_sep, 1)
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def apply_prompt_format(dataset, prompt_format: str):
    if prompt_format == "pseudo":
        return dataset
    if prompt_format == "qwen_chatml":
        return dataset.map(lambda row: {"prompt": pseudo_prompt_to_qwen_chatml(row["prompt"])})
    raise ValueError(f"unsupported prompt_format: {prompt_format}")


def filter_by_prompt_length(dataset, tokenizer, max_prompt_tokens: int):
    lengths = [len(tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]) for row in dataset]
    kept_indexes = [idx for idx, length in enumerate(lengths) if length <= max_prompt_tokens]
    if not kept_indexes:
        raise ValueError(f"all {len(dataset)} samples exceed --max-prompt-tokens={max_prompt_tokens}")
    filtered = dataset.select(kept_indexes)
    stats = {
        "rows_before": len(dataset),
        "rows_after": len(filtered),
        "min_prompt_tokens": min(lengths),
        "max_prompt_tokens": max(lengths),
        "p50_prompt_tokens": sorted(lengths)[len(lengths) // 2],
        "dropped_over_limit": len(dataset) - len(filtered),
    }
    return filtered, stats


def supported_kwargs(cls: Any, values: dict[str, Any]) -> dict[str, Any]:
    supported = set(inspect.signature(cls.__init__).parameters)
    return {key: value for key, value in values.items() if key in supported}


def write_run_config(path: Path, payload: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "capa_trl_grpo_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def install_peft_ddp_no_tp_resume_compat() -> bool:
    """Avoid PEFT's TP-only import when a plain DDP LoRA checkpoint is loaded."""
    import transformers.integrations.tensor_parallel as transformers_tp
    from peft.utils import save_and_load as peft_save_and_load

    if hasattr(transformers_tp, "EmbeddingParallel"):
        return False

    original = peft_save_and_load._maybe_shard_state_dict_for_tp
    if getattr(original, "_capa_ddp_no_tp_compat", False):
        return True

    def shard_only_when_tensor_parallel(model, state_dict, adapter_name):
        has_tensor_parallel_layer = False
        for module in model.modules():
            get_base_layer = getattr(module, "get_base_layer", None)
            if get_base_layer is None:
                continue
            base_layer = get_base_layer()
            if (
                getattr(base_layer, "_hf_tp_plan", None) is not None
                and getattr(base_layer, "_hf_device_mesh", None) is not None
            ):
                has_tensor_parallel_layer = True
                break
        if has_tensor_parallel_layer:
            return original(model, state_dict, adapter_name)
        return None

    shard_only_when_tensor_parallel._capa_ddp_no_tp_compat = True
    peft_save_and_load._maybe_shard_state_dict_for_tp = shard_only_when_tensor_parallel
    return True


def summarize_log_history(log_history: list[dict[str, Any]]) -> dict[str, Any]:
    steps = [
        row
        for row in log_history
        if row.get("step") is not None and row.get("reward") is not None
    ]

    def mean_of(key: str) -> float | None:
        values = [float(row[key]) for row in steps if row.get(key) is not None]
        return sum(values) / len(values) if values else None

    def max_of(key: str) -> float | None:
        values = [float(row[key]) for row in steps if row.get(key) is not None]
        return max(values) if values else None

    return {
        "logged_steps": len(steps),
        "mean_reward": mean_of("reward"),
        "mean_reward_std": mean_of("reward_std"),
        "informative_reward_steps": sum(float(row.get("reward_std") or 0.0) > 0.0 for row in steps),
        "mean_frac_reward_zero_std": mean_of("frac_reward_zero_std"),
        "mean_entropy": mean_of("entropy"),
        "mean_completion_clipped_ratio": mean_of("completions/clipped_ratio"),
        "max_completion_clipped_ratio": max_of("completions/clipped_ratio"),
        "clipped_generation_steps": sum(
            float(row.get("completions/clipped_ratio") or 0.0) > 0.0 for row in steps
        ),
        "nonfinite_gradient_steps": sum(
            not math.isfinite(float(row["grad_norm"]))
            for row in steps
            if row.get("grad_norm") is not None
        ),
        "zero_gradient_steps": sum(float(row.get("grad_norm") or 0.0) == 0.0 for row in steps),
        "final": steps[-1] if steps else {},
    }


def main() -> None:
    args = parse_args()
    is_main_process = int(os.environ.get("RANK", "0")) == 0
    started_at = utc_now()
    started_perf = time.perf_counter()
    set_seed(args.seed)
    model_path = resolve_model_name_or_path(args.model_name_or_path, ROOT)
    cases_path = args.cases if args.cases.is_absolute() else ROOT / args.cases
    step_data_path = None
    if args.step_data is not None:
        step_data_path = args.step_data if args.step_data.is_absolute() else ROOT / args.step_data

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
        padding_side="left",
        truncation_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if step_data_path is not None:
        step_rows = load_jsonl(step_data_path)
        for row in step_rows:
            row.pop("completion", None)
        dataset = Dataset.from_list(step_rows)
        if args.prompt_format == "qwen_chatml" and any(
            not str(row["prompt"]).startswith("<|im_start|>system\n") for row in step_rows
        ):
            raise ValueError("--step-data is not in the requested qwen_chatml format")
    else:
        dataset = apply_prompt_format(build_step_dataset(load_jsonl(cases_path)), args.prompt_format)
    dataset, length_stats = filter_by_prompt_length(dataset, tokenizer, args.max_prompt_tokens)
    reward_func = make_reward_func(
        tokenizer=tokenizer,
        score_first_json_only=args.score_first_json_only,
        max_completion_length=args.max_completion_length,
        task_reward_weight=args.task_reward_weight,
        format_reward_weight=args.format_reward_weight,
        tail_penalty_tokens=args.tail_penalty_tokens,
        prefix_penalty_tokens=args.prefix_penalty_tokens,
        penalize_truncated_completions=args.penalize_truncated_completions,
    )
    reward_smoke = reward_func(
        ['{"decision_type":"tool","action":"qwen_detection","action_input":{"finish_after_tool":true}}'],
        expected_step=[dataset[0]["expected_step"]],
        forbidden_actions=[dataset[0]["forbidden_actions"]],
        reward_spec=[dataset[0]["reward_spec"]],
        previous_action=[dataset[0]["previous_action"]],
        full_expected_actions=[dataset[0]["full_expected_actions"]],
        step_index=[dataset[0]["step_index"]],
    )[0]

    output_dir = Path(args.output_dir)
    run_payload = {
        "schema_version": "1.0",
        "started_at": started_at,
        "command": " ".join(shlex.quote(value) for value in [sys.executable, *sys.argv]),
        "model_name_or_path": model_path,
        "adapter_path": args.adapter_path,
        "cases": str(cases_path),
        "step_data": str(step_data_path) if step_data_path is not None else "",
        "output_dir": str(output_dir),
        "prompt_format": args.prompt_format,
        "length_stats": length_stats,
        "reward_smoke": reward_smoke,
        "num_generations": args.num_generations,
        "generation_batch_size": args.generation_batch_size,
        "max_completion_length": args.max_completion_length,
        "remove_invalid_values": args.remove_invalid_values,
        "renormalize_logits": args.renormalize_logits,
        "mask_truncated_completions": args.mask_truncated_completions,
        "score_first_json_only": args.score_first_json_only,
        "seed": args.seed,
        "training": {
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "max_steps": args.max_steps,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "loss_type": args.loss_type,
            "beta": args.beta,
            "precision": "fp16" if args.fp16 else ("bf16" if args.bf16 else "fp32"),
            "attention_implementation": args.attn_implementation,
        },
        "reward_weights": {
            "task": args.task_reward_weight,
            "format": args.format_reward_weight,
            "tail_penalty_tokens": args.tail_penalty_tokens,
            "prefix_penalty_tokens": args.prefix_penalty_tokens,
            "penalize_truncated_completions": args.penalize_truncated_completions,
        },
        "lora": {
            "enabled": args.use_lora,
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": args.lora_target_modules,
        },
    }
    if is_main_process:
        write_run_config(output_dir, run_payload)
        print(
            json.dumps({"status": "prepared", **run_payload}, ensure_ascii=False, indent=2),
            flush=True,
        )
    if args.dry_run:
        return

    from trl import GRPOConfig, GRPOTrainer

    dtype_name = "float16" if args.fp16 else ("bfloat16" if args.bf16 else "float32")
    model_init_kwargs = {
        "dtype": dtype_name,
        "attn_implementation": args.attn_implementation,
        "low_cpu_mem_usage": True,
    }
    report_to: list[str] | None
    report_to = [] if str(args.report_to).lower() == "none" else [args.report_to]
    config_kwargs = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": True,
        "do_train": True,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "logging_steps": args.logging_steps,
        "logging_first_step": True,
        "report_to": report_to,
        "run_name": args.run_name,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "remove_unused_columns": False,
        "optim": args.optim,
        "seed": args.seed,
        "num_generations": args.num_generations,
        "generation_batch_size": args.generation_batch_size,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "generation_kwargs": {
            "remove_invalid_values": args.remove_invalid_values,
            "renormalize_logits": args.renormalize_logits,
        },
        "use_vllm": False,
        "beta": args.beta,
        "loss_type": args.loss_type,
        "mask_truncated_completions": args.mask_truncated_completions,
        "ddp_find_unused_parameters": args.ddp_find_unused_parameters,
        "average_tokens_across_devices": True,
        "torch_empty_cache_steps": args.logging_steps,
        "trust_remote_code": args.trust_remote_code,
        "model_init_kwargs": None if args.adapter_path else model_init_kwargs,
        "disable_dropout": True,
    }
    training_args = GRPOConfig(**supported_kwargs(GRPOConfig, config_kwargs))

    model: str | Any = model_path
    peft_config = None
    if args.adapter_path:
        dtype = getattr(torch, dtype_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            attn_implementation=args.attn_implementation,
            low_cpu_mem_usage=True,
            trust_remote_code=args.trust_remote_code,
        )
        model = PeftModel.from_pretrained(base_model, args.adapter_path, is_trainable=True)
    elif args.use_lora:
        target_modules = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    if args.resume_from_checkpoint:
        install_peft_ddp_no_tp_resume_compat()
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    if is_main_process:
        tokenizer.save_pretrained(str(output_dir))
    result_payload = {
        "schema_version": "1.0",
        "status": "completed",
        "started_at": started_at,
        "finished_at": utc_now(),
        "runtime_seconds": time.perf_counter() - started_perf,
        "train_metrics": dict(train_result.metrics),
        "log_summary": summarize_log_history(list(trainer.state.log_history)),
        "output_dir": str(output_dir),
    }
    if is_main_process:
        (output_dir / "capa_trl_grpo_result.json").write_text(
            json.dumps(result_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
