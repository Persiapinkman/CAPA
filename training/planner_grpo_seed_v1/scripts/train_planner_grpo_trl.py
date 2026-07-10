#!/usr/bin/env python3
"""TRL GRPO training entrypoint for CAPA Planner routing.

This script is intentionally separate from the older prototype so the TRL path
can run in a clean environment without vLLM compatibility patches.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "demo"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from training.planner_grpo_seed_v1.scripts.train_planner_grpo import (  # noqa: E402
    DEFAULT_CASES,
    build_step_dataset,
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
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "planner-grpo-qwen25-7b-trl-lora"))
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
    parser.add_argument("--ddp-find-unused-parameters", type=parse_bool, default=False)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--run-name", default="qwen25-7b-capa-planner-trl-grpo-lora")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Build dataset/reward/token stats, then exit.")
    return parser.parse_args()


def make_reward_func():
    def reward_func(completions, **kwargs) -> list[float]:
        scores: list[float] = []
        expected_steps = kwargs.get("expected_step", [])
        forbidden_actions = kwargs.get("forbidden_actions", [])
        reward_specs = kwargs.get("reward_spec", [])
        previous_actions = kwargs.get("previous_action", [])
        full_expected_actions = kwargs.get("full_expected_actions", [])
        step_indexes = kwargs.get("step_index", [])
        for idx, completion in enumerate(completions):
            scores.append(
                score_step_completion(
                    completion=completion,
                    expected_step=expected_steps[idx],
                    forbidden_actions=forbidden_actions[idx],
                    reward_spec=reward_specs[idx],
                    previous_action=previous_actions[idx],
                    full_expected_actions=full_expected_actions[idx],
                    step_index=int(step_indexes[idx]),
                )
            )
        return scores

    return reward_func


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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    model_path = resolve_model_name_or_path(args.model_name_or_path, ROOT)
    cases_path = args.cases if args.cases.is_absolute() else ROOT / args.cases

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
        padding_side="left",
        truncation_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = build_step_dataset(load_jsonl(cases_path))
    dataset, length_stats = filter_by_prompt_length(dataset, tokenizer, args.max_prompt_tokens)
    reward_smoke = make_reward_func()(
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
        "model_name_or_path": model_path,
        "adapter_path": args.adapter_path,
        "cases": str(cases_path),
        "output_dir": str(output_dir),
        "length_stats": length_stats,
        "reward_smoke": reward_smoke,
        "num_generations": args.num_generations,
        "generation_batch_size": args.generation_batch_size,
        "max_completion_length": args.max_completion_length,
        "remove_invalid_values": args.remove_invalid_values,
        "renormalize_logits": args.renormalize_logits,
        "mask_truncated_completions": args.mask_truncated_completions,
        "lora": {
            "enabled": args.use_lora,
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": args.lora_target_modules,
        },
    }
    write_run_config(output_dir, run_payload)
    print(json.dumps({"status": "prepared", **run_payload}, ensure_ascii=False, indent=2), flush=True)
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
        reward_funcs=make_reward_func(),
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()
