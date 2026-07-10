#!/usr/bin/env python3
"""TRL SFT warmup for CAPA Planner JSON routing."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "demo"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from util.path_resolver import resolve_model_name_or_path  # noqa: E402


DEFAULT_SFT_DIR = ROOT / "training" / "planner_grpo_seed_v1" / "sft_data"


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
    parser = argparse.ArgumentParser(description="Train Qwen2.5 Planner LoRA with TRL SFT.")
    parser.add_argument("--model-name-or-path", default="/raid/zkq/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter-path", default="", help="Optional trainable PEFT adapter to continue from.")
    parser.add_argument("--train-file", type=Path, default=DEFAULT_SFT_DIR / "train.jsonl")
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_SFT_DIR / "val.jsonl")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "planner-sft-qwen25-7b-trl-lora-warmup"))
    parser.add_argument("--max-length", type=int, default=5120)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--eval-steps", type=int, default=25)
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
    parser.add_argument("--ddp-find-unused-parameters", type=parse_bool, default=False)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--run-name", default="qwen25-7b-capa-planner-sft-lora-warmup")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Load data/tokenizer and report lengths, then exit.")
    return parser.parse_args()


def supported_kwargs(cls: Any, values: dict[str, Any]) -> dict[str, Any]:
    supported = set(inspect.signature(cls.__init__).parameters)
    return {key: value for key, value in values.items() if key in supported}


def path_arg(path: Path) -> str:
    resolved = path if path.is_absolute() else ROOT / path
    return str(resolved)


def load_sft_dataset(train_file: Path, eval_file: Path):
    data_files = {"train": path_arg(train_file), "eval": path_arg(eval_file)}
    return load_dataset("json", data_files=data_files)


def write_run_config(path: Path, payload: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "capa_trl_sft_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    model_path = resolve_model_name_or_path(args.model_name_or_path, ROOT)
    dataset = load_sft_dataset(args.train_file, args.eval_file)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
        padding_side="right",
        truncation_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def token_len(row: dict[str, Any]) -> int:
        return len(tokenizer(row["prompt"] + row["completion"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"])

    train_lengths = [token_len(row) for row in dataset["train"]]
    eval_lengths = [token_len(row) for row in dataset["eval"]]
    length_stats = {
        "train_rows": len(dataset["train"]),
        "eval_rows": len(dataset["eval"]),
        "train_min_tokens": min(train_lengths),
        "train_max_tokens": max(train_lengths),
        "train_p50_tokens": sorted(train_lengths)[len(train_lengths) // 2],
        "eval_min_tokens": min(eval_lengths),
        "eval_max_tokens": max(eval_lengths),
        "eval_p50_tokens": sorted(eval_lengths)[len(eval_lengths) // 2],
        "max_length": args.max_length,
        "train_over_max_length": sum(length > args.max_length for length in train_lengths),
        "eval_over_max_length": sum(length > args.max_length for length in eval_lengths),
    }
    if length_stats["train_over_max_length"] or length_stats["eval_over_max_length"]:
        raise ValueError(f"SFT examples exceed --max-length: {length_stats}")

    output_dir = Path(args.output_dir)
    run_payload = {
        "model_name_or_path": model_path,
        "adapter_path": args.adapter_path,
        "train_file": path_arg(args.train_file),
        "eval_file": path_arg(args.eval_file),
        "output_dir": str(output_dir),
        "length_stats": length_stats,
        "completion_only_loss": True,
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

    from trl import SFTConfig, SFTTrainer

    report_to = [] if str(args.report_to).lower() == "none" else [args.report_to]
    dtype_name = "float16" if args.fp16 else ("bfloat16" if args.bf16 else "float32")
    model_init_kwargs = {
        "dtype": dtype_name,
        "attn_implementation": args.attn_implementation,
        "low_cpu_mem_usage": True,
    }
    config_kwargs = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": True,
        "do_train": True,
        "eval_strategy": "steps",
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "logging_steps": args.logging_steps,
        "logging_first_step": True,
        "report_to": report_to,
        "run_name": args.run_name,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optim": args.optim,
        "seed": args.seed,
        "max_length": args.max_length,
        "completion_only_loss": True,
        "packing": False,
        "dataset_text_field": "text",
        "remove_unused_columns": True,
        "ddp_find_unused_parameters": args.ddp_find_unused_parameters,
        "average_tokens_across_devices": True,
        "torch_empty_cache_steps": args.logging_steps,
        "trust_remote_code": args.trust_remote_code,
        "model_init_kwargs": None if args.adapter_path else model_init_kwargs,
    }
    sft_args = SFTConfig(**supported_kwargs(SFTConfig, config_kwargs))

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

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()
