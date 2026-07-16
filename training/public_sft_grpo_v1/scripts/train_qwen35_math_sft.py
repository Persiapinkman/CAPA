#!/usr/bin/env python3
"""Audited one-epoch LoRA SFT screen on 1,024 verifier-safe MATH rows."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig
from trl.chat_template_utils import (
    qwen3_5_nothink_chat_template,
    qwen3_5_nothink_training_chat_template,
)


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.public_sft_grpo_v1.scripts.math_verify_contract import (  # noqa: E402
    has_strict_terminal_box,
    parse_strict_boxed,
)
from training.public_sft_grpo_v1.scripts.train_qwen35_gsm8k_sft import (  # noqa: E402
    FiniteSFTCallback,
    TARGET_MODULES,
    V100SFTTrainer,
    model_audit,
    package_versions,
    sha256_file,
    summarize_logs,
    utc_now,
    write_json,
)


DEFAULT_MODEL = Path("/raid/zkq/models/Qwen3.5-4B")
DEFAULT_DATA_DIR = ROOT / "training/public_sft_grpo_v1/data/math_sft1024_v1"
DEFAULT_OUTPUT = ROOT / "experiments/runs/20260715_qwen35_4b_math_sft1024_screen_v1"
EXPECTED_MODEL_CLASS = "Qwen3_5ForCausalLM"
EXPECTED_EOS_ID = 248046
EXPECTED_PAD_ID = 248044


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["dry-run", "train"], default="dry-run")
    parser.add_argument("--confirm-train", action="store_true")
    parser.add_argument("--model-name-or-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-world-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--eval-steps", type=int, default=32)
    parser.add_argument("--save-steps", type=int, default=32)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-allocated-gib", type=float, default=28.0)
    parser.add_argument("--minimum-free-gib", type=float, default=2.0)
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args()


def load_and_verify_data(data_dir: Path) -> tuple[Any, dict[str, Any]]:
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["dataset_id"] != "public_sft_grpo_v1_math_sft1024":
        raise ValueError("unexpected MATH dataset manifest")
    for split in ("train", "development"):
        path = data_dir / f"{split}.jsonl"
        if sha256_file(path) != manifest["files"][split]["sha256"]:
            raise ValueError(f"data hash mismatch: {split}")
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "train.jsonl"),
            "development": str(data_dir / "development.jsonl"),
        },
    )
    if len(dataset["train"]) != 1024 or len(dataset["development"]) != 256:
        raise ValueError("MATH SFT1024 requires exact 1024/256 train/development splits")
    return dataset, manifest


def audit_labels(dataset: Any, tokenizer: Any, max_length: int) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for split in ("train", "development"):
        lengths: list[int] = []
        supervised: list[int] = []
        for row in dataset[split]:
            tokenized = tokenizer.apply_chat_template(
                row["messages"],
                chat_template=qwen3_5_nothink_training_chat_template,
                tokenize=True,
                return_dict=True,
                return_assistant_tokens_mask=True,
                add_generation_prompt=False,
            )
            ids = list(tokenized["input_ids"])
            mask = list(tokenized["assistant_masks"])
            if len(ids) > max_length or not any(mask) or len(ids) != len(mask):
                raise RuntimeError(f"invalid labels for {row['sample_id']}")
            supervised_ids = [token for token, enabled in zip(ids, mask, strict=True) if enabled]
            decoded = tokenizer.decode(supervised_ids, skip_special_tokens=False)
            if row["gold_solution"] not in decoded or "<|im_end|>" not in decoded:
                raise RuntimeError(f"supervised assistant/EOS mismatch: {row['sample_id']}")
            if not has_strict_terminal_box(row["gold_solution"]):
                raise RuntimeError(f"non-terminal gold box: {row['sample_id']}")
            if not parse_strict_boxed(row["gold_boxed"]):
                raise RuntimeError(f"unparseable gold box: {row['sample_id']}")
            if int(row["token_audit"]["total_tokens"]) != len(ids):
                raise RuntimeError(f"stored token audit drift: {row['sample_id']}")
            lengths.append(len(ids))
            supervised.append(sum(mask))
        stats[split] = {
            "rows": len(lengths),
            "min_tokens": min(lengths),
            "max_tokens": max(lengths),
            "min_supervised_tokens": min(supervised),
            "max_supervised_tokens": max(supervised),
            "assistant_mask_nonempty_rate": 1.0,
            "eos_supervised_rate": 1.0,
            "strict_terminal_box_rate": 1.0,
            "gold_parse_rate": 1.0,
        }
    return stats


def main() -> None:
    args = parse_args()
    started_at = utc_now()
    started_perf = time.perf_counter()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_main = rank == 0
    if args.mode == "train" and not args.confirm_train:
        raise PermissionError("optimizer steps require --confirm-train")
    if args.mode == "train" and world_size != args.expected_world_size:
        raise RuntimeError(f"expected world size {args.expected_world_size}, got {world_size}")
    if args.mode == "train" and args.expected_world_size != 4:
        raise ValueError("MATH SFT1024 v1 freezes a 4-rank topology")
    if (
        args.max_length != 2048
        or args.max_steps != 128
        or args.per_device_train_batch_size != 1
        or args.gradient_accumulation_steps != 2
    ):
        raise ValueError("MATH SFT1024 v1 freezes max_length=2048, steps=128, batch=1, grad_acc=2")

    model_path = args.model_name_or_path.resolve()
    data_dir = args.data_dir if args.data_dir.is_absolute() else ROOT / args.data_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if args.mode == "train" and output_dir.exists():
        raise FileExistsError(f"refusing to reuse training output directory: {output_dir}")
    dataset, data_manifest = load_and_verify_data(data_dir)
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    mapped_model_class = AutoModelForCausalLM._model_mapping[type(model_config)].__name__
    if mapped_model_class != EXPECTED_MODEL_CLASS:
        raise TypeError(f"expected causal mapping {EXPECTED_MODEL_CLASS}, got {mapped_model_class}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=False, use_fast=True, padding_side="right"
    )
    if tokenizer.eos_token_id != EXPECTED_EOS_ID or tokenizer.pad_token_id != EXPECTED_PAD_ID:
        raise RuntimeError("Qwen3.5 tokenizer stop contract changed")
    label_stats = audit_labels(dataset, tokenizer, args.max_length)
    versions = package_versions()
    versions.update(
        {
            "math-verify": importlib.metadata.version("math-verify"),
            "latex2sympy2-extended": importlib.metadata.version("latex2sympy2-extended"),
            "antlr4-python3-runtime": importlib.metadata.version("antlr4-python3-runtime"),
        }
    )
    config: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "prepared",
        "mode": args.mode,
        "started_at": started_at,
        "optimizer_steps_authorized": bool(args.mode == "train" and args.confirm_train),
        "model_name_or_path": str(model_path),
        "data_dir": str(data_dir),
        "data_manifest_sha256": sha256_file(data_dir / "manifest.json"),
        "output_dir": str(output_dir),
        "packages": versions,
        "distributed": {
            "world_size": world_size,
            "expected_world_size": args.expected_world_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "global_batch_size": args.expected_world_size * args.gradient_accumulation_steps,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
        "sft": {
            "max_steps": args.max_steps,
            "effective_epochs": 1.0,
            "max_length": args.max_length,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "assistant_only_loss": True,
            "completion_only_loss": False,
            "packing": False,
            "loss_type": "nll",
            "template": "qwen3_5_nothink_training",
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "model_loader": "AutoModelForCausalLM",
            "mapped_model_class": mapped_model_class,
        },
        "label_audit": label_stats,
        "lora": {"r": 16, "alpha": 32, "dropout": 0.0, "target_modules": TARGET_MODULES},
        "data_manifest": data_manifest,
    }
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "capa_qwen35_math_sft_config.json", config)

    torch.backends.cudnn.enabled = False
    set_seed(args.seed)
    original_chat_template = tokenizer.chat_template
    tokenizer.chat_template = qwen3_5_nothink_chat_template
    training_args = SFTConfig(
        output_dir=str(output_dir),
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=4,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        logging_nan_inf_filter=False,
        report_to=[],
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.0,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        max_grad_norm=1.0,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_cache=False,
        optim="adamw_torch",
        seed=args.seed,
        data_seed=args.seed,
        max_length=args.max_length,
        assistant_only_loss=True,
        completion_only_loss=False,
        packing=False,
        padding_free=False,
        loss_type="nll",
        eos_token="<|im_end|>",
        remove_unused_columns=True,
        ddp_find_unused_parameters=False,
        average_tokens_across_devices=True,
        torch_empty_cache_steps=1,
        torch_compile=False,
        trust_remote_code=False,
    )
    config["sft_config_signature_validated"] = True
    if is_main:
        write_json(output_dir / "capa_qwen35_math_sft_config.json", config)
    if args.mode == "dry-run":
        return

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    callback = FiniteSFTCallback(output_dir, args.max_allocated_gib, args.minimum_free_gib)
    trainer: V100SFTTrainer | None = None
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
            attn_implementation=args.attn_implementation,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        trainer = V100SFTTrainer(
            model=base_model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["development"],
            processing_class=tokenizer,
            peft_config=peft_config,
            callbacks=[callback],
        )
        audit = model_audit(trainer.model)
        scaler = trainer.accelerator.scaler
        if scaler is None or float(scaler.get_scale()) != 1.0:
            raise RuntimeError(f"unexpected SFT GradScaler: {scaler}")
        growth_interval = int(getattr(scaler, "_growth_interval", -1))
        if growth_interval != 100000:
            raise RuntimeError(f"unexpected SFT GradScaler growth interval: {growth_interval}")
        if is_main:
            config["model_audit"] = audit
            config["grad_scaler"] = {
                "scale": float(scaler.get_scale()),
                "growth_interval": growth_interval,
            }
            write_json(output_dir / "capa_qwen35_math_sft_config.json", config)
        train_result = trainer.train()
        trainer.save_model(str(output_dir))
        tokenizer.chat_template = original_chat_template
        if is_main:
            tokenizer.save_pretrained(str(output_dir))
            result = {
                "schema_version": "1.0",
                "status": "completed",
                "started_at": started_at,
                "finished_at": utc_now(),
                "runtime_seconds": time.perf_counter() - started_perf,
                "optimizer_steps": int(trainer.state.global_step),
                "train_metrics": dict(train_result.metrics),
                "log_summary": summarize_logs(list(trainer.state.log_history)),
                "model_audit": audit,
                "output_dir": str(output_dir),
            }
            write_json(output_dir / "capa_qwen35_math_sft_result.json", result)
    except Exception as exc:
        failure = {
            "schema_version": "1.0",
            "status": "fail",
            "rank": rank,
            "started_at": started_at,
            "finished_at": utc_now(),
            "optimizer_steps": int(trainer.state.global_step) if trainer is not None else 0,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }
        write_json(output_dir / "telemetry" / f"rank{rank}_failure.json", failure)
        raise
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
