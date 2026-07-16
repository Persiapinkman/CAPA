#!/usr/bin/env python3
"""Audited completion-only LoRA SFT for Qwen3.5-4B on Planner V6."""

from __future__ import annotations

import argparse
import hashlib
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


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.public_sft_grpo_v1.scripts.train_qwen35_gsm8k_sft import (  # noqa: E402
    EXPECTED_EOS_ID,
    EXPECTED_MODEL_CLASS,
    EXPECTED_PAD_ID,
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
from training.wandb_observability import (  # noqa: E402
    DEFAULT_WANDB_GROUP,
    DEFAULT_WANDB_PROJECT,
    configure_wandb_environment,
    metric_contract,
    parse_report_to,
)


DATASET_ID = "planner_retry_migrate_v6"
DEFAULT_MODEL = Path("/raid/zkq/models/Qwen3.5-4B")
DEFAULT_DATA_DIR = (
    ROOT
    / "training/planner_grpo_seed_v1/"
    "sft_data_planner_retry_migrate_v6_qwen35_nothinking"
)
DEFAULT_OUTPUT = ROOT / "experiments/runs/20260716_qwen35_4b_planner_v6_sft_seed42"
NONTHINKING_SUFFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
EXPECTED_ROWS = {"train": 1040, "dev": 260}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["dry-run", "train"], default="dry-run")
    parser.add_argument("--confirm-train", action="store_true")
    parser.add_argument("--model-name-or-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-world-size", type=int, default=6)
    parser.add_argument("--max-length", type=int, default=4800)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--eval-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-allocated-gib", type=float, default=28.0)
    parser.add_argument("--minimum-free-gib", type=float, default=2.0)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument(
        "--report-to",
        default="auto",
        help="HF reporting backends. 'auto' enables W&B for train and disables it for dry-run.",
    )
    parser.add_argument("--run-name", default="")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", DEFAULT_WANDB_PROJECT))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument("--wandb-group", default=os.environ.get("WANDB_RUN_GROUP", DEFAULT_WANDB_GROUP))
    parser.add_argument("--wandb-tags", default="stage1,sft,planner-v6")
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    return parser.parse_args()


def load_and_verify_data(data_dir: Path) -> tuple[Any, dict[str, Any]]:
    metadata_path = data_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("dataset_id") != DATASET_ID:
        raise ValueError(f"unexpected dataset_id: {metadata.get('dataset_id')}")
    if metadata.get("prompt_contract", {}).get("chat_template") != "native_qwen35":
        raise ValueError("V6 SFT data is not frozen with the native Qwen3.5 template")
    if metadata.get("prompt_contract", {}).get("enable_thinking") is not False:
        raise ValueError("V6 SFT data must use enable_thinking=false")
    if metadata.get("audits", {}).get("sft_train", {}).get("status") != "pass":
        raise ValueError("frozen SFT-train prompt audit is not passing")
    if metadata.get("audits", {}).get("sft_dev", {}).get("status") != "pass":
        raise ValueError("frozen SFT-dev prompt audit is not passing")

    paths = {"train": data_dir / "train.jsonl", "dev": data_dir / "dev.jsonl"}
    expected_hashes = {
        "train": metadata.get("sha256", {}).get("sft_train"),
        "dev": metadata.get("sha256", {}).get("sft_dev"),
    }
    for split, path in paths.items():
        actual = sha256_file(path)
        if actual != expected_hashes[split]:
            raise ValueError(
                f"{split} data hash mismatch: expected={expected_hashes[split]}, actual={actual}"
            )
    dataset = load_dataset(
        "json",
        data_files={"train": str(paths["train"]), "dev": str(paths["dev"])},
    )
    for split, expected_rows in EXPECTED_ROWS.items():
        if len(dataset[split]) != expected_rows:
            raise ValueError(f"{split}: expected {expected_rows} rows, got {len(dataset[split])}")
    return dataset, metadata


def audit_labels(dataset: Any, tokenizer: Any, max_length: int) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for split in ("train", "dev"):
        lengths: list[int] = []
        prompt_lengths: list[int] = []
        completion_lengths: list[int] = []
        source_case_counts: list[int] = []
        for row in dataset[split]:
            prompt = str(row.get("prompt") or "")
            completion = str(row.get("completion") or "")
            if not prompt.endswith(NONTHINKING_SUFFIX):
                raise RuntimeError(f"{row.get('case_id')}: invalid non-thinking prompt tail")
            if not completion.endswith("<|im_end|>"):
                raise RuntimeError(f"{row.get('case_id')}: completion lacks <|im_end|>")
            if str(row.get("case_id") or "") in prompt or str(row.get("entity_id") or "") in prompt:
                raise RuntimeError(f"{row.get('case_id')}: metadata identity leaked into prompt")
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(prompt + completion, add_special_tokens=False)["input_ids"]
            if len(prompt_ids) != int(row.get("prompt_token_count") or -1):
                raise RuntimeError(f"{row.get('case_id')}: prompt token count drift")
            if len(completion_ids) != int(row.get("completion_token_count") or -1):
                raise RuntimeError(f"{row.get('case_id')}: completion token count drift")
            if not completion_ids or len(full_ids) > max_length:
                raise RuntimeError(
                    f"{row.get('case_id')}: invalid supervised length {len(full_ids)}/{max_length}"
                )
            decoded = tokenizer.decode(completion_ids, skip_special_tokens=False)
            if "<|im_end|>" not in decoded:
                raise RuntimeError(f"{row.get('case_id')}: EOS is not supervised")
            lengths.append(len(full_ids))
            prompt_lengths.append(len(prompt_ids))
            completion_lengths.append(len(completion_ids))
            source_case_counts.append(int(row.get("source_case_count") or 1))
        ordered = sorted(lengths)
        stats[split] = {
            "rows": len(lengths),
            "min_tokens": min(lengths),
            "mean_tokens": sum(lengths) / len(lengths),
            "p50_tokens": ordered[len(ordered) // 2],
            "p95_tokens": ordered[int((len(ordered) - 1) * 0.95)],
            "max_tokens": max(lengths),
            "min_prompt_tokens": min(prompt_lengths),
            "max_prompt_tokens": max(prompt_lengths),
            "min_supervised_tokens": min(completion_lengths),
            "max_supervised_tokens": max(completion_lengths),
            "eos_supervised_rate": 1.0,
            "deduplicated_source_cases": sum(source_case_counts),
        }
    return stats


def model_contract(model_path: Path) -> dict[str, Any]:
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    mapped = AutoModelForCausalLM._model_mapping[type(model_config)].__name__
    if mapped != EXPECTED_MODEL_CLASS:
        raise TypeError(f"expected causal mapping {EXPECTED_MODEL_CLASS}, got {mapped}")
    files = {
        "config": model_path / "config.json",
        "tokenizer_config": model_path / "tokenizer_config.json",
        "chat_template": model_path / "chat_template.jinja",
    }
    return {
        "mapped_model_class": mapped,
        "sha256": {name: sha256_file(path) for name, path in files.items()},
    }


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
    if args.mode == "train" and args.expected_world_size not in {4, 6}:
        raise ValueError("audited V6 SFT topology requires 4 or 6 ranks")
    if args.max_length != 4800 or args.per_device_train_batch_size != 1:
        raise ValueError("V6 SFT freezes max_length=4800 and per-device batch=1")

    model_path = args.model_name_or_path.resolve()
    data_dir = args.data_dir if args.data_dir.is_absolute() else ROOT / args.data_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if args.mode == "train" and output_dir.exists() and not args.resume_from_checkpoint:
        raise FileExistsError(f"refusing to reuse training output directory: {output_dir}")

    effective_report_to = (
        "wandb" if args.report_to == "auto" and args.mode == "train" else
        "none" if args.report_to == "auto" else args.report_to
    )
    report_to = parse_report_to(effective_report_to)
    run_name = args.run_name or output_dir.name
    wandb_settings = configure_wandb_environment(
        report_to=report_to,
        output_dir=output_dir,
        stage="sft",
        run_name=run_name,
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        tags=args.wandb_tags,
        mode=args.wandb_mode,
    )

    dataset, data_metadata = load_and_verify_data(data_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=False,
        use_fast=True,
        padding_side="right",
        truncation_side="right",
    )
    if tokenizer.eos_token_id != EXPECTED_EOS_ID or tokenizer.pad_token_id != EXPECTED_PAD_ID:
        raise RuntimeError(
            f"tokenizer stop contract changed: eos={tokenizer.eos_token_id}, pad={tokenizer.pad_token_id}"
        )
    label_stats = audit_labels(dataset, tokenizer, args.max_length)
    contract = model_contract(model_path)
    config: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "prepared",
        "mode": args.mode,
        "started_at": started_at,
        "optimizer_steps_authorized": bool(args.mode == "train" and args.confirm_train),
        "model_name_or_path": str(model_path),
        "model_contract": contract,
        "data_dir": str(data_dir),
        "data_metadata_sha256": sha256_file(data_dir / "metadata.json"),
        "output_dir": str(output_dir),
        "packages": package_versions(),
        "distributed": {
            "world_size": world_size,
            "expected_world_size": args.expected_world_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "global_batch_size": args.expected_world_size * args.gradient_accumulation_steps,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
        "sft": {
            "max_steps": args.max_steps,
            "max_length": args.max_length,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "completion_only_loss": True,
            "assistant_only_loss": False,
            "packing": False,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
        "label_audit": label_stats,
        "lora": {"r": 16, "alpha": 32, "dropout": 0.0, "target_modules": TARGET_MODULES},
        "hardware_guards": {
            "cudnn_enabled": False,
            "max_allocated_gib": args.max_allocated_gib,
            "minimum_free_gib": args.minimum_free_gib,
            "grad_scaler_init_scale": 1.0,
            "grad_scaler_growth_interval": 100000,
        },
        "observability": {
            "wandb": wandb_settings,
            "metric_contract": metric_contract("sft"),
            "dashboard_config": "configs/wandb/post_training_v1.json",
        },
        "data_metadata": data_metadata,
    }
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "capa_qwen35_planner_v6_sft_config.json", config)
        print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)
    if args.mode == "dry-run":
        return

    torch.backends.cudnn.enabled = False
    set_seed(args.seed)
    training_args = SFTConfig(
        output_dir=str(output_dir),
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        logging_nan_inf_filter=False,
        report_to=report_to,
        run_name=run_name,
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
        completion_only_loss=True,
        assistant_only_loss=False,
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
            eval_dataset=dataset["dev"],
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
            write_json(output_dir / "capa_qwen35_planner_v6_sft_config.json", config)
        train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        trainer.save_model(str(output_dir))
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
                "nonfinite_state_values": sum(
                    not math.isfinite(float(value))
                    for entry in trainer.state.log_history
                    for key, value in entry.items()
                    if key in {"loss", "eval_loss", "grad_norm"} and value is not None
                ),
                "output_dir": str(output_dir),
            }
            write_json(output_dir / "capa_qwen35_planner_v6_sft_result.json", result)
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
