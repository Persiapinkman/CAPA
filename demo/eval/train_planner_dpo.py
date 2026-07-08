#!/usr/bin/env python3
"""Train the Planner router with TRL DPO on reviewed preference pairs."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOConfig, DPOTrainer

torch.backends.cudnn.enabled = False


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "results" / "planner_routing_eval" / "dpo_train_seed_v1" / "training_data"


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
    parser = argparse.ArgumentParser(description="Run TRL DPO training for the Planner router.")
    parser.add_argument(
        "--model-name-or-path",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Base model path or Hugging Face model id.",
    )
    parser.add_argument(
        "--train-file",
        default=str(DEFAULT_DATA_DIR / "planner_dpo_text_train.jsonl"),
        help="JSONL with prompt/chosen/rejected fields.",
    )
    parser.add_argument(
        "--validation-file",
        default=str(DEFAULT_DATA_DIR / "planner_dpo_text_val.jsonl"),
        help="Validation JSONL with prompt/chosen/rejected fields.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "planner-dpo-qwen25-05b-lora"),
        help="Directory for checkpoints and final adapter/model.",
    )
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-prompt-length", type=int, default=3072)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", type=parse_bool, default=True)
    parser.add_argument("--bf16", type=parse_bool, default=False)
    parser.add_argument("--gradient-checkpointing", type=parse_bool, default=True)
    parser.add_argument("--trust-remote-code", type=parse_bool, default=True)
    parser.add_argument("--use-lora", type=parse_bool, default=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names for LoRA.",
    )
    parser.add_argument(
        "--report-to",
        default="tensorboard",
        help='Trainer reporting backend. Use "none" to disable.',
    )
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser.parse_args()


def normalize_dataset_columns(dataset):
    keep = {"prompt", "chosen", "rejected"}
    missing = keep.difference(dataset.column_names)
    if missing:
        raise ValueError(f"dataset is missing required columns: {sorted(missing)}")
    remove = [name for name in dataset.column_names if name not in keep]
    if remove:
        dataset = dataset.remove_columns(remove)
    return dataset


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    if args.gradient_checkpointing:
        model.config.use_cache = False

    data_files = {"train": args.train_file}
    if args.validation_file:
        data_files["validation"] = args.validation_file
    dataset = load_dataset("json", data_files=data_files)
    train_dataset = normalize_dataset_columns(dataset["train"])
    eval_dataset = normalize_dataset_columns(dataset["validation"]) if "validation" in dataset else None

    peft_config = None
    if args.use_lora:
        target_modules = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )

    report_to = [] if str(args.report_to).lower() == "none" else [args.report_to]
    dpo_kwargs = {
        "output_dir": args.output_dir,
        "do_train": True,
        "do_eval": eval_dataset is not None,
        "eval_strategy": "steps" if eval_dataset is not None else "no",
        "eval_steps": args.eval_steps if eval_dataset is not None else None,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "logging_steps": args.logging_steps,
        "logging_first_step": True,
        "report_to": report_to,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "beta": args.beta,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "truncation_mode": "keep_end",
        "remove_unused_columns": False,
        "optim": "adamw_torch",
    }
    supported_dpo_args = set(inspect.signature(DPOConfig.__init__).parameters)
    training_args = DPOConfig(
        **{key: value for key, value in dpo_kwargs.items() if key in supported_dpo_args}
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
