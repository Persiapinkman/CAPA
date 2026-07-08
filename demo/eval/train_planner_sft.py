#!/usr/bin/env python3
"""Run LoRA SFT on the approved Planner DPO chosen responses."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

torch.backends.cudnn.enabled = False


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "training" / "planner_dpo_train_seed_v1" / "training_data"


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
    parser = argparse.ArgumentParser(description="Run Planner SFT on DPO chosen responses.")
    parser.add_argument("--model-name-or-path", default="/mnt/zkq/models/Qwen3.5-4B")
    parser.add_argument(
        "--train-file",
        default=str(DEFAULT_DATA_DIR / "planner_dpo_text_train.jsonl"),
        help="JSONL with prompt/chosen fields.",
    )
    parser.add_argument(
        "--validation-file",
        default=str(DEFAULT_DATA_DIR / "planner_dpo_text_val.jsonl"),
        help="Validation JSONL with prompt/chosen fields.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "planner-sft-qwen35-4b-chosen-lora"),
    )
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", type=parse_bool, default=False)
    parser.add_argument("--bf16", type=parse_bool, default=False)
    parser.add_argument("--gradient-checkpointing", type=parse_bool, default=True)
    parser.add_argument("--trust-remote-code", type=parse_bool, default=True)
    parser.add_argument("--use-lora", type=parse_bool, default=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated module names for LoRA.",
    )
    parser.add_argument("--report-to", default="tensorboard")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--ddp-backend", default=None)
    return parser.parse_args()


def normalize_dataset_columns(dataset):
    keep = {"prompt", "chosen"}
    missing = keep.difference(dataset.column_names)
    if missing:
        raise ValueError(f"dataset is missing required columns: {sorted(missing)}")
    remove = [name for name in dataset.column_names if name not in keep]
    if remove:
        dataset = dataset.remove_columns(remove)
    return dataset


def build_tokenize_fn(tokenizer, max_length: int):
    eos = tokenizer.eos_token or ""

    def tokenize_row(row: dict[str, Any]) -> dict[str, list[int]]:
        prompt = str(row.get("prompt") or "")
        chosen = str(row.get("chosen") or "")
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(chosen + eos, add_special_tokens=False)["input_ids"]
        if len(completion_ids) >= max_length:
            completion_ids = completion_ids[-max_length:]
            prompt_ids = []
        else:
            prompt_budget = max_length - len(completion_ids)
            prompt_ids = prompt_ids[-prompt_budget:]
        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    return tokenize_row


@dataclass
class CompletionOnlyCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        labels = [feature.pop("labels") for feature in features]
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        max_len = int(batch["input_ids"].shape[1])
        padded_labels = [
            label + [-100] * (max_len - len(label))
            for label in labels
        ]
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


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
        model.enable_input_require_grads()

    data_files = {"train": args.train_file}
    if args.validation_file:
        data_files["validation"] = args.validation_file
    raw_dataset = load_dataset("json", data_files=data_files)
    train_dataset = normalize_dataset_columns(raw_dataset["train"])
    eval_dataset = normalize_dataset_columns(raw_dataset["validation"]) if "validation" in raw_dataset else None

    tokenize_fn = build_tokenize_fn(tokenizer, max_length=args.max_length)
    train_dataset = train_dataset.map(tokenize_fn, remove_columns=train_dataset.column_names)
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(tokenize_fn, remove_columns=eval_dataset.column_names)

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
        from peft import get_peft_model

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    report_to = [] if str(args.report_to).lower() == "none" else [args.report_to]
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=eval_dataset is not None,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        report_to=report_to,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        optim="adamw_torch",
        seed=args.seed,
        ddp_find_unused_parameters=False,
        ddp_backend=args.ddp_backend,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=CompletionOnlyCollator(tokenizer),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
