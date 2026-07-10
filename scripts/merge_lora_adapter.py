#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into a local Hugging Face causal LM."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a base model.")
    parser.add_argument("--base-model", default="/raid/zkq/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-shard-size", default="4GB")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float16,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    merged = model.merge_and_unload()
    merged.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(output_dir)
    print(f"merged_model={output_dir}")


if __name__ == "__main__":
    main()
