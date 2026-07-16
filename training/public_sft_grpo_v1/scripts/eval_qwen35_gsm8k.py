#!/usr/bin/env python3
"""Deterministic GSM8K generation evaluation for Qwen3.5 base or LoRA."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.public_sft_grpo_v1.scripts.public_math_contract import (  # noqa: E402
    score_gsm8k_completion,
    strip_qwen_special_tokens,
    trim_completion_token_ids,
)


DEFAULT_DATA_DIR = ROOT / "training/public_sft_grpo_v1/data/gsm8k_sft32_v1"
DEFAULT_MODEL = Path("/raid/zkq/models/Qwen3.5-4B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", action="append", choices=["train", "development", "sealed_test"], default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * fraction)]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen3.5 evaluation")
    if args.batch_size < 1 or args.max_new_tokens < 1:
        raise ValueError("batch size and generation length must be positive")
    splits = args.split or ["train", "development"]
    data_dir = args.data_dir if args.data_dir.is_absolute() else ROOT / args.data_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output directory: {output_dir}")
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in splits:
        path = data_dir / f"{split}.jsonl"
        expected_hash = manifest["files"][split]["sha256"]
        if sha256_file(path) != expected_hash:
            raise ValueError(f"data hash mismatch for {split}")
        rows_by_split[split] = load_jsonl(path)

    set_seed(args.seed)
    torch.backends.cudnn.enabled = False
    model_path = args.model_name_or_path.resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=False, use_fast=True, padding_side="left"
    )
    if tokenizer.eos_token_id != 248046 or tokenizer.pad_token_id != 248044:
        raise RuntimeError("Qwen3.5 tokenizer stop contract changed")
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).to("cuda")
    adapter_path = args.adapter_path.resolve() if args.adapter_path is not None else None
    model = PeftModel.from_pretrained(base, adapter_path).to("cuda") if adapter_path else base
    model.eval()

    started = time.perf_counter()
    sample_rows: list[dict[str, Any]] = []
    for split, rows in rows_by_split.items():
        for offset in range(0, len(rows), args.batch_size):
            batch = rows[offset : offset + args.batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    row["messages"][:-1],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for row in batch
            ]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    remove_invalid_values=True,
                    renormalize_logits=True,
                    use_cache=True,
                )
            prompt_width = int(inputs["input_ids"].shape[1])
            for row, output in zip(batch, outputs, strict=True):
                completion_ids = output[prompt_width:].detach().cpu().tolist()
                natural_completion_ids, eos_found = trim_completion_token_ids(
                    completion_ids, tokenizer.eos_token_id
                )
                completion_raw = tokenizer.decode(natural_completion_ids, skip_special_tokens=False)
                completion = strip_qwen_special_tokens(completion_raw)
                score = score_gsm8k_completion(completion, row["ground_truth"])
                sample_rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "split": split,
                        "source_index": row["source_index"],
                        "gold_answer": row["ground_truth"],
                        "completion": completion,
                        "completion_token_count": len(natural_completion_ids),
                        "eos_found": eos_found,
                        "clipped": bool(not eos_found and len(completion_ids) >= args.max_new_tokens),
                        **score.as_dict(),
                    }
                )
            print(f"{split}: {min(offset + len(batch), len(rows))}/{len(rows)}", flush=True)

    metrics: dict[str, Any] = {}
    for split in splits:
        items = [row for row in sample_rows if row["split"] == split]
        lengths = [int(row["completion_token_count"]) for row in items]
        metrics[split] = {
            "rows": len(items),
            "exact_numeric_accuracy": sum(float(row["exact_numeric"]) for row in items) / len(items),
            "strict_format_rate": sum(float(row["strict_format"]) for row in items) / len(items),
            "loose_numeric_accuracy": sum(float(row["loose_numeric"]) for row in items) / len(items),
            "eos_rate": sum(float(row["eos_found"]) for row in items) / len(items),
            "clipped_rate": sum(float(row["clipped"]) for row in items) / len(items),
            "completion_tokens": {
                "min": min(lengths),
                "p50": percentile(lengths, 0.5),
                "p95": percentile(lengths, 0.95),
                "max": max(lengths),
                "mean": sum(lengths) / len(lengths),
            },
        }
    result = {
        "schema_version": "1.0",
        "status": "completed",
        "finished_at": utc_now(),
        "runtime_seconds": time.perf_counter() - started,
        "model_name_or_path": str(model_path),
        "adapter_path": str(adapter_path) if adapter_path else "",
        "data_dir": str(data_dir),
        "dataset_manifest_sha256": sha256_file(data_dir / "manifest.json"),
        "generation": {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "metrics": metrics,
        "nonfinite_metric_count": sum(
            not math.isfinite(float(value))
            for split_metrics in metrics.values()
            for value in (
                split_metrics["exact_numeric_accuracy"],
                split_metrics["strict_format_rate"],
                split_metrics["loose_numeric_accuracy"],
            )
        ),
    }
    write_jsonl(output_dir / "samples.jsonl", sample_rows)
    result["samples_sha256"] = sha256_file(output_dir / "samples.jsonl")
    write_json(output_dir / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
