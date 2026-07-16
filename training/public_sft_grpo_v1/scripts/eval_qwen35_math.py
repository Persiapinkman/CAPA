#!/usr/bin/env python3
"""Deterministic sharded MATH generation evaluation for Qwen3.5 base or LoRA."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.public_sft_grpo_v1.scripts.math_verify_contract import (  # noqa: E402
    score_math_completion,
)
from training.public_sft_grpo_v1.scripts.public_math_contract import (  # noqa: E402
    strip_qwen_special_tokens,
    trim_completion_token_ids,
)


DEFAULT_DATA_DIR = ROOT / "training/public_sft_grpo_v1/data/math_sft1024_v1"
DEFAULT_MODEL = Path("/raid/zkq/models/Qwen3.5-4B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--split", action="append", choices=["train", "development", "sealed_test"], default=[]
    )
    parser.add_argument("--allow-sealed-test", action="store_true")
    parser.add_argument("--train-limit", type=int, default=128)
    parser.add_argument("--development-limit", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
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


def sha256_values(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


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


def metric_block(items: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [int(row["completion_token_count"]) for row in items]
    return {
        "rows": len(items),
        "symbolic_accuracy": sum(float(row["symbolic_accuracy"]) for row in items) / len(items),
        "strict_format_rate": sum(float(row["strict_format"]) for row in items) / len(items),
        "strict_exact_accuracy": sum(float(row["strict_exact"]) for row in items) / len(items),
        "parse_success_rate": sum(float(row["parse_success"]) for row in items) / len(items),
        "eos_rate": sum(float(row["eos_found"]) for row in items) / len(items),
        "clipped_rate": sum(float(row["clipped"]) for row in items) / len(items),
        "completion_tokens": {
            "min": min(lengths),
            "p50": percentile(lengths, 0.50),
            "p95": percentile(lengths, 0.95),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
        },
    }


def build_metrics(sample_rows: list[dict[str, Any]], splits: list[str]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for split in splits:
        items = [row for row in sample_rows if row["split"] == split]
        if not items:
            continue
        by_level: dict[str, Any] = {}
        by_type: dict[str, Any] = {}
        grouped_level: dict[str, list[dict[str, Any]]] = defaultdict(list)
        grouped_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in items:
            grouped_level[row["level"]].append(row)
            grouped_type[row["type"]].append(row)
        for key, group in sorted(grouped_level.items()):
            by_level[key] = metric_block(group)
        for key, group in sorted(grouped_type.items()):
            by_type[key] = metric_block(group)
        metrics[split] = {
            "overall": metric_block(items),
            "by_level": by_level,
            "by_type": by_type,
        }
    return metrics


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen3.5 MATH evaluation")
    if args.batch_size < 1 or args.max_new_tokens < 1:
        raise ValueError("batch size and generation length must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard topology")
    splits = args.split or ["train", "development"]
    if "sealed_test" in splits and not args.allow_sealed_test:
        raise PermissionError("sealed_test requires explicit --allow-sealed-test")
    if args.train_limit != 128 or args.development_limit != 256:
        raise ValueError("MATH screen v1 freezes train/development generation limits at 128/256")
    data_dir = args.data_dir if args.data_dir.is_absolute() else ROOT / args.data_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output directory: {output_dir}")
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    limits = {"train": args.train_limit, "development": args.development_limit}
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    selected_all_ids: list[str] = []
    shard_all_ids: list[str] = []
    for split in splits:
        path = data_dir / f"{split}.jsonl"
        if sha256_file(path) != manifest["files"][split]["sha256"]:
            raise ValueError(f"data hash mismatch for {split}")
        rows = load_jsonl(path)
        selected = rows[: limits.get(split, len(rows))]
        shard_rows = [row for offset, row in enumerate(selected) if offset % args.num_shards == args.shard_index]
        rows_by_split[split] = shard_rows
        selected_all_ids.extend(row["sample_id"] for row in selected)
        shard_all_ids.extend(row["sample_id"] for row in shard_rows)

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
            inputs = tokenizer(
                prompts, return_tensors="pt", padding=True, add_special_tokens=False
            ).to("cuda")
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
                natural_ids, eos_found = trim_completion_token_ids(
                    completion_ids, tokenizer.eos_token_id
                )
                completion_raw = tokenizer.decode(natural_ids, skip_special_tokens=False)
                completion = strip_qwen_special_tokens(completion_raw)
                score = score_math_completion(completion, row["gold_boxed"])
                sample_rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "split": split,
                        "source_index": row["source_index"],
                        "level": row["level"],
                        "type": row["type"],
                        "gold_boxed": row["gold_boxed"],
                        "completion": completion,
                        "completion_token_count": len(natural_ids),
                        "eos_found": eos_found,
                        "clipped": bool(
                            not eos_found and len(completion_ids) >= args.max_new_tokens
                        ),
                        **score.as_dict(),
                    }
                )
            print(
                f"shard {args.shard_index}/{args.num_shards} {split}: "
                f"{min(offset + len(batch), len(rows))}/{len(rows)}",
                flush=True,
            )

    metrics = build_metrics(sample_rows, splits)
    result = {
        "schema_version": "1.0",
        "status": "completed",
        "finished_at": utc_now(),
        "runtime_seconds": time.perf_counter() - started,
        "model_name_or_path": str(model_path),
        "adapter_path": str(adapter_path) if adapter_path else "",
        "data_dir": str(data_dir),
        "dataset_manifest_sha256": sha256_file(data_dir / "manifest.json"),
        "packages": {
            "math-verify": importlib.metadata.version("math-verify"),
            "latex2sympy2-extended": importlib.metadata.version("latex2sympy2-extended"),
        },
        "generation": {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "train_limit": args.train_limit,
            "development_limit": args.development_limit,
        },
        "sharding": {
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "global_selected_rows": len(selected_all_ids),
            "shard_rows": len(shard_all_ids),
            "global_selected_ids_sha256": sha256_values(selected_all_ids),
            "shard_ids_sha256": sha256_values(shard_all_ids),
        },
        "metrics": metrics,
        "nonfinite_metric_count": sum(
            not math.isfinite(float(value))
            for split_metrics in metrics.values()
            for value in (
                split_metrics["overall"]["symbolic_accuracy"],
                split_metrics["overall"]["strict_format_rate"],
                split_metrics["overall"]["strict_exact_accuracy"],
            )
        ),
    }
    write_jsonl(output_dir / "samples.jsonl", sample_rows)
    result["samples_sha256"] = sha256_file(output_dir / "samples.jsonl")
    write_json(output_dir / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
