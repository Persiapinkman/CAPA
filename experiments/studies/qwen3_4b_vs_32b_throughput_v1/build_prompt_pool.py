#!/usr/bin/env python3
"""Build deterministic, exact-token-length prompt pools for a local tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer


SENTENCES = [
    "A reproducible benchmark records the model, hardware, software, and workload before measurement.",
    "The service receives independent requests and returns a fixed number of generated tokens.",
    "Throughput and latency describe different properties and should be reported together.",
    "Warmup requests are excluded because initial kernels and caches differ from steady state.",
    "Each prompt is varied deterministically so prefix reuse cannot inflate the measured capacity.",
    "Long prompts emphasize prefill work, while fixed outputs make decode work directly comparable.",
    "Trial level repetitions are the unit used to estimate short term measurement uncertainty.",
    "Resource normalized results divide service throughput by the number of allocated accelerators.",
    "All raw request timings and device telemetry remain available for independent review.",
    "A deployment comparison is conditional on its inference engine and precision configuration.",
    "The experiment uses local loopback networking to minimize unrelated network variability.",
    "Errors and incomplete generations invalidate a trial instead of being removed silently.",
]


def sha256_json(value: object) -> str:
    encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_text(rng: random.Random, sample_index: int, minimum_chars: int) -> str:
    parts = [f"Benchmark sample {sample_index}; deterministic nonce {rng.getrandbits(64):016x}."]
    while sum(len(part) + 1 for part in parts) < minimum_chars:
        shuffled = list(SENTENCES)
        rng.shuffle(shuffled)
        parts.extend(shuffled)
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[256, 2048])
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts: dict[str, list[list[int]]] = {}
    decoded_hashes: dict[str, list[str]] = {}

    for length in args.lengths:
        rows: list[list[int]] = []
        row_hashes: list[str] = []
        for index in range(args.count):
            rng = random.Random(args.seed + length * 1_000_003 + index)
            text = make_text(rng, index, minimum_chars=length * 12)
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) < length:
                raise RuntimeError(f"prompt {index} for length {length} only has {len(token_ids)} tokens")
            token_ids = token_ids[:length]
            if len(token_ids) != length:
                raise AssertionError("token slicing failed")
            rows.append(token_ids)
            row_hashes.append(hashlib.sha256(tokenizer.decode(token_ids).encode("utf-8")).hexdigest())
        prompts[str(length)] = rows
        decoded_hashes[str(length)] = row_hashes

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(Path(args.model).resolve()),
        "label": args.label,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": len(tokenizer),
        "seed": args.seed,
        "count_per_length": args.count,
        "lengths": args.lengths,
        "decoded_prompt_sha256": decoded_hashes,
        "prompts": prompts,
    }
    payload["prompt_ids_sha256"] = sha256_json(prompts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False) + "\n")
    print(json.dumps({
        "schema_version": payload["schema_version"],
        "model_path": payload["model_path"],
        "label": payload["label"],
        "tokenizer_class": payload["tokenizer_class"],
        "tokenizer_vocab_size": payload["tokenizer_vocab_size"],
        "seed": payload["seed"],
        "count_per_length": payload["count_per_length"],
        "lengths": payload["lengths"],
        "prompt_ids_sha256": payload["prompt_ids_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
