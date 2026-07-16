#!/usr/bin/env python3
"""Freeze stratified, verifier-safe MATH splits for the Qwen3.5 SFT screen."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from transformers import AutoTokenizer
from trl.chat_template_utils import qwen3_5_nothink_training_chat_template


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.public_sft_grpo_v1.scripts.math_verify_contract import (  # noqa: E402
    extract_boxed_spans,
    has_strict_terminal_box,
    normalize_solution_with_terminal_box,
    parse_strict_boxed,
)


MODEL_PATH = Path("/raid/zkq/models/Qwen3.5-4B")
ARTIFACT_ROOT = Path("/raid/zkq/artifacts/CAPA/datasets/public_sft_grpo_v1/raw/hf")
MATH_REVISION = "0530c78699ea5e8eb5530600900e1f328b48acad"
MATH_DIR = ARTIFACT_ROOT / "DigitalLearningGmbH__MATH-lighteval" / MATH_REVISION
TRAIN_SOURCE = MATH_DIR / "data/train-00000-of-00001.parquet"
TEST_SOURCE = MATH_DIR / "data/test-00000-of-00001.parquet"
DEFAULT_OUTPUT = ROOT / "training/public_sft_grpo_v1/data/math_sft1024_v1"
SYSTEM_PROMPT = (
    "You are a careful competition math assistant. Solve the problem step by step. "
    "End with exactly one final line of the form \\boxed{answer}. Do not put any text "
    "after that boxed answer."
)
SOURCE_HASHES = {
    MATH_DIR / "README.md": "3bb056303034d0f49925396ede385ee2feed638c26e2abbf3a8ff0b9976a9bc4",
    TRAIN_SOURCE: "eca6e667f4305dd5e5ba09b4fd55e7f3174a0fbe361cdfd4c44758b593a76933",
    TEST_SOURCE: "7dca8d6e41af88ecf82f2b5f36eb5530e083aaaa86ee325f62bd5c31535178c6",
}
EXPECTED_LEVELS = {f"Level {level}" for level in range(1, 6)}
EXPECTED_TYPES = {
    "Algebra",
    "Counting & Probability",
    "Geometry",
    "Intermediate Algebra",
    "Number Theory",
    "Prealgebra",
    "Precalculus",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-name-or-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-rows", type=int, default=1024)
    parser.add_argument("--dev-rows", type=int, default=256)
    parser.add_argument("--sealed-test-rows", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--parse-workers", type=int, default=16)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def normalized_question(text: str) -> str:
    return " ".join(str(text).split())


def selection_key(question: str, seed: int, namespace: str) -> str:
    payload = f"{seed}\n{namespace}\n{normalized_question(question)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * fraction)]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def audit_sources() -> None:
    for path, expected_hash in SOURCE_HASHES.items():
        if not path.exists():
            raise FileNotFoundError(path)
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"source hash mismatch for {path}: {actual_hash} != {expected_hash}")


def parse_gold_task(task: tuple[str, int, str]) -> tuple[str, int, bool, str]:
    source_split, source_index, gold_boxed = task
    try:
        parsed = parse_strict_boxed(gold_boxed, timeout_seconds=2, max_box_chars=1024)
        return source_split, source_index, bool(parsed), repr(parsed)[:512] if parsed else ""
    except BaseException:
        return source_split, source_index, False, ""


def label_audit(
    messages: list[dict[str, str]],
    sample_id: str,
    gold_solution: str,
    tokenizer: Any,
    max_length: int,
) -> dict[str, Any]:
    tokenized = tokenizer.apply_chat_template(
        messages,
        chat_template=qwen3_5_nothink_training_chat_template,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        add_generation_prompt=False,
    )
    ids = list(tokenized["input_ids"])
    mask = list(tokenized["assistant_masks"])
    if len(ids) != len(mask) or not any(mask):
        raise RuntimeError(f"invalid assistant mask for {sample_id}")
    assistant_spans = sum(
        bool(enabled) and (offset == 0 or not mask[offset - 1])
        for offset, enabled in enumerate(mask)
    )
    if assistant_spans != 1:
        raise RuntimeError(f"expected one assistant span for {sample_id}, got {assistant_spans}")
    supervised_ids = [token for token, enabled in zip(ids, mask, strict=True) if enabled]
    supervised_text = tokenizer.decode(supervised_ids, skip_special_tokens=False)
    if gold_solution not in supervised_text or "<|im_end|>" not in supervised_text:
        raise RuntimeError(f"assistant span/EOS mismatch for {sample_id}")
    prompt = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=True,
        return_dict=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return {
        "total_tokens": len(ids),
        "assistant_tokens": sum(mask),
        "assistant_fraction": sum(mask) / len(mask),
        "prompt_tokens": len(prompt["input_ids"]),
        "within_max_length": len(ids) <= max_length,
        "eos_supervised": True,
        "assistant_spans": assistant_spans,
    }


def proportional_quotas(counts: dict[tuple[str, str], int], total: int) -> dict[tuple[str, str], int]:
    keys = sorted(counts)
    if total < len(keys):
        raise ValueError(f"total={total} cannot cover all {len(keys)} strata")
    quotas = {key: 1 for key in keys}
    remaining = total - len(keys)
    denominator = sum(counts.values())
    raw = {key: remaining * counts[key] / denominator for key in keys}
    for key in keys:
        quotas[key] += math.floor(raw[key])
    leftovers = total - sum(quotas.values())
    remainder_order = sorted(keys, key=lambda key: (-(raw[key] - math.floor(raw[key])), key))
    for key in remainder_order[:leftovers]:
        quotas[key] += 1
    for key, quota in quotas.items():
        if quota > counts[key]:
            raise ValueError(f"quota {quota} exceeds stratum capacity {counts[key]} for {key}")
    if sum(quotas.values()) != total:
        raise RuntimeError("quota allocation did not reach requested total")
    return quotas


def summarize_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = [int(row["token_audit"]["total_tokens"]) for row in rows]
    assistants = [int(row["token_audit"]["assistant_tokens"]) for row in rows]
    prompts = [int(row["token_audit"]["prompt_tokens"]) for row in rows]
    by_level = Counter(row["level"] for row in rows)
    by_type = Counter(row["type"] for row in rows)
    by_stratum = Counter(f"{row['level']}|{row['type']}" for row in rows)
    return {
        "rows": len(rows),
        "total_tokens": {
            "min": min(totals),
            "p50": percentile(totals, 0.50),
            "p90": percentile(totals, 0.90),
            "p95": percentile(totals, 0.95),
            "p99": percentile(totals, 0.99),
            "max": max(totals),
        },
        "assistant_tokens": {
            "min": min(assistants),
            "p50": percentile(assistants, 0.50),
            "p95": percentile(assistants, 0.95),
            "p99": percentile(assistants, 0.99),
            "max": max(assistants),
        },
        "prompt_tokens": {
            "min": min(prompts),
            "p50": percentile(prompts, 0.50),
            "p95": percentile(prompts, 0.95),
            "p99": percentile(prompts, 0.99),
            "max": max(prompts),
        },
        "by_level": dict(sorted(by_level.items())),
        "by_type": dict(sorted(by_type.items())),
        "by_stratum": dict(sorted(by_stratum.items())),
        "assistant_mask_nonempty_rate": 1.0,
        "eos_supervised_rate": 1.0,
        "strict_terminal_box_rate": 1.0,
        "gold_parse_rate": 1.0,
    }


def main() -> None:
    args = parse_args()
    if args.max_length != 2048:
        raise ValueError("MATH SFT1024 v1 freezes max_length=2048")
    if args.parse_workers < 1 or args.parse_workers > 32:
        raise ValueError("parse-workers must be in [1, 32]")
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    audit_sources()

    source_rows = {
        "train": pq.read_table(TRAIN_SOURCE).to_pylist(),
        "test": pq.read_table(TEST_SOURCE).to_pylist(),
    }
    if len(source_rows["train"]) != 7500 or len(source_rows["test"]) != 5000:
        raise ValueError("unexpected MATH-lighteval source sizes")

    normalized_by_split = {
        split: [normalized_question(row["problem"]) for row in rows]
        for split, rows in source_rows.items()
    }
    cross_overlap = set(normalized_by_split["train"]) & set(normalized_by_split["test"])
    if len(cross_overlap) != 1:
        raise ValueError(f"expected one known upstream train/test overlap, got {len(cross_overlap)}")

    filter_counts: dict[str, Counter[str]] = {"train": Counter(), "test": Counter()}
    preparse: dict[tuple[str, int], dict[str, Any]] = {}
    parse_tasks: list[tuple[str, int, str]] = []
    for source_split, rows in source_rows.items():
        seen_questions: set[str] = set()
        for source_index, source in enumerate(rows):
            question = normalized_by_split[source_split][source_index]
            if question in cross_overlap:
                filter_counts[source_split]["upstream_cross_overlap"] += 1
                continue
            if question in seen_questions:
                filter_counts[source_split]["duplicate_normalized_question"] += 1
                continue
            seen_questions.add(question)
            if source["level"] not in EXPECTED_LEVELS or source["type"] not in EXPECTED_TYPES:
                filter_counts[source_split]["unexpected_level_or_type"] += 1
                continue
            if "\\boxed" in source["problem"]:
                filter_counts[source_split]["problem_contains_boxed"] += 1
                continue
            try:
                spans = extract_boxed_spans(source["solution"])
            except ValueError:
                filter_counts[source_split]["unclosed_boxed_gold"] += 1
                continue
            if len(spans) != 1:
                filter_counts[source_split][f"boxed_count_{len(spans)}"] += 1
                continue
            try:
                normalized_solution, gold_boxed, gold_content = normalize_solution_with_terminal_box(
                    source["solution"]
                )
            except ValueError:
                filter_counts[source_split]["empty_or_invalid_single_boxed_gold"] += 1
                continue
            record = {
                "source": source,
                "normalized_question": question,
                "normalized_solution": normalized_solution,
                "gold_boxed": gold_boxed,
                "gold_content": gold_content,
            }
            preparse[(source_split, source_index)] = record
            parse_tasks.append((source_split, source_index, gold_boxed))

    parsed_gold: dict[tuple[str, int], str] = {}
    with ProcessPoolExecutor(max_workers=args.parse_workers) as executor:
        for source_split, source_index, parse_ok, parsed_repr in executor.map(
            parse_gold_task, parse_tasks, chunksize=32
        ):
            if parse_ok:
                parsed_gold[(source_split, source_index)] = parsed_repr
            else:
                filter_counts[source_split]["unparseable_strict_gold"] += 1

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path.resolve(), trust_remote_code=False, use_fast=True
    )
    if tokenizer.eos_token_id != 248046 or tokenizer.pad_token_id != 248044:
        raise RuntimeError("Qwen3.5 tokenizer stop contract changed")

    eligible: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    for (source_split, source_index), record in preparse.items():
        if (source_split, source_index) not in parsed_gold:
            continue
        source = record["source"]
        question_sha = sha256_text(record["normalized_question"])
        sample_id = f"math-{source_split}-{source_index:05d}-{question_sha[:12]}"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(source["problem"])},
            {"role": "assistant", "content": record["normalized_solution"]},
        ]
        audit = label_audit(
            messages,
            sample_id,
            record["normalized_solution"],
            tokenizer,
            args.max_length,
        )
        if not audit["within_max_length"]:
            filter_counts[source_split]["over_max_length"] += 1
            continue
        if not has_strict_terminal_box(record["normalized_solution"]):
            raise RuntimeError(f"terminal box normalization failed for {sample_id}")
        eligible[source_split].append(
            {
                "sample_id": sample_id,
                "dataset_id": "public_sft_grpo_v1",
                "source_dataset": "DigitalLearningGmbH/MATH-lighteval",
                "source_revision": MATH_REVISION,
                "source_config": "default",
                "source_split": source_split,
                "source_index": source_index,
                "question_sha256": question_sha,
                "level": str(source["level"]),
                "type": str(source["type"]),
                "messages": messages,
                "chat_template_kwargs": {"enable_thinking": False},
                "verifier_type": "math_verify_strict_boxed_v1",
                "ground_truth": record["gold_boxed"],
                "gold_boxed": record["gold_boxed"],
                "gold_box_content": record["gold_content"],
                "gold_parse_repr": parsed_gold[(source_split, source_index)],
                "gold_solution": record["normalized_solution"],
                "source_solution_sha256": sha256_text(source["solution"]),
                "token_audit": audit,
                "reward_metadata": {
                    "safe_accuracy_weight": 0.95,
                    "safe_format_weight": 0.05,
                    "safe_accuracy_requires_strict_terminal_box": True,
                    "format_contract": "exactly one final line: \\boxed{answer}",
                    "verify_argument_order": "gold_then_prediction",
                    "allow_set_relation_comp": False,
                },
            }
        )

    train_strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible["train"]:
        train_strata[(row["level"], row["type"])].append(row)
    test_strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible["test"]:
        test_strata[(row["level"], row["type"])].append(row)
    expected_strata = {(level, type_name) for level in EXPECTED_LEVELS for type_name in EXPECTED_TYPES}
    if set(train_strata) != expected_strata or set(test_strata) != expected_strata:
        raise ValueError("eligible data does not cover all 35 level/type strata")

    train_counts = {key: len(rows) for key, rows in train_strata.items()}
    train_quotas = proportional_quotas(train_counts, args.train_rows)
    remaining_counts = {key: train_counts[key] - train_quotas[key] for key in train_counts}
    dev_quotas = proportional_quotas(remaining_counts, args.dev_rows)
    sealed_quotas = proportional_quotas(
        {key: len(rows) for key, rows in test_strata.items()}, args.sealed_test_rows
    )

    rows_by_split: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "development": [],
        "sealed_test": [],
    }
    for key, rows in train_strata.items():
        ordered = sorted(
            rows,
            key=lambda row: selection_key(
                row["messages"][1]["content"], args.seed, "official_train_order"
            ),
        )
        train_end = train_quotas[key]
        dev_end = train_end + dev_quotas[key]
        rows_by_split["train"].extend(ordered[:train_end])
        rows_by_split["development"].extend(ordered[train_end:dev_end])
    for key, rows in test_strata.items():
        ordered = sorted(
            rows,
            key=lambda row: selection_key(
                row["messages"][1]["content"], args.seed, "official_test_order"
            ),
        )
        rows_by_split["sealed_test"].extend(ordered[: sealed_quotas[key]])

    for split, rows in rows_by_split.items():
        rows.sort(key=lambda row: selection_key(row["messages"][1]["content"], args.seed, split))
        for row in rows:
            row["split"] = split
    all_ids = [row["sample_id"] for rows in rows_by_split.values() for row in rows]
    all_questions = [row["question_sha256"] for rows in rows_by_split.values() for row in rows]
    if len(all_ids) != len(set(all_ids)) or len(all_questions) != len(set(all_questions)):
        raise ValueError("derived MATH splits overlap")

    output_files = {
        split: output_dir / f"{split}.jsonl" for split in rows_by_split
    }
    for split, path in output_files.items():
        write_jsonl(path, rows_by_split[split])

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "dataset_id": "public_sft_grpo_v1_math_sft1024",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "model_name_or_path": str(args.model_name_or_path.resolve()),
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "template": "trl_qwen3_5_nothink_training_chat_template",
        "template_sha256": sha256_text(qwen3_5_nothink_training_chat_template),
        "tokenizer": {
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
        "max_length": args.max_length,
        "environment": {
            "path": "/raid/zkq/artifacts/CAPA/runtime/venv-qwen35-math-cu124-v1",
            "math_verify": importlib.metadata.version("math-verify"),
            "latex2sympy2_extended": importlib.metadata.version("latex2sympy2-extended"),
            "antlr4_python3_runtime": importlib.metadata.version("antlr4-python3-runtime"),
        },
        "sources": {
            "math_lighteval_revision": MATH_REVISION,
            "source_file_sha256": {str(path): digest for path, digest in SOURCE_HASHES.items()},
            "source_rows": {split: len(rows) for split, rows in source_rows.items()},
            "normalized_train_unique": len(set(normalized_by_split["train"])),
            "normalized_test_unique": len(set(normalized_by_split["test"])),
            "normalized_upstream_train_test_overlap": len(cross_overlap),
        },
        "filters": {
            split: dict(sorted(counts.items())) for split, counts in filter_counts.items()
        },
        "eligible_rows": {split: len(rows) for split, rows in eligible.items()},
        "selection": {
            "method": "deterministic_sha256_with_proportional_level_type_stratification",
            "strata": 35,
            "train_quotas": {
                f"{key[0]}|{key[1]}": value for key, value in sorted(train_quotas.items())
            },
            "development_quotas": {
                f"{key[0]}|{key[1]}": value for key, value in sorted(dev_quotas.items())
            },
            "sealed_test_quotas": {
                f"{key[0]}|{key[1]}": value for key, value in sorted(sealed_quotas.items())
            },
        },
        "isolation": {
            "known_upstream_cross_overlap_excluded_from_both_sources": len(cross_overlap),
            "derived_question_overlap": 0,
            "development_source": "official train",
            "sealed_test_source": "official test",
            "sealed_test_allowed_for_checkpoint_selection": False,
        },
        "splits": {
            split: summarize_split(rows) for split, rows in rows_by_split.items()
        },
        "files": {},
    }
    for split, path in output_files.items():
        manifest["files"][split] = {"path": str(path), "sha256": sha256_file(path)}
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
