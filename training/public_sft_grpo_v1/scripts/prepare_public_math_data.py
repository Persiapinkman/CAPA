#!/usr/bin/env python3
"""Freeze deterministic GSM8K smoke splits and audit Qwen3.5 SFT labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import AutoTokenizer
from trl.chat_template_utils import qwen3_5_nothink_training_chat_template


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.public_sft_grpo_v1.scripts.public_math_contract import extract_gsm8k_gold


MODEL_PATH = Path("/raid/zkq/models/Qwen3.5-4B")
ARTIFACT_ROOT = Path("/raid/zkq/artifacts/CAPA/datasets/public_sft_grpo_v1/raw/hf")
GSM_REVISION = "740312add88f781978c0658806c59bc2815b9866"
MATH_REVISION = "0530c78699ea5e8eb5530600900e1f328b48acad"
GSM_DIR = ARTIFACT_ROOT / "openai__gsm8k" / GSM_REVISION
MATH_DIR = ARTIFACT_ROOT / "DigitalLearningGmbH__MATH-lighteval" / MATH_REVISION
DEFAULT_OUTPUT = ROOT / "training/public_sft_grpo_v1/data/gsm8k_sft32_v1"
SYSTEM_PROMPT = (
    "You are a careful math assistant. Solve the problem step by step and end your "
    "response with exactly one final line in the form: #### <integer>"
)
SOURCE_HASHES = {
    GSM_DIR / "README.md": "a17e882503578c9e324560630e31017617714ee025c87cf4fea6fd916895f3c1",
    GSM_DIR / "main/train-00000-of-00001.parquet": "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d",
    GSM_DIR / "main/test-00000-of-00001.parquet": "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59",
    MATH_DIR / "README.md": "3bb056303034d0f49925396ede385ee2feed638c26e2abbf3a8ff0b9976a9bc4",
    MATH_DIR / "data/train-00000-of-00001.parquet": "eca6e667f4305dd5e5ba09b4fd55e7f3174a0fbe361cdfd4c44758b593a76933",
    MATH_DIR / "data/test-00000-of-00001.parquet": "7dca8d6e41af88ecf82f2b5f36eb5530e083aaaa86ee325f62bd5c31535178c6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-name-or-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-rows", type=int, default=32)
    parser.add_argument("--dev-rows", type=int, default=32)
    parser.add_argument("--sealed-test-rows", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=1024)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_question(text: str) -> str:
    return " ".join(str(text).split())


def selection_key(question: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\n{normalized_question(question)}".encode("utf-8")).hexdigest()


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


def make_row(source: dict[str, Any], source_split: str, source_index: int, split: str) -> dict[str, Any]:
    question = str(source["question"])
    solution = str(source["answer"])
    question_sha = hashlib.sha256(normalized_question(question).encode("utf-8")).hexdigest()
    gold = extract_gsm8k_gold(solution)
    return {
        "sample_id": f"gsm8k-{source_split}-{source_index:05d}-{question_sha[:12]}",
        "dataset_id": "public_sft_grpo_v1",
        "source_dataset": "openai/gsm8k",
        "source_revision": GSM_REVISION,
        "source_config": "main",
        "source_split": source_split,
        "source_index": source_index,
        "split": split,
        "question_sha256": question_sha,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": solution},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
        "verifier_type": "gsm8k_integer_v1",
        "ground_truth": gold,
        "gold_solution": solution,
        "reward_metadata": {
            "accuracy_weight": 0.95,
            "format_weight": 0.05,
            "format_contract": "exactly one final line: #### <integer>",
        },
    }


def label_audit(row: dict[str, Any], tokenizer: Any, max_length: int) -> dict[str, Any]:
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
    if len(ids) != len(mask) or not any(mask):
        raise RuntimeError(f"invalid assistant mask for {row['sample_id']}")
    assistant_spans = sum(
        bool(enabled) and (index == 0 or not mask[index - 1])
        for index, enabled in enumerate(mask)
    )
    if assistant_spans != 1:
        raise RuntimeError(
            f"expected one assistant span for {row['sample_id']}, got {assistant_spans}"
        )
    supervised_ids = [token for token, enabled in zip(ids, mask, strict=True) if enabled]
    supervised_text = tokenizer.decode(supervised_ids, skip_special_tokens=False)
    if row["gold_solution"] not in supervised_text or "<|im_end|>" not in supervised_text:
        raise RuntimeError(f"assistant span/EOS mismatch for {row['sample_id']}")
    if len(ids) > max_length:
        raise ValueError(f"{row['sample_id']} has {len(ids)} tokens > max_length={max_length}")
    prompt_messages = row["messages"][:-1]
    prompt_tokenized = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        return_dict=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return {
        "total_tokens": len(ids),
        "assistant_tokens": sum(mask),
        "assistant_fraction": sum(mask) / len(mask),
        "prompt_tokens": len(prompt_tokenized["input_ids"]),
        "eos_supervised": True,
        "assistant_spans": 1,
    }


def main() -> None:
    args = parse_args()
    audit_sources()
    train_path = GSM_DIR / "main/train-00000-of-00001.parquet"
    test_path = GSM_DIR / "main/test-00000-of-00001.parquet"
    upstream_train = Dataset.from_parquet(str(train_path))
    upstream_test = Dataset.from_parquet(str(test_path))
    if len(upstream_train) != 7473 or len(upstream_test) != 1319:
        raise ValueError(f"unexpected GSM8K sizes: {len(upstream_train)}/{len(upstream_test)}")

    train_questions = [normalized_question(value) for value in upstream_train["question"]]
    test_questions = [normalized_question(value) for value in upstream_test["question"]]
    if len(set(train_questions)) != len(train_questions) or len(set(test_questions)) != len(test_questions):
        raise ValueError("duplicate normalized question inside an upstream split")
    overlap = set(train_questions) & set(test_questions)
    if overlap:
        raise ValueError(f"upstream train/test question overlap: {len(overlap)}")

    ordered_train = sorted(
        range(len(upstream_train)),
        key=lambda index: selection_key(upstream_train[index]["question"], args.seed),
    )
    ordered_test = sorted(
        range(len(upstream_test)),
        key=lambda index: selection_key(upstream_test[index]["question"], args.seed),
    )
    required = args.train_rows + args.dev_rows
    if required > len(ordered_train) or args.sealed_test_rows > len(ordered_test):
        raise ValueError("requested split is larger than upstream data")

    selections = {
        "train": [("train", index) for index in ordered_train[: args.train_rows]],
        "development": [
            ("train", index) for index in ordered_train[args.train_rows : required]
        ],
        "sealed_test": [("test", index) for index in ordered_test[: args.sealed_test_rows]],
    }
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split, selected in selections.items():
        rows = []
        for source_split, source_index in selected:
            source_dataset = upstream_train if source_split == "train" else upstream_test
            rows.append(make_row(source_dataset[source_index], source_split, source_index, split))
        rows_by_split[split] = rows

    ids = [row["sample_id"] for rows in rows_by_split.values() for row in rows]
    questions = [row["question_sha256"] for rows in rows_by_split.values() for row in rows]
    if len(ids) != len(set(ids)) or len(questions) != len(set(questions)):
        raise ValueError("derived splits overlap")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path.resolve(), trust_remote_code=False, use_fast=True
    )
    if tokenizer.eos_token_id != 248046 or tokenizer.pad_token_id != 248044:
        raise RuntimeError(
            f"Qwen3.5 tokenizer contract changed: eos={tokenizer.eos_token_id}, pad={tokenizer.pad_token_id}"
        )
    label_audits: dict[str, list[dict[str, Any]]] = {}
    for split, rows in rows_by_split.items():
        label_audits[split] = [label_audit(row, tokenizer, args.max_length) for row in rows]

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_files = {
        "train": output_dir / "train.jsonl",
        "development": output_dir / "development.jsonl",
        "sealed_test": output_dir / "sealed_test.jsonl",
    }
    for split, path in output_files.items():
        write_jsonl(path, rows_by_split[split])

    split_stats: dict[str, Any] = {}
    for split, audits in label_audits.items():
        totals = [row["total_tokens"] for row in audits]
        assistants = [row["assistant_tokens"] for row in audits]
        prompts = [row["prompt_tokens"] for row in audits]
        split_stats[split] = {
            "rows": len(audits),
            "total_tokens": {
                "min": min(totals),
                "p50": percentile(totals, 0.5),
                "p95": percentile(totals, 0.95),
                "max": max(totals),
            },
            "assistant_tokens": {"min": min(assistants), "p50": percentile(assistants, 0.5), "max": max(assistants)},
            "prompt_tokens": {"min": min(prompts), "p50": percentile(prompts, 0.5), "max": max(prompts)},
            "eos_supervised_rate": 1.0,
            "assistant_mask_nonempty_rate": 1.0,
        }

    manifest = {
        "schema_version": "1.0",
        "dataset_id": "public_sft_grpo_v1_gsm8k_sft32",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "model_name_or_path": str(args.model_name_or_path.resolve()),
        "template": "trl_qwen3_5_nothink_training_chat_template",
        "template_sha256": hashlib.sha256(qwen3_5_nothink_training_chat_template.encode("utf-8")).hexdigest(),
        "tokenizer": {"eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id},
        "max_length": args.max_length,
        "sources": {
            "gsm8k_revision": GSM_REVISION,
            "math_lighteval_revision": MATH_REVISION,
            "source_file_sha256": {str(path): digest for path, digest in SOURCE_HASHES.items()},
        },
        "isolation": {
            "normalized_upstream_train_test_overlap": 0,
            "derived_question_overlap": 0,
            "development_source": "official train",
            "sealed_test_source": "official test",
            "sealed_test_allowed_for_checkpoint_selection": False,
        },
        "splits": split_stats,
        "files": {},
    }
    for split, path in output_files.items():
        manifest["files"][split] = {"path": str(path), "sha256": sha256_file(path)}
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
