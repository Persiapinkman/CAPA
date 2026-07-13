#!/usr/bin/env python3
"""Expand stateful retrieval v2 cases into fixed ChatML step artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = ROOT / "training" / "planner_grpo_seed_v1" / "cases"
DEFAULT_OUTPUT_DIR = (
    ROOT / "training" / "planner_grpo_seed_v1" / "sft_data_stateful_retrieval_v2_chatml"
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.planner_grpo_seed_v1.scripts.build_planner_sft_data import build_rows  # noqa: E402
from training.planner_grpo_seed_v1.scripts.train_planner_grpo import load_jsonl  # noqa: E402


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cases": len({str(row["case_id"]) for row in rows}),
        "steps": len(rows),
        "entities": len({str(row["entity_id"]) for row in rows}),
        "categories": dict(sorted(Counter(str(row["category"]) for row in rows).items())),
        "max_steps_per_case": max(Counter(str(row["case_id"]) for row in rows).values(), default=0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, default=CASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_dir = args.case_dir if args.case_dir.is_absolute() else ROOT / args.case_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    metadata: dict[str, Any] = {
        "dataset_id": "planner_stateful_retrieval_v2",
        "prompt_format": "qwen_chatml",
        "append_im_end": True,
        "splits": {},
    }
    seen_entities: dict[str, set[str]] = {}
    for split in ("train", "dev", "test"):
        case_path = case_dir / f"planner_stateful_retrieval_v2_{split}_cases.jsonl"
        rows = build_rows(
            load_jsonl(case_path), indent=-1, prompt_format="qwen_chatml", append_im_end=True
        )
        write_jsonl(output_dir / f"{split}.jsonl", rows)
        metadata["splits"][split] = {
            **stats(rows),
            "source_cases": str(case_path),
            "output": str(output_dir / f"{split}.jsonl"),
        }
        seen_entities[split] = {str(row["entity_id"]) for row in rows}

    support_entities = sorted(seen_entities["dev"])[:8]
    dev_rows = load_jsonl(output_dir / "dev.jsonl")
    support_rows = [
        row
        for row in dev_rows
        if row["entity_id"] in support_entities
        and row["category"] == "rag_double_miss_recovery"
    ]
    if len(support_rows) != 40:
        raise RuntimeError(f"expected 40 support-audit rows, got {len(support_rows)}")
    write_jsonl(output_dir / "support_audit.jsonl", support_rows)
    metadata["support_audit"] = {
        **stats(support_rows),
        "output": str(output_dir / "support_audit.jsonl"),
        "selection": "all five double-miss steps for the first eight sorted dev entity IDs",
    }
    metadata["entity_overlaps"] = {
        "train_dev": len(seen_entities["train"] & seen_entities["dev"]),
        "train_test": len(seen_entities["train"] & seen_entities["test"]),
        "dev_test": len(seen_entities["dev"] & seen_entities["test"]),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
