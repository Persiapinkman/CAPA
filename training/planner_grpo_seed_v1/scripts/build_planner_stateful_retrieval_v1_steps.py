#!/usr/bin/env python3
"""Expand predefined stateful retrieval case splits into ChatML step rows."""

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
    ROOT / "training" / "planner_grpo_seed_v1" / "sft_data_stateful_retrieval_v1_chatml"
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.planner_grpo_seed_v1.scripts.build_planner_sft_data import (  # noqa: E402
    build_rows,
)
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
        "max_steps_per_case": max(
            Counter(str(row["case_id"]) for row in rows).values(), default=0
        ),
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
        "dataset_id": "planner_stateful_retrieval_v1",
        "prompt_format": "qwen_chatml",
        "append_im_end": True,
        "splits": {},
    }
    seen_case_ids: dict[str, set[str]] = {}
    for split in ("train", "dev", "test"):
        case_path = case_dir / f"planner_stateful_retrieval_v1_{split}_cases.jsonl"
        rows = build_rows(
            load_jsonl(case_path),
            indent=-1,
            prompt_format="qwen_chatml",
            append_im_end=True,
        )
        write_jsonl(output_dir / f"{split}.jsonl", rows)
        metadata["splits"][split] = {
            **stats(rows),
            "source_cases": str(case_path),
            "output": str(output_dir / f"{split}.jsonl"),
        }
        seen_case_ids[split] = {str(row["case_id"]) for row in rows}

    overlaps = {}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlaps[f"{left}_{right}"] = len(seen_case_ids[left] & seen_case_ids[right])
    metadata["case_id_overlaps"] = overlaps
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
