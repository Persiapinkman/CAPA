#!/usr/bin/env python3
"""Build hard-case SFT refresh data from the train split only."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "demo"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from training.planner_grpo_seed_v1.scripts.build_planner_sft_data import build_rows, write_json, write_jsonl  # noqa: E402
from training.planner_grpo_seed_v1.scripts.train_planner_grpo import load_jsonl  # noqa: E402


DEFAULT_TRAIN_CASES = ROOT / "training" / "planner_grpo_seed_v1" / "cases" / "planner_grpo_focused_train_v3_cases.jsonl"
DEFAULT_EVAL_FILE = ROOT / "training" / "planner_grpo_seed_v1" / "sft_data_v3_chatml" / "val.jsonl"
DEFAULT_BASE_TRAIN_FILE = ROOT / "training" / "planner_grpo_seed_v1" / "sft_data_v3_chatml" / "train.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "training" / "planner_grpo_seed_v1" / "sft_data_v4_hard_chatml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hard-case Planner SFT refresh data.")
    parser.add_argument("--train-cases", type=Path, default=DEFAULT_TRAIN_CASES)
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--base-train-file", type=Path, default=DEFAULT_BASE_TRAIN_FILE)
    parser.add_argument("--base-repeat", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--indent", type=int, default=-1)
    parser.add_argument("--prompt-format", choices=["qwen_chatml"], default="qwen_chatml")
    parser.add_argument("--append-im-end", action="store_true", default=True)
    parser.add_argument("--clarify-repeat", type=int, default=12)
    parser.add_argument("--pipeline-repeat", type=int, default=8)
    parser.add_argument("--migration-step2-repeat", type=int, default=2)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repeat_row(row: dict[str, Any], repeat: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in range(repeat):
        copied = dict(row)
        copied["hard_repeat_index"] = idx
        rows.append(copied)
    return rows


def tagged_rows(rows: list[dict[str, Any]], *, tag: str, repeat: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        for idx in range(repeat):
            copied = dict(row)
            copied["row_origin"] = tag
            copied["base_repeat_index"] = idx
            output.append(copied)
    return output


def main() -> None:
    args = parse_args()
    train_cases_path = resolve(args.train_cases)
    eval_file = resolve(args.eval_file)
    base_train_file = resolve(args.base_train_file)
    output_dir = resolve(args.output_dir)
    base_rows = load_jsonl(base_train_file) if args.base_repeat > 0 else []
    rows = build_rows(
        load_jsonl(train_cases_path),
        indent=args.indent,
        prompt_format=args.prompt_format,
        append_im_end=args.append_im_end,
    )
    hard_rows: list[dict[str, Any]] = []
    for row in rows:
        category = str(row.get("category") or "")
        step_index = int(row.get("step_index") or 0)
        if category == "clarify_intent_ambiguity":
            hard_rows.extend(repeat_row(row, args.clarify_repeat))
        elif category == "full_detection_eval":
            hard_rows.extend(repeat_row(row, args.pipeline_repeat))
        elif category in {"probe_then_migration", "probe_then_migration_strict"} and step_index == 2:
            hard_rows.extend(repeat_row(row, args.migration_step2_repeat))

    train_rows = tagged_rows(base_rows, tag="base_sft_v3_train", repeat=args.base_repeat) + hard_rows
    if not train_rows:
        raise RuntimeError("hard SFT data is empty")

    eval_rows = load_jsonl(eval_file)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "eval.jsonl", eval_rows)
    metadata = {
        "source_train_cases": str(train_cases_path),
        "source_base_train_file": str(base_train_file),
        "source_eval_file": str(eval_file),
        "leakage_guard": "hard train rows are built only from the train-case file; base train rows come from SFTv3 train; eval rows are copied from held-out SFTv3 val.",
        "format": {
            "prompt_format": args.prompt_format,
            "append_im_end": args.append_im_end,
            "json_indent": args.indent,
        },
        "repeat": {
            "base_sft_v3_train": args.base_repeat,
            "clarify_intent_ambiguity": args.clarify_repeat,
            "full_detection_eval": args.pipeline_repeat,
            "migration_step2": args.migration_step2_repeat,
        },
        "train": {
            "rows": len(train_rows),
            "base_rows": len(base_rows) * max(args.base_repeat, 0),
            "hard_rows": len(hard_rows),
            "cases": len({str(row.get("case_id") or "") for row in train_rows}),
            "categories": dict(sorted(Counter(str(row.get("category") or "") for row in train_rows).items())),
            "step_categories": dict(
                sorted(Counter(f"{row.get('category')}#step{row.get('step_index')}" for row in train_rows).items())
            ),
        },
        "eval": {
            "rows": len(eval_rows),
            "cases": len({str(row.get("case_id") or "") for row in eval_rows}),
            "categories": dict(sorted(Counter(str(row.get("category") or "") for row in eval_rows).items())),
        },
    }
    write_json(output_dir / "metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
