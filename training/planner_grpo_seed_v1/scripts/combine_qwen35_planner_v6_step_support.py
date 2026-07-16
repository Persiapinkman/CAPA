#!/usr/bin/env python3
"""Combine disjoint Planner V6 local-support sample shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.planner_grpo_seed_v1.scripts.eval_qwen35_planner_v6_step_support import (
    load_jsonl,
    summarize_samples,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, nargs="+", required=True)
    parser.add_argument("--samples-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = [row for path in args.samples for row in load_jsonl(path)]
    keys = [(str(row["case_id"]), int(row["sample_index"])) for row in samples]
    if len(keys) != len(set(keys)):
        raise ValueError("combined sample shards overlap")
    samples.sort(key=lambda row: (int(row["global_row_index"]), int(row["sample_index"])))
    args.samples_out.parent.mkdir(parents=True, exist_ok=True)
    args.samples_out.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in samples),
        encoding="utf-8",
    )
    summary: dict[str, Any] = {
        "input_shards": [str(path) for path in args.samples],
        **summarize_samples(samples),
    }
    write_json(args.summary_out, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
