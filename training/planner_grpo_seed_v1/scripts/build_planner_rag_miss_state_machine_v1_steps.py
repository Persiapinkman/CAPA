#!/usr/bin/env python3
"""Build the preregistered RAG-miss state-machine GRPO training view."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = (
    ROOT / "training/planner_grpo_seed_v1/sft_data_stateful_retrieval_v1_chatml/train.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT / "training/planner_grpo_seed_v1/sft_data_rag_miss_state_machine_v1_chatml/train.jsonl"
)

SELECTION_WEIGHTS = {
    ("rag_miss_rewrite_then_rag", 1): 2,
    ("rag_miss_rewrite_then_rag", 2): 2,
    ("rag_miss_rewrite_then_rag", 3): 2,
    ("coref_rewrite_then_rag", 1): 1,
    ("direct_rag_guardrail", 1): 1,
    ("memory_hit_end", 1): 1,
    ("general_answer_guardrail", 1): 1,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_rows(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in source_rows:
        key = (str(row["category"]), int(row["step_index"]))
        for replica in range(SELECTION_WEIGHTS.get(key, 0)):
            item = dict(row)
            item["source_case_id"] = str(row["case_id"])
            item["sampling_replica"] = replica + 1
            item["training_row_id"] = f"{row['case_id']}#step{row['step_index']}#rep{replica + 1}"
            output.append(item)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source if args.source.is_absolute() else ROOT / args.source
    output = args.output if args.output.is_absolute() else ROOT / args.output
    rows = build_rows(load_jsonl(source))
    if len(rows) != 240:
        raise ValueError(f"expected 240 weighted rows, got {len(rows)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    metadata = {
        "schema_version": "1.0",
        "dataset_id": "planner_rag_miss_state_machine_v1",
        "source": str(source),
        "source_sha256": sha256(source),
        "output": str(output),
        "output_sha256": sha256(output),
        "rows": len(rows),
        "unique_source_rows": len({row["training_row_id"].rsplit("#rep", 1)[0] for row in rows}),
        "entities": len({str(row["entity_id"]) for row in rows}),
        "selection_weights": {
            f"{category}#step{step}": weight
            for (category, step), weight in SELECTION_WEIGHTS.items()
        },
        "weighted_category_steps": {
            f"{category}#step{step}": count
            for (category, step), count in sorted(
                Counter((str(row["category"]), int(row["step_index"])) for row in rows).items()
            )
        },
    }
    (output.parent / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
