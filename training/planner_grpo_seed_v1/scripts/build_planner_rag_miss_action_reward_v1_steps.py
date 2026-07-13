#!/usr/bin/env python3
"""Build the action-dominant RAG-miss GRPO training view."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = (
    ROOT / "training/planner_grpo_seed_v1/sft_data_stateful_retrieval_v1_chatml/train.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT / "training/planner_grpo_seed_v1/sft_data_rag_miss_action_reward_v1_chatml/train.jsonl"
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.planner_grpo_seed_v1.scripts.build_planner_rag_miss_state_machine_v1_steps import (  # noqa: E402
    SELECTION_WEIGHTS,
    build_rows,
    load_jsonl,
)


ACTION_DOMINANT_REWARD = {
    "json_valid": 0.02,
    "decision_type_valid": 0.03,
    "action_match": 0.75,
    "argument_match": 0.10,
    "finish_after_tool": 0.05,
    "no_forbidden_action": 0.05,
    "wrong_action_cap": 0.20,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    encoded_reward = json.dumps(ACTION_DOMINANT_REWARD, ensure_ascii=False)
    for row in rows:
        row["reward_spec"] = encoded_reward
        row["reward_profile"] = "action_dominant_v1"
    if len(rows) != 240:
        raise ValueError(f"expected 240 weighted rows, got {len(rows)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    metadata = {
        "schema_version": "1.0",
        "dataset_id": "planner_rag_miss_action_reward_v1",
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
        "reward_profile": ACTION_DOMINANT_REWARD,
        "outer_reward_weights": {"task": 0.95, "format": 0.05},
        "maximum_total_reward_for_wrong_action": 0.24,
    }
    (output.parent / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
