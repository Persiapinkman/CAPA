#!/usr/bin/env python3
"""Build grouped Planner SFT prompt/completion data from GRPO cases."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "demo"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from training.planner_grpo_seed_v1.scripts.train_planner_grpo import (  # noqa: E402
    DEFAULT_CASES,
    build_prompt_for_step,
    expected_decision_to_planner_step,
    load_jsonl,
)


DEFAULT_OUTPUT_DIR = ROOT / "training" / "planner_grpo_seed_v1" / "sft_data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CAPA Planner SFT warmup data.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--indent", type=int, default=2)
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def canonical_completion(expected_step: dict[str, Any], *, indent: int) -> str:
    decision = expected_decision_to_planner_step(expected_step)
    if indent < 0:
        return json.dumps(decision, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(decision, ensure_ascii=False, indent=indent)


def build_rows(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    run_root = ROOT / "training" / "planner_grpo_seed_v1" / "reports" / "sft_prompt_contexts"
    rows: list[dict[str, Any]] = []
    for case in cases:
        expected = case.get("expected_decisions") if isinstance(case.get("expected_decisions"), list) else []
        case_id = str(case.get("case_id") or "")
        category = str(case.get("category") or "")
        for step_index, expected_step in enumerate(expected, start=1):
            if not isinstance(expected_step, dict):
                continue
            previous_action = ""
            if step_index > 1 and isinstance(expected[step_index - 2], dict):
                previous_action = str(expected[step_index - 2].get("action") or "").strip()
            rows.append(
                {
                    "prompt": build_prompt_for_step(case, step_index, run_root),
                    "completion": canonical_completion(expected_step, indent=2),
                    "case_id": case_id,
                    "category": category,
                    "step_index": step_index,
                    "expected_step": json.dumps(expected_step, ensure_ascii=False),
                    "forbidden_actions": json.dumps(case.get("forbidden_actions") or [], ensure_ascii=False),
                    "reward_spec": json.dumps(case.get("reward_spec") or {}, ensure_ascii=False),
                    "previous_action": previous_action,
                    "full_expected_actions": json.dumps(
                        [
                            (
                                "clarify"
                                if str(step.get("decision_type") or "tool") == "clarify"
                                else str(step.get("action") or "")
                            )
                            for step in expected
                            if isinstance(step, dict)
                        ],
                        ensure_ascii=False,
                    ),
                }
            )
    return rows


def grouped_split(rows: list[dict[str, Any]], *, val_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    by_category: dict[str, list[str]] = defaultdict(list)
    case_category: dict[str, str] = {}
    for row in rows:
        case_id = str(row["case_id"])
        category = str(row["category"])
        if case_id not in case_category:
            case_category[case_id] = category
            by_category[category].append(case_id)

    rng = random.Random(seed)
    train_ids: list[str] = []
    val_ids: list[str] = []
    for category, case_ids in sorted(by_category.items()):
        shuffled = list(case_ids)
        rng.shuffle(shuffled)
        if len(shuffled) <= 1:
            train_ids.extend(shuffled)
            continue
        val_count = max(1, round(len(shuffled) * val_ratio))
        val_ids.extend(shuffled[:val_count])
        train_ids.extend(shuffled[val_count:])
    return sorted(train_ids), sorted(val_ids)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    case_ids = {str(row["case_id"]) for row in rows}
    return {
        "cases": len(case_ids),
        "steps": len(rows),
        "categories": dict(sorted(Counter(str(row["category"]) for row in rows).items())),
    }


def main() -> None:
    args = parse_args()
    cases_path = args.cases if args.cases.is_absolute() else ROOT / args.cases
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    rows = build_rows(load_jsonl(cases_path))
    train_ids, val_ids = grouped_split(rows, val_ratio=args.val_ratio, seed=args.seed)
    train_set = set(train_ids)
    val_set = set(val_ids)
    train_rows = [row for row in rows if row["case_id"] in train_set]
    val_rows = [row for row in rows if row["case_id"] in val_set]
    overlap = train_set & val_set
    if overlap:
        raise RuntimeError(f"grouped split leaked case ids: {sorted(overlap)[:5]}")

    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "val.jsonl", val_rows)
    metadata = {
        "source_cases": str(cases_path),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "train": summarize(train_rows),
        "val": summarize(val_rows),
        "total": summarize(rows),
        "train_case_ids": train_ids,
        "val_case_ids": val_ids,
        "format": {
            "prompt": "Planner prompt ending at assistant turn.",
            "completion": "One canonical planner JSON decision. TRL SFT adds EOS if missing.",
            "loss": "completion_only_loss=True in SFT training.",
        },
    }
    write_json(output_dir / "metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
