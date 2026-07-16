#!/usr/bin/env python3
"""Freeze an action-balanced 80-case support subset from Qwen3.5 V5 step data."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = (
    ROOT
    / "training/planner_grpo_seed_v1/step_data/"
    "planner_multistep_grpo_value_v5_train_v1_qwen35_4b_nothinking_step2.jsonl"
)
OUTPUT = (
    ROOT
    / "training/planner_grpo_seed_v1/step_data/"
    "planner_multistep_grpo_value_v5_train_v1_qwen35_4b_support80.jsonl"
)
QUOTAS = {"migrate": 6, "retry": 4}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_key(category: str, row: dict) -> str:
    value = f"20260715|{category}|{row['case_id']}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_category[str(row["category"])].append(row)
    if len(by_category) != 8:
        raise ValueError(f"expected 8 categories, found {len(by_category)}")

    entity_usage: Counter[str] = Counter()
    selected: list[dict] = []
    for category in sorted(by_category):
        category_selected: list[dict] = []
        for action_class, quota in QUOTAS.items():
            candidates = [
                row for row in by_category[category] if str(row.get("target_action_class")) == action_class
            ]
            candidates.sort(
                key=lambda row: (
                    entity_usage[str(row["entity_id"])],
                    stable_key(category, row),
                )
            )
            chosen = candidates[:quota]
            if len(chosen) != quota:
                raise ValueError(f"{category}/{action_class}: found {len(chosen)}, expected {quota}")
            category_selected.extend(chosen)
            for row in chosen:
                entity_usage[str(row["entity_id"])] += 1
        category_selected.sort(key=lambda row: str(row["case_id"]))
        selected.extend(category_selected)

    if len(selected) != 80 or len({row["case_id"] for row in selected}) != 80:
        raise ValueError("support subset must contain 80 unique cases")
    category_counts = Counter(str(row["category"]) for row in selected)
    action_counts = Counter(str(row["target_action_class"]) for row in selected)
    if set(category_counts.values()) != {10} or action_counts != Counter({"migrate": 48, "retry": 32}):
        raise ValueError(f"support distribution mismatch: {category_counts}, {action_counts}")
    if max(entity_usage.values()) > 2:
        raise ValueError(f"entity reuse exceeds two categories: {entity_usage.most_common(5)}")

    OUTPUT.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    metadata = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "optimization_pool_stochastic_support_only",
        "source": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "output": str(OUTPUT),
        "output_sha256": sha256(OUTPUT),
        "rows": len(selected),
        "samples_per_prompt": 8,
        "categories": dict(sorted(category_counts.items())),
        "target_action_classes": dict(sorted(action_counts.items())),
        "unique_entities": len(entity_usage),
        "maximum_rows_per_entity": max(entity_usage.values()),
        "entity_exposure_histogram": dict(sorted(Counter(entity_usage.values()).items())),
        "selection": {
            "seed_material": "20260715",
            "per_category_quotas": QUOTAS,
            "priority": "minimize prior entity use, then stable SHA256 order",
        },
    }
    metadata_path = OUTPUT.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
