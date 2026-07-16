#!/usr/bin/env python3
"""Apply the frozen Qwen3.5 V5 migrate-target and retry-anchor support gate."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stats(groups: list[dict[str, Any]]) -> dict[str, float | int]:
    if not groups:
        raise ValueError("support gate scope is empty")
    return {
        "groups": len(groups),
        "nonzero_std_rate": statistics.mean(float(float(row["reward_std"]) > 1e-6) for row in groups),
        "usable_support_rate": statistics.mean(
            float(float(row["reward_std"]) > 1e-6 and float(row["max_reward"]) >= 0.75)
            for row in groups
        ),
        "exact_action_support_rate": statistics.mean(
            float(bool(row["exact_action_support"])) for row in groups
        ),
        "mean_distinct_valid_actions": statistics.mean(
            float(row["distinct_valid_actions"]) for row in groups
        ),
        "fully_saturated_rate": statistics.mean(
            float(float(row["min_reward"]) >= 0.95) for row in groups
        ),
        "near_exact_task_support_rate": statistics.mean(
            float(float(row["max_task_reward"]) >= 0.95) for row in groups
        ),
    }


def check(name: str, observed: float, operator: str, threshold: float) -> dict[str, Any]:
    if operator == ">=":
        passed = observed >= threshold
    elif operator == "<=":
        passed = observed <= threshold
    else:
        raise ValueError(operator)
    return {
        "name": name,
        "observed": observed,
        "operator": operator,
        "threshold": threshold,
        "passed": passed,
    }


def main() -> None:
    args = parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    gate = spec["support_gate"]
    summary = json.loads((args.audit_dir / "summary.json").read_text(encoding="utf-8"))
    groups = load_jsonl(args.audit_dir / "groups.jsonl")
    samples = load_jsonl(args.audit_dir / "samples.jsonl")
    data = {str(row["case_id"]): row for row in load_jsonl(args.data)}
    if len(groups) != 80 or len(samples) != 640 or len(data) != 80:
        raise ValueError(
            f"expected 80 groups/640 samples/80 data rows, got {len(groups)}/{len(samples)}/{len(data)}"
        )
    for row in groups:
        case_id = str(row["case_id"])
        if case_id not in data:
            raise KeyError(case_id)
        row["target_action_class"] = str(data[case_id]["target_action_class"])

    primary_classes = set(gate["primary_target_action_classes"])
    anchor_classes = set(gate["anchor_target_action_classes"])
    primary_groups = [row for row in groups if row["target_action_class"] in primary_classes]
    anchor_groups = [row for row in groups if row["target_action_class"] in anchor_classes]
    primary = stats(primary_groups)
    anchor = stats(anchor_groups)

    model_path = str(summary["model"])
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False, use_fast=True)
    eos_id = tokenizer.eos_token_id
    clipped = 0
    eos_lengths: list[int] = []
    for sample in samples:
        ids = tokenizer(str(sample.get("completion") or ""), add_special_tokens=False)["input_ids"]
        if eos_id in ids:
            eos_lengths.append(ids.index(eos_id) + 1)
        else:
            clipped += 1
            eos_lengths.append(len(ids))
    clipped_rate = clipped / len(samples)

    checks = [
        check(
            "primary_nonzero_reward_std_rate",
            float(primary["nonzero_std_rate"]),
            ">=",
            float(gate["minimum_nonzero_reward_std_rate"]),
        ),
        check(
            "primary_usable_support_rate",
            float(primary["usable_support_rate"]),
            ">=",
            float(gate["minimum_usable_support_rate"]),
        ),
        check(
            "primary_exact_action_support_rate",
            float(primary["exact_action_support_rate"]),
            ">=",
            float(gate["minimum_exact_action_support_rate"]),
        ),
        check(
            "primary_mean_distinct_valid_actions",
            float(primary["mean_distinct_valid_actions"]),
            ">=",
            float(gate["minimum_mean_distinct_valid_actions"]),
        ),
        check(
            "primary_fully_saturated_rate",
            float(primary["fully_saturated_rate"]),
            "<=",
            float(gate["maximum_fully_saturated_rate"]),
        ),
        check(
            "anchor_exact_action_support_rate",
            float(anchor["exact_action_support_rate"]),
            ">=",
            float(gate["minimum_anchor_exact_action_support_rate"]),
        ),
        check(
            "completion_clipped_rate",
            clipped_rate,
            "<=",
            float(gate["maximum_completion_clipped_rate"]),
        ),
    ]
    payload = {
        "schema_version": "1.0",
        "study_id": spec["study_id"],
        "model": model_path,
        "audit": str(args.audit_dir),
        "data": str(args.data),
        "primary_target_action_classes": sorted(primary_classes),
        "anchor_target_action_classes": sorted(anchor_classes),
        "primary": primary,
        "anchor": anchor,
        "completion": {
            "samples": len(samples),
            "eos_token_id": eos_id,
            "natural_eos_length_min": min(eos_lengths),
            "natural_eos_length_p50": sorted(eos_lengths)[len(eos_lengths) // 2],
            "natural_eos_length_p95": sorted(eos_lengths)[int((len(eos_lengths) - 1) * 0.95)],
            "natural_eos_length_p99": sorted(eos_lengths)[int((len(eos_lengths) - 1) * 0.99)],
            "natural_eos_length_max": max(eos_lengths),
            "clipped_samples": clipped,
            "clipped_rate": clipped_rate,
        },
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
