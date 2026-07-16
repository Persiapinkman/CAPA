#!/usr/bin/env python3
"""Combine and validate independently generated Qwen3.5 MATH evaluation shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.public_sft_grpo_v1.scripts.eval_qwen35_math import (  # noqa: E402
    build_metrics,
    load_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def canonical_json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite combined output directory: {output_dir}")
    input_dirs = [path if path.is_absolute() else ROOT / path for path in args.input_dir]
    results: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    input_artifacts: list[dict[str, str]] = []
    for input_dir in input_dirs:
        result_path = input_dir / "result.json"
        sample_path = input_dir / "samples.jsonl"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed" or sha256_file(sample_path) != result["samples_sha256"]:
            raise ValueError(f"invalid or incomplete shard: {input_dir}")
        shard_samples = load_jsonl(sample_path)
        if len(shard_samples) != result["sharding"]["shard_rows"]:
            raise ValueError(f"shard row count mismatch: {input_dir}")
        results.append(result)
        samples.extend(shard_samples)
        input_artifacts.append(
            {
                "input_dir": str(input_dir),
                "result_sha256": sha256_file(result_path),
                "samples_sha256": sha256_file(sample_path),
            }
        )

    first = results[0]
    num_shards = int(first["sharding"]["num_shards"])
    if len(results) != num_shards:
        raise ValueError(f"expected {num_shards} shard directories, got {len(results)}")
    if {int(result["sharding"]["shard_index"]) for result in results} != set(range(num_shards)):
        raise ValueError("shard indices are incomplete or duplicated")
    invariant_fields = [
        "model_name_or_path",
        "adapter_path",
        "data_dir",
        "dataset_manifest_sha256",
        "generation",
    ]
    for result in results[1:]:
        for field in invariant_fields:
            if result[field] != first[field]:
                raise ValueError(f"shard invariant mismatch for {field}")
        if (
            result["sharding"]["num_shards"] != num_shards
            or result["sharding"]["global_selected_ids_sha256"]
            != first["sharding"]["global_selected_ids_sha256"]
        ):
            raise ValueError("shard topology or selected population mismatch")
    sample_ids = [row["sample_id"] for row in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("combined shards contain duplicate sample ids")
    expected_rows = int(first["sharding"]["global_selected_rows"])
    if len(samples) != expected_rows:
        raise ValueError(f"combined rows {len(samples)} != expected {expected_rows}")
    split_order = {"train": 0, "development": 1, "sealed_test": 2}
    samples.sort(key=lambda row: (split_order[row["split"]], row["sample_id"]))
    splits = sorted({row["split"] for row in samples}, key=lambda split: split_order[split])
    metrics = build_metrics(samples, splits)
    combined = {
        "schema_version": "1.0",
        "status": "completed",
        "model_name_or_path": first["model_name_or_path"],
        "adapter_path": first["adapter_path"],
        "data_dir": first["data_dir"],
        "dataset_manifest_sha256": first["dataset_manifest_sha256"],
        "generation": first["generation"],
        "sharding": {
            "num_shards": num_shards,
            "combined_rows": len(samples),
            "global_selected_ids_sha256": first["sharding"]["global_selected_ids_sha256"],
            "shard_indices": sorted(result["sharding"]["shard_index"] for result in results),
            "wall_runtime_seconds_proxy": max(float(result["runtime_seconds"]) for result in results),
            "sum_shard_runtime_seconds": sum(float(result["runtime_seconds"]) for result in results),
        },
        "metrics": metrics,
        "nonfinite_metric_count": sum(int(result["nonfinite_metric_count"]) for result in results),
        "inputs": sorted(input_artifacts, key=lambda item: item["input_dir"]),
        "input_contract_sha256": canonical_json_hash(input_artifacts),
    }
    write_jsonl(output_dir / "samples.jsonl", samples)
    combined["samples_sha256"] = sha256_file(output_dir / "samples.jsonl")
    write_json(output_dir / "result.json", combined)
    print(json.dumps(combined, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
