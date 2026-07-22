#!/usr/bin/env python3
"""Verify that the benchmark copy changes only Qwen3.5 MTP metadata/weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    original_config = json.loads((args.original / "config.json").read_text())
    prepared_config = json.loads((args.prepared / "config.json").read_text())
    config_differences = {
        key: {"original": original_config.get(key), "prepared": prepared_config.get(key)}
        for key in sorted(set(original_config) | set(prepared_config))
        if original_config.get(key) != prepared_config.get(key)
    }

    original_index = json.loads((args.original / "model.safetensors.index.json").read_text())
    prepared_index = json.loads((args.prepared / "model.safetensors.index.json").read_text())
    original_map = original_index["weight_map"]
    prepared_map = prepared_index["weight_map"]
    expected_map = {key: shard for key, shard in original_map.items() if not key.startswith("mtp.")}
    removed_mtp_keys = sorted(key for key in original_map if key.startswith("mtp."))

    mismatched_tensors: list[str] = []
    checked = 0
    for shard_name in sorted(set(expected_map.values())):
        keys = [key for key, shard in expected_map.items() if shard == shard_name]
        with safe_open(args.original / shard_name, framework="pt", device="cpu") as original_shard:
            with safe_open(args.prepared / shard_name, framework="pt", device="cpu") as prepared_shard:
                for key in keys:
                    original_tensor = original_shard.get_tensor(key)
                    prepared_tensor = prepared_shard.get_tensor(key)
                    if (
                        original_tensor.shape != prepared_tensor.shape
                        or original_tensor.dtype != prepared_tensor.dtype
                        or not torch.equal(original_tensor, prepared_tensor)
                    ):
                        mismatched_tensors.append(key)
                    checked += 1

    prepared_shard_keys: list[str] = []
    for shard_path in sorted(args.prepared.glob("*.safetensors")):
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            prepared_shard_keys.extend(shard.keys())

    payload = {
        "schema_version": 1,
        "original": str(args.original.resolve()),
        "prepared": str(args.prepared.resolve()),
        "config_differences": config_differences,
        "config_change_is_architectures_only": set(config_differences) == {"architectures"},
        "original_index_key_count": len(original_map),
        "prepared_index_key_count": len(prepared_map),
        "removed_mtp_key_count": len(removed_mtp_keys),
        "removed_mtp_keys": removed_mtp_keys,
        "prepared_index_matches_original_without_mtp": prepared_map == expected_map,
        "prepared_shard_mtp_key_count": sum(key.startswith("mtp.") for key in prepared_shard_keys),
        "checked_non_mtp_tensor_count": checked,
        "mismatched_non_mtp_tensors": mismatched_tensors,
        "logical_non_mtp_weights_identical": not mismatched_tensors,
        "original_config_sha256": sha256(args.original / "config.json"),
        "prepared_config_sha256": sha256(args.prepared / "config.json"),
        "original_index_sha256": sha256(args.original / "model.safetensors.index.json"),
        "prepared_index_sha256": sha256(args.prepared / "model.safetensors.index.json"),
        "original_safetensors_bytes": sum(path.stat().st_size for path in args.original.glob("*.safetensors")),
        "prepared_safetensors_bytes": sum(path.stat().st_size for path in args.prepared.glob("*.safetensors")),
    }
    payload["valid"] = all([
        payload["config_change_is_architectures_only"],
        payload["prepared_index_matches_original_without_mtp"],
        payload["prepared_shard_mtp_key_count"] == 0,
        payload["logical_non_mtp_weights_identical"],
    ])
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not payload["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
