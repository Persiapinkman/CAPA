#!/usr/bin/env python3
"""Open the committed V9 sealed test only after target checkpoint selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
STUDY_DIR = ROOT / "experiments/studies/planner_retry_safe_end_hard_residual_v9_qwen35_4b_v1"
DATASET_MANIFEST = ROOT / "data/datasets/planner_retry_safe_end_hard_residual_v9/manifest.json"
BUILDER = ROOT / "training/planner_grpo_seed_v1/scripts/build_planner_retry_safe_end_hard_residual_v9.py"
DEFAULT_MODEL = Path("/raid/zkq/models/Qwen3.5-4B")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_selection(
    selection: dict[str, Any], manifest: dict[str, Any], screen_dir: Path
) -> dict[str, Any]:
    if selection.get("status") != "promote" or selection.get("sealed_test_authorized") is not True:
        raise ValueError("selection decision does not authorize sealed-test opening")
    if selection.get("larger_reference_used_for_selection") is not False:
        raise ValueError("target selection was contaminated by the larger reference")
    selected = selection.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("selection decision has no selected target checkpoint")
    checkpoint = int(selected.get("checkpoint") or 0)
    if checkpoint not in {10, 20, 40}:
        raise ValueError(f"selected checkpoint is not preregistered: {checkpoint}")
    commitment = manifest.get("sealed_test_commitment")
    if not isinstance(commitment, dict) or commitment.get("materialized") is not False:
        raise ValueError("sealed test is already materialized or commitment is missing")
    if int(commitment.get("rows") or 0) != 432 or int(commitment.get("entities") or 0) != 24:
        raise ValueError("sealed-test commitment dimensions changed")
    adapter = screen_dir / f"checkpoint-{checkpoint}"
    if not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"selected adapter is missing: {adapter}")
    return {
        "checkpoint": checkpoint,
        "label": str(selected.get("label") or ""),
        "adapter": str(adapter.resolve()),
        "commitment_sha256": str(commitment["sha256"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-decision", type=Path, default=STUDY_DIR / "selection_decision.json")
    parser.add_argument("--dataset-manifest", type=Path, default=DATASET_MANIFEST)
    parser.add_argument("--screen-dir", type=Path, required=True)
    parser.add_argument("--model-name-or-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=STUDY_DIR / "sealed_test_opening.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection = load_json(args.selection_decision)
    before_manifest = load_json(args.dataset_manifest)
    selected = validate_selection(selection, before_manifest, args.screen_dir)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
    subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--model-name-or-path",
            str(args.model_name_or_path),
            "--materialize-test",
            "--confirm-materialize-test",
            "OPEN_V9_TEST",
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    after_manifest = load_json(args.dataset_manifest)
    after = after_manifest["sealed_test_commitment"]
    combined = STUDY_DIR / "sealed_test_data/combined_test_cases.jsonl"
    if after.get("materialized") is not True:
        raise RuntimeError("builder did not mark the sealed test as materialized")
    if str(after["sha256"]) != selected["commitment_sha256"]:
        raise RuntimeError("sealed-test commitment changed during materialization")
    if sha256_file(combined) != selected["commitment_sha256"]:
        raise RuntimeError("materialized combined test does not match its commitment")
    payload = {
        "schema_version": "1.0",
        "status": "sealed_test_materialized_once",
        "larger_reference_used_for_selection": False,
        "selected": selected,
        "selection_decision": str(args.selection_decision),
        "selection_decision_sha256": sha256_file(args.selection_decision),
        "combined_test_cases": str(combined),
        "combined_test_sha256": sha256_file(combined),
        "rows": 432,
        "entities": 24,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
