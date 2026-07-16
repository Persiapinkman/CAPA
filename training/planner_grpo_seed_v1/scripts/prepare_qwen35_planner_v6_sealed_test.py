#!/usr/bin/env python3
"""Materialize the V6 entity-disjoint test rows only after model selection.

The source test cases are part of the immutable dataset manifest, but their
Qwen3.5 chat-formatted rows are intentionally produced only after SFT selection
and both GRPO support-gate decisions have been frozen.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.planner_grpo_seed_v1.scripts import build_planner_retry_migrate_v6 as v6  # noqa: E402


STUDY_DIR = ROOT / "experiments/studies/planner_retry_migrate_v6_qwen35_4b_v1"
DEFAULT_CASE_DIR = ROOT / "training/planner_grpo_seed_v1/cases"
DEFAULT_STAGE_DIR = ROOT / "training/planner_grpo_seed_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/datasets/planner_retry_migrate_v6/manifest.json",
    )
    parser.add_argument(
        "--test-cases",
        type=Path,
        default=DEFAULT_CASE_DIR / "planner_retry_migrate_v6_test_cases.jsonl",
    )
    parser.add_argument("--model-name-or-path", type=Path, default=Path("/raid/zkq/models/Qwen3.5-4B"))
    parser.add_argument("--output-dir", type=Path, default=STUDY_DIR / "sealed_test_data")
    parser.add_argument("--selection-file", type=Path, default=STUDY_DIR / "sft_selection.json")
    parser.add_argument(
        "--grpo-decisions",
        type=Path,
        nargs=2,
        default=[
            STUDY_DIR / "grpo_support_checkpoint100_decision.json",
            STUDY_DIR / "grpo_support_checkpoint75_decision.json",
        ],
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def detector_family_for_row(row: dict[str, Any]) -> str:
    source_ids = " ".join(str(value) for value in row.get("source_case_ids", []))
    matches = [family for marker, family in (("-QWEN-", "qwen"), ("-REX-", "rex")) if marker in source_ids]
    if len(matches) != 1:
        raise ValueError(f"cannot recover one detector family from source_case_ids={row.get('source_case_ids')!r}")
    return matches[0]


def assert_source_contract(cases: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    expected_cases = int(manifest["stats"]["test"]["cases"])
    expected_entities = int(manifest["construction"]["split_entity_counts"]["test"])
    entity_ids = {str(row.get("entity_id") or "") for row in cases}
    if len(cases) != expected_cases:
        raise ValueError(f"test case count mismatch: {len(cases)} != {expected_cases}")
    if "" in entity_ids or len(entity_ids) != expected_entities:
        raise ValueError(f"test entity count mismatch: {len(entity_ids - {''})} != {expected_entities}")

    required = {
        "split": "test",
        "evaluation_only": True,
        "exclude_from_training": True,
        "training_only": False,
        "sealed": True,
    }
    for field, expected in required.items():
        bad = sum(row.get(field) != expected for row in cases)
        if bad:
            raise ValueError(f"{bad} test cases violate {field}={expected!r}")


def comparison_case_paths() -> list[Path]:
    return [
        DEFAULT_CASE_DIR / "planner_retry_migrate_v6_sft_train_cases.jsonl",
        DEFAULT_CASE_DIR / "planner_retry_migrate_v6_sft_dev_cases.jsonl",
        DEFAULT_CASE_DIR / "planner_retry_migrate_v6_grpo_train_cases.jsonl",
        DEFAULT_CASE_DIR / "planner_retry_migrate_v6_grpo_dev_cases.jsonl",
    ]


def comparison_stage_paths() -> list[Path]:
    return [
        DEFAULT_STAGE_DIR / "sft_data_planner_retry_migrate_v6_qwen35_nothinking/train.jsonl",
        DEFAULT_STAGE_DIR / "sft_data_planner_retry_migrate_v6_qwen35_nothinking/dev.jsonl",
        DEFAULT_STAGE_DIR / "step_data/planner_retry_migrate_v6_grpo_train_qwen35_4b_nothinking_step2.jsonl",
        DEFAULT_STAGE_DIR / "step_data/planner_retry_migrate_v6_grpo_dev_qwen35_4b_nothinking_step2.jsonl",
    ]


def main() -> None:
    args = parse_args()
    manifest = load_json(args.manifest)
    if manifest.get("dataset_id") != v6.DATASET_ID:
        raise ValueError(f"wrong manifest dataset: {manifest.get('dataset_id')!r}")
    expected_source_hash = str(manifest["sha256"]["cases_test"])
    actual_source_hash = v6.sha256_file(args.test_cases)
    if actual_source_hash != expected_source_hash:
        raise ValueError(f"test source hash mismatch: {actual_source_hash} != {expected_source_hash}")

    selection = load_json(args.selection_file)
    if selection.get("status") != "selected" or selection.get("selected", {}).get("name") != "checkpoint-100":
        raise ValueError("SFT selection is not frozen at checkpoint-100")
    decision_payloads = [load_json(path) for path in args.grpo_decisions]
    if any(payload.get("status") != "optimizer_not_authorized" for payload in decision_payloads):
        raise ValueError("both GRPO support-gate decisions must be frozen before test materialization")

    test_cases = v6.load_jsonl(args.test_cases)
    assert_source_contract(test_cases, manifest)

    isolation_fields = ("case_id", "entity_id", "project_entity", "target_entity", "counterfactual_bundle_id")
    train_dev_cases = [row for path in comparison_case_paths() for row in v6.load_jsonl(path)]
    isolation_overlaps: dict[str, list[str]] = {}
    for field in isolation_fields:
        test_values = {str(row.get(field) or "") for row in test_cases} - {""}
        other_values = {str(row.get(field) or "") for row in train_dev_cases} - {""}
        isolation_overlaps[field] = sorted(test_values & other_values)
    if any(isolation_overlaps.values()):
        raise ValueError(f"test source leakage: {isolation_overlaps}")

    tokenizer = v6._load_tokenizer(args.model_name_or_path)
    rows = v6.build_sft_rows(test_cases, tokenizer)
    audit = v6.prompt_audit(rows)
    if audit.get("status") != "pass":
        raise ValueError(f"formatted test prompt audit failed: {audit}")
    if any(row.get("split") != "test" for row in rows):
        raise ValueError("formatted output contains a non-test row")
    detector_families = [detector_family_for_row(row) for row in rows]

    prior_prompt_hashes = {
        str(row.get("prompt_sha256") or "")
        for path in comparison_stage_paths()
        for row in v6.load_jsonl(path)
    } - {""}
    test_prompt_hashes = {str(row.get("prompt_sha256") or "") for row in rows}
    prompt_overlap = sorted(test_prompt_hashes & prior_prompt_hashes)
    if prompt_overlap:
        raise ValueError(f"formatted prompt leakage: {len(prompt_overlap)} overlapping prompt hashes")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "test.jsonl"
    v6.write_jsonl(output_path, rows)
    metadata = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": v6.DATASET_ID,
        "role": "sealed_final_evaluation_only",
        "selection_frozen_before_materialization": {
            "sft_selection": display_path(args.selection_file),
            "sft_selection_sha256": v6.sha256_file(args.selection_file),
            "selected_checkpoint": selection["selected"],
            "grpo_decisions": [
                {
                    "path": display_path(path),
                    "sha256": v6.sha256_file(path),
                    "initializer": payload.get("initializer"),
                    "status": payload.get("status"),
                }
                for path, payload in zip(args.grpo_decisions, decision_payloads, strict=True)
            ],
        },
        "source": {
            "path": display_path(args.test_cases),
            "sha256": actual_source_hash,
            "manifest": display_path(args.manifest),
            "manifest_sha256": v6.sha256_file(args.manifest),
            "cases": len(test_cases),
            "entities": len({str(row["entity_id"]) for row in test_cases}),
        },
        "formatted": {
            "path": display_path(output_path),
            "sha256": v6.sha256_file(output_path),
            "rows": len(rows),
            "unique_prompt_hashes": len(test_prompt_hashes),
            "categories": dict(sorted(Counter(str(row.get("category") or "") for row in rows).items())),
            "detector_families": dict(sorted(Counter(detector_families).items())),
            "target_action_classes": dict(
                sorted(Counter(str(row.get("target_action_class") or "") for row in rows).items())
            ),
            "step_indices": dict(sorted(Counter(str(row.get("step_index") or "") for row in rows).items())),
        },
        "isolation": {
            "comparison_case_files": [display_path(path) for path in comparison_case_paths()],
            "comparison_stage_files": [display_path(path) for path in comparison_stage_paths()],
            "source_field_overlaps": {field: len(values) for field, values in isolation_overlaps.items()},
            "formatted_prompt_hash_overlap": len(prompt_overlap),
            "status": "pass",
        },
        "prompt_audit": audit,
        "human_review_status": manifest["integrity"]["independent_human_review_status"],
    }
    metadata_path = args.output_dir / "metadata.json"
    v6.write_json(metadata_path, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
