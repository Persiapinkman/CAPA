#!/usr/bin/env python3
"""Freeze the preregistered V7 optimizer subset after a passing support audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_SCENARIO = "fresh_retry_step2"
DETECTORS = ("qwen", "rex")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def freeze_optimizer_data(
    *,
    source_path: Path,
    support_decision_path: Path,
    accepted_scenarios_path: Path,
    output_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    decision = load_json(support_decision_path)
    accepted = tuple(
        line.strip()
        for line in accepted_scenarios_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    decision_scenarios = tuple(str(item) for item in decision.get("optimizer_scenarios") or [])
    if decision.get("status") != "pass" or not bool(decision.get("optimizer_authorized")):
        raise ValueError("support decision does not authorize optimizer steps")
    if accepted != decision_scenarios:
        raise ValueError("accepted-scenario file does not match the frozen support decision")
    if REQUIRED_SCENARIO not in accepted:
        raise ValueError(f"required scenario is absent: {REQUIRED_SCENARIO}")
    if len(set(accepted)) != len(accepted):
        raise ValueError("accepted scenarios must be unique")

    parsed_lines: list[tuple[str, dict[str, Any]]] = []
    for line_number, raw_line in enumerate(
        source_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        if not isinstance(row, dict):
            raise ValueError(f"{source_path}:{line_number}: row must be an object")
        parsed_lines.append((raw_line, row))
    if not parsed_lines:
        raise ValueError("source step data is empty")

    source_rows = [row for _, row in parsed_lines]
    dataset_ids = {str(row.get("dataset_id") or "") for row in source_rows}
    entities = {str(row.get("entity_id") or "") for row in source_rows}
    if len(dataset_ids) != 1 or "" in dataset_ids:
        raise ValueError("source rows must have one non-empty dataset_id")
    if "" in entities:
        raise ValueError("source rows must have non-empty entity_id values")

    selected = [
        (raw_line, row)
        for raw_line, row in parsed_lines
        if str(row.get("scenario_id") or "") in set(accepted)
    ]
    expected_pairs = {(entity, detector) for entity in entities for detector in DETECTORS}
    for scenario in accepted:
        scenario_rows = [row for _, row in selected if row.get("scenario_id") == scenario]
        observed_pairs = {
            (str(row.get("entity_id") or ""), str(row.get("detector_family") or ""))
            for row in scenario_rows
        }
        if observed_pairs != expected_pairs or len(scenario_rows) != len(expected_pairs):
            raise ValueError(f"scenario does not contain one row per entity/detector pair: {scenario}")
        if any(row.get("optimization_scope") != "primary_residual" for row in scenario_rows):
            raise ValueError(f"non-primary row selected for optimizer: {scenario}")

    case_ids = [str(row.get("case_id") or "") for _, row in selected]
    prompt_hashes = [str(row.get("prompt_sha256") or "") for _, row in selected]
    if "" in case_ids or len(case_ids) != len(set(case_ids)):
        raise ValueError("selected case_id values must be non-empty and unique")
    if "" in prompt_hashes or len(prompt_hashes) != len(set(prompt_hashes)):
        raise ValueError("selected prompt hashes must be non-empty and unique")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(raw_line + "\n" for raw_line, _ in selected), encoding="utf-8")
    selected_rows = [row for _, row in selected]
    step_counts = Counter(int(row["step_index"]) for row in selected_rows)
    scenario_counts = Counter(str(row["scenario_id"]) for row in selected_rows)
    detector_counts = Counter(str(row["detector_family"]) for row in selected_rows)
    manifest = {
        "schema_version": "1.0",
        "status": "frozen_support_selected_optimizer_data",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": next(iter(dataset_ids)),
        "role": "optimization_only",
        "selection_rule": "preregistered stochastic-support gate",
        "optimizer_authorized": True,
        "accepted_scenarios": list(accepted),
        "rows": len(selected_rows),
        "entities": len(entities),
        "allowed_step_indices": sorted(step_counts),
        "distribution": {
            "scenarios": dict(sorted(scenario_counts.items())),
            "detectors": dict(sorted(detector_counts.items())),
            "step_indices": {str(key): value for key, value in sorted(step_counts.items())},
        },
        "files": {
            "source_step_data": str(source_path),
            "support_decision": str(support_decision_path),
            "accepted_scenarios": str(accepted_scenarios_path),
            "step_data": str(output_path),
        },
        "sha256": {
            "source_step_data": sha256_file(source_path),
            "support_decision": sha256_file(support_decision_path),
            "accepted_scenarios": sha256_file(accepted_scenarios_path),
            "step_data": sha256_file(output_path),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--support-decision", type=Path, required=True)
    parser.add_argument("--accepted-scenarios", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = freeze_optimizer_data(
        source_path=args.source,
        support_decision_path=args.support_decision,
        accepted_scenarios_path=args.accepted_scenarios,
        output_path=args.output,
        manifest_path=args.manifest,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
