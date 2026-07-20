#!/usr/bin/env python3
"""Materialize the single-use, result-blind exact-scene V15 confirmation set."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import random
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.planner_grpo_seed_v1.scripts import (  # noqa: E402
    build_planner_retry_migrate_residual_v7 as v7,
)


STUDY_ID = "planner_retry_ladder_v15_confirmation_v1"
DATASET_ID = STUDY_ID
SCHEMA_VERSION = "1.0"
SPLIT = "sealed_confirmation"
CREATED_AT = "2026-07-20T08:50:00Z"
EXPECTED_SPEC_SHA256 = "93c4ace6954b2cd2614118d8881e4bbb8d7e866cdb7defe1f89000c5033796fb"
MATERIALIZATION_TOKEN = "FREEZE_V15_CONFIRMATION"
STUDY_DIR = ROOT / "experiments/studies" / STUDY_ID
SPEC_PATH = STUDY_DIR / "generation_spec.json"
SEALED_DIR = STUDY_DIR / "sealed_data"
CASES_PATH = SEALED_DIR / "v15_confirmation_cases.jsonl"
SEALED_MANIFEST_PATH = STUDY_DIR / "sealed_manifest.json"
CONTAMINATION_PATH = STUDY_DIR / "contamination_audit.json"
DATASET_DIR = ROOT / "data/datasets" / DATASET_ID
DATASET_MANIFEST_PATH = DATASET_DIR / "manifest.json"
FIXTURE_DIR = ROOT / "examples/images" / DATASET_ID
SCENARIOS = ("post_retry_metric_veto_step3", "current_success_step2")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for row in rows
    ).encode("utf-8")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


def load_spec() -> dict[str, Any]:
    if sha256_file(SPEC_PATH) != EXPECTED_SPEC_SHA256:
        raise ValueError("V15 generation spec changed")
    spec = load_json(SPEC_PATH)
    if spec.get("status") != "preregistered_before_v15_materialization":
        raise ValueError("V15 spec was not preregistered")
    design = spec["design"]
    expected = {
        "entities": 6,
        "detector_families": ["qwen", "rex"],
        "scenarios": list(SCENARIOS),
        "cases_per_scenario": 12,
        "cases_per_scenario_detector": 6,
        "total_cases": 24,
        "resampling": "forbidden",
        "ratio_adaptation": "forbidden",
    }
    for key, value in expected.items():
        if design.get(key) != value:
            raise ValueError(f"V15 design {key} changed")
    protocol = spec["common_inference_protocol"]
    if protocol.get("runs_per_arm") != 3 or protocol.get("max_new_tokens") != 4096:
        raise ValueError("V15 common inference protocol changed")
    if spec["frozen_ladder"]["scenario_weights"] != {
        SCENARIOS[0]: 111,
        SCENARIOS[1]: 14,
    }:
        raise ValueError("V15 weights changed")
    return spec


def fixture_specs(spec: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    fixtures = spec["static_content"]["fixtures"]
    if not isinstance(fixtures, list) or len(fixtures) != 2:
        raise ValueError("V15 requires two fixtures")
    result = []
    for fixture in fixtures:
        item = dict(fixture)
        item["fg"] = tuple(item["fg"])
        item["bg"] = tuple(item["bg"])
        result.append(item)
    return tuple(result)


def error_aliases(spec: dict[str, Any]) -> dict[str, tuple[str, str]]:
    aliases = spec["static_content"]["error_aliases"]
    if set(aliases) != {"timeout", "transport", "quota", "payload"}:
        raise ValueError("V15 error families changed")
    result = {key: tuple(map(str, value)) for key, value in aliases.items()}
    if any(len(value) != 2 for value in result.values()):
        raise ValueError("V15 requires two aliases per family")
    return result


def factor_layout(spec: dict[str, Any]) -> list[dict[str, int]]:
    combinations = list(itertools.product(range(1), range(3), range(2)))
    random.Random(int(spec["design"]["seed"]) + 211).shuffle(combinations)
    return [
        {"style_index": style, "badge_index": badge, "fixture_index": fixture}
        for style, badge, fixture in combinations
    ]


def alias_layout(spec: dict[str, Any]) -> list[tuple[str, str]]:
    aliases = [
        (family, alias)
        for family, values in error_aliases(spec).items()
        for alias in values
    ]
    layout = aliases + aliases[:4]
    random.Random(int(spec["design"]["seed"]) + 307).shuffle(layout)
    if len(layout) != 12:
        raise ValueError("V15 alias layout changed")
    return layout


def entity_id(_split: str, entity_index: int) -> str:
    return f"prlv15_sc_entity_{entity_index + 1:03d}"


@contextlib.contextmanager
def configured_v15(spec: dict[str, Any]) -> Iterator[None]:
    split_specs = {
        SPLIT: {
            "entities": 6,
            "code": "SC",
            "training_only": False,
            "evaluation_only": True,
            "exclude_from_training": True,
            "sealed": True,
            "role": "single_use_blind_confirmation_v15",
        }
    }
    static = spec["static_content"]
    lexicons = {
        SPLIT: {
            key: tuple(map(str, static["lexicon"][key]))
            for key in ("roots", "suffixes", "styles")
        }
    }
    policy = {SPLIT: tuple(map(str, static["policy_wording"]))}
    aliases = {SPLIT: error_aliases(spec)}
    fixtures = {SPLIT: fixture_specs(spec)}
    names = (
        "DATASET_ID", "SCHEMA_VERSION", "SEED", "CREATED_AT", "STUDY_ID",
        "DATASET_DIR", "FIXTURE_DIR", "STUDY_DIR", "PRIMARY_SCENARIOS",
        "CONTROL_SCENARIOS", "ALL_SCENARIOS", "SPLIT_SPECS", "LEXICONS",
        "POLICY_WORDING", "ERROR_ALIASES", "FIXTURES", "PROJECTS",
        "FACTOR_LAYOUTS", "ALIAS_LAYOUTS", "entity_id", "base_case",
    )
    saved = {name: getattr(v7, name) for name in names}
    original_base_case = v7.base_case
    try:
        v7.DATASET_ID = DATASET_ID
        v7.SCHEMA_VERSION = SCHEMA_VERSION
        v7.SEED = int(spec["design"]["seed"])
        v7.CREATED_AT = CREATED_AT
        v7.STUDY_ID = STUDY_ID
        v7.DATASET_DIR = DATASET_DIR
        v7.FIXTURE_DIR = FIXTURE_DIR
        v7.STUDY_DIR = STUDY_DIR
        v7.PRIMARY_SCENARIOS = SCENARIOS
        v7.CONTROL_SCENARIOS = ()
        v7.ALL_SCENARIOS = SCENARIOS
        v7.SPLIT_SPECS = split_specs
        v7.LEXICONS = lexicons
        v7.POLICY_WORDING = policy
        v7.ERROR_ALIASES = aliases
        v7.FIXTURES = fixtures
        v7.entity_id = entity_id
        v7.PROJECTS = {SPLIT: v7.build_projects(SPLIT)}
        v7.FACTOR_LAYOUTS = {SPLIT: factor_layout(spec)}
        v7.ALIAS_LAYOUTS = {SPLIT: alias_layout(spec)}

        def base_case_v15(**kwargs: Any) -> dict[str, Any]:
            row = original_base_case(**kwargs)
            row["case_id"] = str(row["case_id"]).replace("PRRV7-", "PRLV15-", 1)
            row["template_id"] = str(row["template_id"]).replace("prrv7_", "prlv15_", 1)
            row["query_style_index"] = 2
            row["grpo_eligible"] = False
            row["difficulty_family"] = "exact_scene_entity_lexicon_blind_confirmation_v15"
            row["support_block"] = ""
            row["provenance_class"] = "independent_preregistered_exact_scene_v15"
            return row

        v7.base_case = base_case_v15
        yield
    finally:
        for name, value in saved.items():
            setattr(v7, name, value)


def build_cases_in_memory(spec: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    spec = spec or load_spec()
    with configured_v15(spec):
        rows = [
            case
            for entity_index in range(6)
            for detector_index in range(len(v7.DETECTORS))
            for case in v7.make_bundle(SPLIT, entity_index, detector_index)
            if str(case["scenario_id"]) in SCENARIOS
        ]
    random.Random(int(spec["design"]["seed"]) + 401).shuffle(rows)
    return rows


def validate_cases(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    case_ids = [str(row.get("case_id") or "") for row in rows]
    if len(rows) != 24 or len(set(case_ids)) != 24 or not all(case_ids):
        errors.append("V15 requires exact24 unique cases")
    scenarios = Counter(str(row.get("scenario_id")) for row in rows)
    cells = Counter(f"{row.get('scenario_id')}|{row.get('detector_family')}" for row in rows)
    entities: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in rows:
        entities[str(row["entity_id"])].add(
            (str(row["scenario_id"]), str(row["detector_family"]))
        )
        expected_flags = {
            "dataset_id": DATASET_ID,
            "split": SPLIT,
            "selection_role": "single_use_blind_confirmation_v15",
            "training_only": False,
            "evaluation_only": True,
            "exclude_from_training": True,
            "sealed": True,
            "grpo_eligible": False,
            "sft_eligible": False,
            "query_style_index": 2,
        }
        for key, expected in expected_flags.items():
            if row.get(key) != expected:
                errors.append(f"{row.get('case_id')}: {key} changed")
        if not bool(v7.v6.score_case(row)["passed"]):
            errors.append(f"{row.get('case_id')}: canonical trajectory failed")
    if scenarios != Counter({SCENARIOS[0]: 12, SCENARIOS[1]: 12}):
        errors.append(f"scenario counts changed: {dict(scenarios)}")
    expected_cells = Counter(
        {f"{scenario}|{detector}": 6 for scenario in SCENARIOS for detector in ("qwen", "rex")}
    )
    if cells != expected_cells:
        errors.append(f"scenario-detector counts changed: {dict(cells)}")
    expected_pairs = {(scenario, detector) for scenario in SCENARIOS for detector in ("qwen", "rex")}
    if len(entities) != 6 or any(pairs != expected_pairs for pairs in entities.values()):
        errors.append("V15 entity clusters are incomplete")
    representatives = {entity: next(row for row in rows if row["entity_id"] == entity) for entity in entities}
    factorial = Counter(
        (str(row["badge_condition"]), str(row["image_fixture_family"]))
        for row in representatives.values()
    )
    if len(factorial) != 6 or set(factorial.values()) != {1}:
        errors.append("V15 badge x fixture factorial is incomplete")
    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "rows": len(rows),
        "entities": len(entities),
        "scenario_counts": dict(sorted(scenarios.items())),
        "scenario_detector_counts": dict(sorted(cells.items())),
        "factorial_cells": len(factorial),
        "case_ids_sha256": sha256_bytes("\n".join(sorted(case_ids)).encode("utf-8")),
        "canonical_complete_trajectory_pass_rate": (
            sum(bool(v7.v6.score_case(row)["passed"]) for row in rows) / len(rows)
        ),
    }


def protected_tokens(spec: dict[str, Any]) -> list[str]:
    static = spec["static_content"]
    values: list[str] = []
    for key in ("roots", "suffixes", "styles"):
        values.extend(map(str, static["lexicon"][key]))
    for aliases in static["error_aliases"].values():
        values.extend(map(str, aliases))
    for fixture in static["fixtures"]:
        values.extend(map(str, (fixture["target"], fixture["slug"], fixture["family"])))
    if not all(values) or len(values) != len(set(values)):
        raise ValueError("V15 protected tokens are empty or duplicated")
    return values


def contamination_audit(spec: dict[str, Any]) -> dict[str, Any]:
    overlaps: list[dict[str, str]] = []
    files_scanned = 0
    tokens = protected_tokens(spec)
    extensions = {".json", ".jsonl", ".md", ".py", ".sh", ".yaml", ".yml"}
    for relative_root in spec["anti_leakage_contract"]["reference_roots"]:
        directory = ROOT / relative_root
        if not directory.exists():
            raise FileNotFoundError(directory)
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            files_scanned += 1
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in tokens:
                if token in text:
                    overlaps.append({"token": token, "path": str(path.relative_to(ROOT))})
    return {
        "schema_version": "1.0",
        "study_id": STUDY_ID,
        "status": "pass" if not overlaps else "fail",
        "files_scanned": files_scanned,
        "protected_tokens": len(tokens),
        "protected_tokens_sha256": sha256_bytes("\n".join(sorted(tokens)).encode("utf-8")),
        "exact_overlaps": overlaps,
        "model_outputs_used_for_case_selection": False,
        "v13_prediction_or_result_files_used_for_generation": False,
    }


def ensure_outputs_absent() -> None:
    for path in (CASES_PATH, SEALED_MANIFEST_PATH, CONTAMINATION_PATH, DATASET_MANIFEST_PATH, FIXTURE_DIR):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")


def write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)


def materialize(spec: dict[str, Any]) -> dict[str, Any]:
    ensure_outputs_absent()
    rows = build_cases_in_memory(spec)
    validation = validate_cases(rows)
    if validation["status"] != "pass":
        raise ValueError(validation["errors"])
    contamination = contamination_audit(spec)
    if contamination["status"] != "pass":
        raise ValueError(f"V15 protected-token overlap: {contamination['exact_overlaps']}")

    fixtures = fixture_specs(spec)
    fixture_payloads: dict[str, bytes] = {}
    with tempfile.TemporaryDirectory(prefix="prlv15-fixtures-") as temp:
        temp_dir = Path(temp)
        for index, fixture in enumerate(fixtures):
            path = temp_dir / f"{fixture['slug']}.png"
            v7.v6.draw_fixture(path, fixture, index=150 + index)
            fixture_payloads[path.name] = path.read_bytes()
    new_hashes = {name: sha256_bytes(data) for name, data in fixture_payloads.items()}
    if len(set(new_hashes.values())) != len(new_hashes):
        raise ValueError("V15 fixtures are not content-unique")
    historical_hashes = {
        sha256_file(path) for path in (ROOT / "examples/images").rglob("*.png") if path.is_file()
    }
    if historical_hashes.intersection(new_hashes.values()):
        raise ValueError("V15 fixture image overlaps historical content")

    cases_data = jsonl_bytes(rows)
    manifest = {
        "schema_version": "1.0",
        "study_id": STUDY_ID,
        "dataset_id": DATASET_ID,
        "status": "sealed",
        "created_at": CREATED_AT,
        "generation_spec_path": str(SPEC_PATH.relative_to(ROOT)),
        "generation_spec_sha256": EXPECTED_SPEC_SHA256,
        "cases_path": str(CASES_PATH.relative_to(ROOT)),
        "cases_sha256": sha256_bytes(cases_data),
        "rows": 24,
        "validation": validation,
        "contamination_audit_path": str(CONTAMINATION_PATH.relative_to(ROOT)),
        "fixture_sha256": dict(sorted(new_hashes.items())),
        "model_outputs_used_for_case_selection": False,
        "resampling_performed": False,
        "v15_opened_for_inference": False,
    }
    write_new(CASES_PATH, cases_data)
    write_new(CONTAMINATION_PATH, json_bytes(contamination))
    write_new(SEALED_MANIFEST_PATH, json_bytes(manifest))
    write_new(DATASET_MANIFEST_PATH, json_bytes(manifest))
    FIXTURE_DIR.mkdir(parents=True)
    for name, data in fixture_payloads.items():
        write_new(FIXTURE_DIR / name, data)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialize-token", required=True)
    args = parser.parse_args()
    if args.materialize_token != MATERIALIZATION_TOKEN:
        raise ValueError("invalid V15 materialization token")
    payload = materialize(load_spec())
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
