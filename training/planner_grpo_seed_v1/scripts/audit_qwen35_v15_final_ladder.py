#!/usr/bin/env python3
"""Open once and aggregate the exact-scene V15 four-arm confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.planner_grpo_seed_v1.scripts.reward_planner_grpo import score_case  # noqa: E402


EXPECTED_CONFIG_SHA256 = "063932854510fafe7b7952ed8a3f2d937bbdbf98f78710b3dec849b8ebfacae0"
ARMS = ("qwen35_4b_base", "qwen35_4b_sft", "qwen35_35b_a3b", "qwen35_4b_grpo_n64")
METRIC = "post_retry_metric_veto_step3"
CURRENT = "current_success_step2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: expected object rows")
    return rows


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} hash changed: {observed} != {expected}")
    return observed


def validate_config(path: Path) -> dict[str, Any]:
    require_hash(path, EXPECTED_CONFIG_SHA256, "V15 config")
    config = load_json(path)
    if config.get("study_id") != "planner_retry_ladder_v15_confirmation_v1":
        raise ValueError("V15 study changed")
    sealed = config["sealed_confirmation"]
    if sealed.get("rows") != 24 or sealed.get("entity_clusters") != 6:
        raise ValueError("V15 sealed geometry changed")
    protocol = config["protocol"]
    expected_protocol = {
        "runs_per_arm": 3,
        "temperature": 0,
        "top_p": 1,
        "do_sample": False,
        "max_steps": 3,
        "max_new_tokens": 4096,
        "exact_prediction_rows_per_run": 24,
        "selective_case_arm_or_run_retry": False,
        "missing_prediction_row_or_run_invalidates_confirmation": True,
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(f"V15 protocol {key} changed")
    analysis = config["analysis"]
    if analysis["scenario_weights"] != {METRIC: 111, CURRENT: 14}:
        raise ValueError("V15 weights changed")
    if analysis["required_mean_order"] != list(ARMS):
        raise ValueError("V15 arm order changed")
    if config["execution"].get("resume_or_partial_rerun") is not False:
        raise ValueError("V15 partial rerun policy changed")
    return config


def validate_assets(config: dict[str, Any], *, include_cases: bool) -> dict[str, str]:
    hashes: dict[str, str] = {}
    spec = config["generation_spec"]
    hashes["generation_spec"] = require_hash(resolve(spec["path"]), spec["sha256"], "V15 spec")
    sealed = config["sealed_confirmation"]
    for key, label in (
        ("manifest", "sealed manifest"),
        ("dataset_manifest", "dataset manifest"),
        ("contamination_audit", "contamination audit"),
    ):
        hashes[key] = require_hash(
            resolve(sealed[f"{key}_path"]), sealed[f"{key}_sha256"], label
        )
    contamination = load_json(resolve(sealed["contamination_audit_path"]))
    if contamination.get("status") != "pass" or contamination.get("exact_overlaps") != []:
        raise ValueError("V15 contamination audit failed")
    if include_cases:
        hashes["cases"] = require_hash(resolve(sealed["cases_path"]), sealed["cases_sha256"], "V15 cases")
    for relative, expected in sealed["fixture_sha256"].items():
        hashes[f"fixture:{relative}"] = require_hash(resolve(relative), expected, relative)
    model_path = Path(config["arms"]["qwen35_4b_base"]["model_path"])
    for filename, expected in config["model_files_sha256"].items():
        hashes[f"model:{filename}"] = require_hash(model_path / filename, expected, filename)
    for arm in ("qwen35_4b_sft", "qwen35_4b_grpo_n64"):
        item = config["arms"][arm]
        adapter = Path(item["adapter_path"])
        for filename, expected in item["adapter_files_sha256"].items():
            hashes[f"{arm}:{filename}"] = require_hash(adapter / filename, expected, f"{arm} {filename}")
    candidate = config["arms"]["qwen35_4b_grpo_n64"]
    hashes["candidate_selection"] = require_hash(
        Path(candidate["selection_receipt_path"]), candidate["selection_receipt_sha256"], "n64 selection"
    )
    hashes["candidate_health"] = require_hash(
        Path(candidate["training_health_path"]), candidate["training_health_sha256"], "n64 health"
    )
    selection = load_json(Path(candidate["selection_receipt_path"]))
    health = load_json(Path(candidate["training_health_path"]))
    if selection.get("status") not in {"pass", "selected_before_v15_materialization"} or selection.get("selected", {}).get("adapter_sha256") != candidate["adapter_files_sha256"]["adapter_model.safetensors"]:
        raise ValueError("n64 selection is not the frozen passing adapter")
    final_health = health.get("log_summary", {}).get("final", {})
    raw_training_health_passed = (
        health.get("status") == "completed"
        and int(health.get("optimizer_steps") or 0) == 1
        and final_health.get("completions/clipped_ratio") == 0
        and float(final_health.get("reward_std") or 0) > 0
        and float(final_health.get("advantage/std") or 0) > 0
        and float(final_health.get("grad_norm") or 0) > 0
    )
    structured_health_passed = health.get("status") == "pass" and all(health.get("checks", {}).values())
    if not (raw_training_health_passed or structured_health_passed):
        raise ValueError("n64 training health changed")
    return hashes


def validate_cases(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sealed = config["sealed_confirmation"]
    cases = load_jsonl(resolve(sealed["cases_path"]))
    ids = [str(row.get("case_id") or "") for row in cases]
    if len(cases) != 24 or len(set(ids)) != 24 or not all(ids):
        raise ValueError("V15 requires exact24 unique cases")
    identity = hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()
    if identity != sealed["case_ids_sha256"]:
        raise ValueError("V15 case identity changed")
    scenarios = Counter(str(row.get("scenario_id")) for row in cases)
    cells = Counter(f"{row.get('scenario_id')}|{row.get('detector_family')}" for row in cases)
    entities: dict[str, set[tuple[str, str]]] = defaultdict(set)
    badges: Counter[str] = Counter()
    fixtures: Counter[str] = Counter()
    for row in cases:
        if row.get("split") != "sealed_confirmation" or row.get("query_style_index") != 2:
            raise ValueError("V15 scene metadata changed")
        entities[str(row["entity_id"])].add((str(row["scenario_id"]), str(row["detector_family"])))
        badges[str(row["badge_condition"])] += 1
        fixtures[str(row["image_fixture_family"])] += 1
    if dict(scenarios) != sealed["scenario_counts"] or dict(cells) != sealed["scenario_detector_counts"]:
        raise ValueError("V15 scenario geometry changed")
    expected_pairs = {(scenario, detector) for scenario in (METRIC, CURRENT) for detector in ("qwen", "rex")}
    if len(entities) != 6 or any(value != expected_pairs for value in entities.values()):
        raise ValueError("V15 entity clusters changed")
    scene = config["analysis"]["scene"]
    if sorted(badges.values()) != sorted(scene["expected_badge_cell_counts"]):
        raise ValueError("V15 badge geometry changed")
    if sorted(fixtures.values()) != sorted(scene["expected_image_fixture_cell_counts"]):
        raise ValueError("V15 fixture geometry changed")
    return cases, {
        "rows": 24,
        "entity_clusters": 6,
        "scenario_counts": dict(scenarios),
        "scenario_detector_counts": dict(cells),
        "badge_cell_counts_sorted": sorted(badges.values()),
        "image_fixture_cell_counts_sorted": sorted(fixtures.values()),
        "case_ids_listed": False,
        "case_ids_sha256": identity,
    }


def committed_harness(config_path: Path) -> dict[str, Any]:
    relative = config_path.resolve().relative_to(ROOT.resolve())
    committed = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
    if committed != config_path.read_bytes():
        raise ValueError("V15 config is not committed")
    if subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip():
        raise ValueError("V15 opening requires clean worktree")
    paths = {
        "auditor": Path(__file__).resolve(),
        "all_runner": ROOT / "scripts/run_qwen35_v15_final_all_scopes.sh",
        "local_runner": ROOT / "scripts/run_qwen35_v15_local_4b_final_eval.sh",
        "larger_runner": ROOT / "scripts/run_qwen35_v15_35b_final_eval.sh",
    }
    return {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "files": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()},
    }


def write_new(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)


def open_confirmation(config_path: Path, output: Path) -> dict[str, Any]:
    config = validate_config(config_path)
    assets = validate_assets(config, include_cases=True)
    _, geometry = validate_cases(config)
    harness = committed_harness(config_path)
    if output.exists() or output.parent.exists():
        raise FileExistsError(f"refusing to reuse V15 output root {output.parent}")
    receipt = {
        "schema_version": "1.0",
        "study_id": config["study_id"],
        "status": "opened_once",
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.resolve()),
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "assets": assets,
        "geometry": geometry,
        "harness": harness,
        "frozen_arms": list(ARMS),
        "runs_per_arm": 3,
        "selective_rerun_permitted": False,
        "v15_used_for_selection": False,
    }
    write_new(output, json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return receipt


def verify_opening(config_path: Path, receipt_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = validate_config(config_path)
    receipt = load_json(receipt_path)
    if receipt.get("status") != "opened_once" or receipt.get("config_sha256") != EXPECTED_CONFIG_SHA256:
        raise ValueError("invalid V15 opening receipt")
    current_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if receipt.get("harness", {}).get("git_commit") != current_commit:
        raise ValueError("V15 harness commit changed after opening")
    return config, receipt


def prediction_index(path: Path, expected_ids: set[str]) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    indexed = {str(row.get("case_id") or ""): row for row in rows}
    if len(rows) != 24 or len(indexed) != 24 or set(indexed) != expected_ids:
        raise ValueError(f"{path}: incomplete V15 prediction coverage")
    return indexed


def case_passed(case: dict[str, Any], prediction: dict[str, Any]) -> bool:
    if prediction.get("errors"):
        return False
    try:
        return score_case(case, prediction, use_expected_when_missing=False).get("passed") is True
    except Exception:
        return False


def weighted(counts: dict[str, int]) -> float:
    return 100.0 * ((111 * counts[METRIC] / 12) + (14 * counts[CURRENT] / 12)) / 125


def adjacent_pairs(values: list[float]) -> list[tuple[float, float]]:
    """Return every adjacent pair while retaining strict length validation."""
    if len(values) < 2:
        raise ValueError("at least two values are required for an adjacent comparison")
    return list(zip(values[:-1], values[1:], strict=True))


def aggregate(config_path: Path, receipt_path: Path, output_root: Path) -> dict[str, Any]:
    config, receipt = verify_opening(config_path, receipt_path)
    validate_assets(config, include_cases=True)
    cases, geometry = validate_cases(config)
    expected_ids = {str(row["case_id"]) for row in cases}
    table = []
    artifacts: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        prefix = config["arms"][arm]["report_prefix"]
        arm_dir = output_root / "raw" / arm
        per_run = []
        artifacts[arm] = {}
        for run in (1, 2, 3):
            path = arm_dir / f"{prefix}_run{run}_predictions.jsonl"
            indexed = prediction_index(path, expected_ids)
            counts = {METRIC: 0, CURRENT: 0}
            error_cases = 0
            for case in cases:
                prediction = indexed[str(case["case_id"])]
                error_cases += int(bool(prediction.get("errors")))
                if case_passed(case, prediction):
                    counts[str(case["scenario_id"])] += 1
            per_run.append(
                {
                    "run": run,
                    "scenario_passed": counts,
                    "weighted_strict_percent": weighted(counts),
                    "prediction_error_cases": error_cases,
                }
            )
            artifacts[arm][str(run)] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "rows": 24,
            }
        values = [row["weighted_strict_percent"] for row in per_run]
        table.append(
            {
                "arm": arm,
                "run1_weighted_strict_percent": values[0],
                "run2_weighted_strict_percent": values[1],
                "run3_weighted_strict_percent": values[2],
                "mean_weighted_strict_percent": sum(values) / 3,
                "run_range_pp": max(values) - min(values),
                "per_run_scenario_passed": per_run,
            }
        )
    indexed_table = {row["arm"]: row for row in table}
    means = [indexed_table[arm]["mean_weighted_strict_percent"] for arm in ARMS]
    mean_pairs = adjacent_pairs(means)
    larger_runs = [
        indexed_table["qwen35_35b_a3b"][f"run{run}_weighted_strict_percent"]
        for run in (1, 2, 3)
    ]
    analysis = config["analysis"]
    hard_gates = {
        "complete_four_arms_three_runs_exact24": True,
        "strict_mean_order_base_sft_larger_grpo": all(left < right for left, right in mean_pairs),
        "base_mean_strictly_below_65": means[0] < analysis["base_mean_ceiling_percent_exclusive"],
        "larger_mean_strictly_above_85": means[2] > analysis["larger_mean_floor_percent_exclusive"],
        "larger_each_run_strictly_above_85": all(value > analysis["larger_each_run_floor_percent_exclusive"] for value in larger_runs),
        "larger_run_range_at_most_5pp": max(larger_runs) - min(larger_runs) <= analysis["larger_run_range_ceiling_pp_inclusive"],
        "grpo_mean_strictly_exceeds_larger": means[3] > means[2],
        "confirmation_not_used_for_selection": analysis["confirmation_used_for_selection"] is False,
        "no_selective_rerun": receipt.get("selective_rerun_permitted") is False,
    }
    report = {
        "schema_version": "1.0",
        "study_id": config["study_id"],
        "status": "pass" if all(hard_gates.values()) else "fail",
        "weighted_formula": "(111 * metric_veto_rate + 14 * current_success_rate) / 125",
        "scene_geometry": geometry,
        "table": table,
        "sorted_by_mean_ascending": [arm for _, arm in sorted(zip(means, ARMS, strict=True))],
        "hard_gates": hard_gates,
        "minimum_adjacent_mean_margin_pp": min(right - left for left, right in mean_pairs),
        "grpo_minus_larger_mean_pp": means[3] - means[2],
        "artifacts": artifacts,
        "opening_receipt": {"path": str(receipt_path.resolve()), "sha256": sha256_file(receipt_path)},
        "model_lineage": {
            "qwen35_4b_grpo_n64": "targeted-SFT warm-start plus exactly one GRPO optimizer step at learning rate 2e-8",
            "comparison_sft_arm": "frozen original Qwen3.5-4B SFT checkpoint-100",
        },
    }
    return report


def markdown_table(report: dict[str, Any]) -> str:
    labels = {
        "qwen35_4b_base": "Qwen3.5-4B Base",
        "qwen35_4b_sft": "Qwen3.5-4B SFT",
        "qwen35_35b_a3b": "Qwen3.5-35B-A3B",
        "qwen35_4b_grpo_n64": "Qwen3.5-4B targeted-SFT + one-step GRPO (LR 2e-8)",
    }
    lines = [
        "| Model | Run 1 (%) | Run 2 (%) | Run 3 (%) | Mean (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["table"]:
        lines.append(
            f"| {labels[row['arm']]} | {row['run1_weighted_strict_percent']:.4f} | "
            f"{row['run2_weighted_strict_percent']:.4f} | {row['run3_weighted_strict_percent']:.4f} | "
            f"{row['mean_weighted_strict_percent']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Status: **{report['status']}**",
            "",
            "Scene: exact24 entity/lexicon-disjoint V15 cases; weights metric-veto:current-success = 111:14.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    open_parser = subparsers.add_parser("open")
    open_parser.add_argument("--config", type=Path, required=True)
    open_parser.add_argument("--output", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify-opening")
    verify_parser.add_argument("--config", type=Path, required=True)
    verify_parser.add_argument("--opening-receipt", type=Path, required=True)
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--config", type=Path, required=True)
    aggregate_parser.add_argument("--opening-receipt", type=Path, required=True)
    aggregate_parser.add_argument("--output-root", type=Path, required=True)
    aggregate_parser.add_argument("--report-output", type=Path, required=True)
    aggregate_parser.add_argument("--table-output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "open":
        payload = open_confirmation(args.config, args.output)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "verify-opening":
        _, payload = verify_opening(args.config, args.opening_receipt)
        print(json.dumps({"status": "pass", "opened_at": payload["opened_at"]}, indent=2))
    else:
        payload = aggregate(args.config, args.opening_receipt, args.output_root)
        if args.report_output.exists() or args.table_output.exists():
            raise FileExistsError("refusing to overwrite V15 aggregate outputs")
        write_new(args.report_output, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        write_new(args.table_output, markdown_table(payload))
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
