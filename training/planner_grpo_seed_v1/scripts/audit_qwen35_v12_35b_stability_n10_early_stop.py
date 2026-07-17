#!/usr/bin/env python3
"""Audit an intentionally early-stopped n10 run without inventing missing rows."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.planner_grpo_seed_v1.scripts.evaluate_qwen35_v12_35b_stability_n10 import (
    index_unique,
    load_json,
    load_jsonl,
    operational_audit,
    percent,
    primary_prediction_path,
    runtime_messages,
    scenario_and_case_scores,
    sha256,
    validate_config,
    weighted_rate,
    write_json,
    write_jsonl,
)


def newline_terminated(path: Path) -> bool:
    if path.stat().st_size == 0:
        return True
    with path.open("rb") as handle:
        handle.seek(-1, 2)
        return handle.read(1) == b"\n"


def load_partial_runs(
    *,
    cases: list[dict[str, Any]],
    raw_root: Path,
    prefix: str,
    repetitions: int,
    shards: int,
    shard_size: int,
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    by_run: dict[int, list[dict[str, Any]]] = {}
    raw_audit: dict[str, Any] = {}
    case_order = {str(case["case_id"]): index for index, case in enumerate(cases)}
    for run in range(1, repetitions + 1):
        combined: dict[str, dict[str, Any]] = {}
        run_audit: list[dict[str, Any]] = []
        for shard in range(shards):
            path = primary_prediction_path(raw_root, prefix=prefix, shard=shard, run=run)
            expected_slice = cases[shard * shard_size : (shard + 1) * shard_size]
            expected_prefix_ids: list[str] = []
            if path.is_file():
                rows = load_jsonl(path)
                indexed = index_unique(rows, label=str(path))
                expected_prefix_ids = [
                    str(case["case_id"]) for case in expected_slice[: len(rows)]
                ]
                observed_ids = [str(row["case_id"]) for row in rows]
                if observed_ids != expected_prefix_ids:
                    raise ValueError(
                        f"run{run}/shard{shard} partial rows are not the exact fixed-slice prefix"
                    )
                overlap = set(combined).intersection(indexed)
                if overlap:
                    raise ValueError(f"run{run}: duplicated partial IDs: {sorted(overlap)[:3]}")
                combined.update(indexed)
                path_audit = {
                    "path": str(path.resolve()),
                    "rows": len(rows),
                    "sha256": sha256(path),
                    "newline_terminated": newline_terminated(path),
                    "valid_fixed_slice_prefix": True,
                }
            else:
                path_audit = {
                    "path": str(path.resolve()),
                    "rows": 0,
                    "sha256": "",
                    "newline_terminated": True,
                    "valid_fixed_slice_prefix": True,
                }
            run_audit.append({"shard": shard, **path_audit})
        by_run[run] = sorted(
            combined.values(), key=lambda row: case_order[str(row["case_id"])]
        )
        raw_audit[str(run)] = run_audit
    return by_run, raw_audit


def optimistic_full_rate(
    *,
    cases: list[dict[str, Any]],
    observed_cases: list[dict[str, Any]],
    observed_predictions: list[dict[str, Any]],
    metric_scenario: str,
    current_scenario: str,
    metric_weight: int,
    current_weight: int,
) -> tuple[float, dict[str, Any]]:
    total_by_scenario = Counter(
        str(case.get("scenario_id") or case.get("category") or "unknown")
        for case in cases
    )
    if observed_cases:
        observed_table, _ = scenario_and_case_scores(
            observed_cases, observed_predictions
        )
    else:
        observed_table = {}
    optimistic_table: dict[str, dict[str, Any]] = {}
    for scenario in (metric_scenario, current_scenario):
        total = int(total_by_scenario[scenario])
        observed = observed_table.get(
            scenario,
            {"cases": 0, "passed": 0, "failed": 0, "strict_pass_rate_percent": None},
        )
        observed_count = int(observed["cases"])
        observed_passed = int(observed["passed"])
        optimistic_passed = observed_passed + (total - observed_count)
        optimistic_table[scenario] = {
            "total_cases": total,
            "observed_cases": observed_count,
            "observed_passed": observed_passed,
            "unseen_cases_assumed_passed": total - observed_count,
            "optimistic_passed": optimistic_passed,
            "optimistic_strict_pass_rate_percent": percent(
                Fraction(optimistic_passed, total)
            ),
        }
    scorer_table = {
        scenario: {
            "cases": optimistic_table[scenario]["total_cases"],
            "passed": optimistic_table[scenario]["optimistic_passed"],
        }
        for scenario in (metric_scenario, current_scenario)
    }
    rate = weighted_rate(
        scorer_table,
        metric_scenario=metric_scenario,
        current_scenario=current_scenario,
        metric_weight=metric_weight,
        current_weight=current_weight,
    )
    return percent(rate), optimistic_table


def audit_early_stop(
    *,
    config_path: Path,
    checkpoint_path: Path,
    raw_root: Path,
    analysis_dir: Path,
    stop_reason: str,
) -> dict[str, Any]:
    if analysis_dir.exists():
        raise FileExistsError(f"refusing to overwrite analysis directory: {analysis_dir}")
    config = load_json(config_path)
    cases = validate_config(config, config_path=config_path)
    checkpoint = load_json(checkpoint_path)
    execution = config["execution"]
    runs, raw_audit = load_partial_runs(
        cases=cases,
        raw_root=raw_root,
        prefix=str(execution["report_prefix"]),
        repetitions=int(execution["repetitions"]),
        shards=int(execution["fixed_shards"]),
        shard_size=int(execution["cases_per_shard"]),
    )
    analysis_dir.mkdir(parents=True)
    for run, rows in runs.items():
        if rows:
            write_jsonl(analysis_dir / f"partial_run{run}_predictions.jsonl", rows)

    cases_by_id = {str(case["case_id"]): case for case in cases}
    observed_tables: dict[str, Any] = {}
    operations: dict[str, Any] = {}
    optimistic_rates: list[float] = []
    optimistic_tables: dict[str, Any] = {}
    scenario_spec = config["scenario_strata"]
    mixture = config["weighted_mixture"]
    metric_scenario = str(scenario_spec["metric_veto"])
    current_scenario = str(scenario_spec["current"])
    for run, predictions in runs.items():
        observed_cases = [cases_by_id[str(row["case_id"])] for row in predictions]
        operations[str(run)] = operational_audit(predictions)
        if predictions:
            scenario_table, _ = scenario_and_case_scores(observed_cases, predictions)
        else:
            scenario_table = {}
        observed_tables[str(run)] = {
            "status": "partial" if predictions else "not_started",
            "observed_rows": len(predictions),
            "scenario_table": scenario_table,
            "weighted_rate_not_reported": True,
            "reason": "incomplete repetition; no denominator substitution",
        }
        optimistic_rate, optimistic_table = optimistic_full_rate(
            cases=cases,
            observed_cases=observed_cases,
            observed_predictions=predictions,
            metric_scenario=metric_scenario,
            current_scenario=current_scenario,
            metric_weight=int(mixture["metric_veto_weight"]),
            current_weight=int(mixture["current_weight"]),
        )
        optimistic_rates.append(optimistic_rate)
        optimistic_tables[str(run)] = {
            "optimistic_weighted_rate_percent": optimistic_rate,
            "assumption": "every unseen case passes",
            "scenario_table": optimistic_table,
        }

    checkpoint_ids = [str(item) for item in checkpoint["case_ids"]]
    run_index = index_unique(runs[int(checkpoint["run"])], label="partial checkpoint run")
    if any(case_id not in run_index for case_id in checkpoint_ids):
        raise ValueError("falsification checkpoint case is absent from preserved partial rows")
    checkpoint_predictions = [run_index[case_id] for case_id in checkpoint_ids]
    checkpoint_cases = [cases_by_id[case_id] for case_id in checkpoint_ids]
    checkpoint_ops = operational_audit(checkpoint_predictions)
    _, checkpoint_passes = scenario_and_case_scores(
        checkpoint_cases, checkpoint_predictions
    )
    checkpoint_runtime_signatures = Counter(
        message
        for prediction in checkpoint_predictions
        for message in runtime_messages(prediction)
    )
    if checkpoint_ops["runtime_error_cases"] != int(
        checkpoint["expected_runtime_error_rows"]
    ):
        raise ValueError("falsification checkpoint runtime count mismatch")
    if checkpoint_ops["transport_failure_cases"] != int(
        checkpoint["expected_transport_failure_rows"]
    ):
        raise ValueError("falsification checkpoint transport count mismatch")
    if sum(checkpoint_passes.values()) != int(checkpoint["expected_strict_passes"]):
        raise ValueError("falsification checkpoint strict-pass count mismatch")

    gates = config["hard_gates"]
    observed_runtime_total = sum(
        int(item["runtime_error_cases"]) for item in operations.values()
    )
    hard_success_possible = observed_runtime_total <= int(
        gates["maximum_runtime_error_cases"]
    )
    # This is a valid loose upper bound on the declared scalar score: each run
    # rate cannot exceed its all-unseen-pass rate; range and agreement terms
    # cannot exceed +2pp and +5pp respectively.
    dev_score_upper_components = {
        **{
            f"run{run}_optimistic_minus_85": round(optimistic_rates[run - 1] - 85.0, 6)
            for run in (1, 2, 3)
        },
        "maximum_possible_two_minus_range": 2.0,
        "maximum_possible_agreement_minus_95": 5.0,
    }
    dev_score_upper_bound = min(dev_score_upper_components.values())
    all_newline = all(
        bool(item["newline_terminated"])
        for run_audit in raw_audit.values()
        for item in run_audit
    )

    report = {
        "schema_version": "1.0",
        "study_id": config["study_id"],
        "status": "early_stopped_irreversible_failure",
        "hard_success": False,
        "hard_success_possible_after_stop": hard_success_possible,
        "hard_success_upper_bound": 1 if hard_success_possible else 0,
        "stop_reason": stop_reason,
        "stopped_on_complete_jsonl_newlines": all_newline,
        "held_out_or_new_test_used": False,
        "retry_used": False,
        "retry_authorized": False,
        "retry_reason": "observed failures are non-transport",
        "inputs": {
            "config": {"path": str(config_path.resolve()), "sha256": sha256(config_path)},
            "cases": {
                "path": str(Path(config["cases"]["path"]).resolve()),
                "rows": len(cases),
                "sha256": sha256(Path(config["cases"]["path"])),
            },
            "checkpoint": {
                "path": str(checkpoint_path.resolve()),
                "sha256": sha256(checkpoint_path),
            },
        },
        "raw_partial_audit": raw_audit,
        "partial_run_tables": observed_tables,
        "operations_by_run": operations,
        "falsification_checkpoint": {
            "observed_rows": len(checkpoint_predictions),
            "runtime_error_rows": checkpoint_ops["runtime_error_cases"],
            "transport_failure_rows": checkpoint_ops["transport_failure_cases"],
            "strict_passes": sum(checkpoint_passes.values()),
            "strict_failures": len(checkpoint_passes) - sum(checkpoint_passes.values()),
            "case_ids": checkpoint_ids,
            "runtime_error_signatures": dict(checkpoint_runtime_signatures),
            "hard_gate_irreversibly_failed": True,
        },
        "optimistic_completion_upper_bound": {
            "tables_by_run": optimistic_tables,
            "standard_dev_score_computed": False,
            "dev_score_upper_bound_components_pp": dev_score_upper_components,
            "dev_score_upper_bound_pp": round(dev_score_upper_bound, 6),
            "is_nonpositive": dev_score_upper_bound <= 0.0,
        },
        "raw_retry_accounting": {
            "expected_primary_rows": 288,
            "observed_primary_rows": sum(len(rows) for rows in runs.values()),
            "unobserved_primary_rows": 288 - sum(len(rows) for rows in runs.values()),
            "retry_rows": 0,
            "silent_expected_decision_fill_used": False,
        },
    }
    report_path = analysis_dir / "early_stop_report.json"
    write_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--stop-reason", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_early_stop(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        raw_root=args.raw_root,
        analysis_dir=args.analysis_dir,
        stop_reason=args.stop_reason,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "observed_primary_rows": report["raw_retry_accounting"][
                    "observed_primary_rows"
                ],
                "checkpoint_runtime_failures": report["falsification_checkpoint"][
                    "runtime_error_rows"
                ],
                "dev_score_upper_bound_pp": report[
                    "optimistic_completion_upper_bound"
                ]["dev_score_upper_bound_pp"],
                "report": str((args.analysis_dir / "early_stop_report.json").resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
