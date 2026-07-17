#!/usr/bin/env python3
"""Combine and audit the three-run n10 Qwen3.5-35B stability experiment.

This evaluator is intentionally specific to the frozen, already-open n8 scene.
It never substitutes expected decisions.  An optional retry overlay is accepted
only when it exactly matches the transport-failure IDs emitted by a prior
primary evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.planner_grpo_seed_v1.scripts.reward_planner_grpo import (  # noqa: E402
    score_case,
)


TRANSPORT_MARKERS = (
    "timeout",
    "timed out",
    "connection",
    "transport",
    "http",
    "proxy",
    "gateway",
    "remoteprotocol",
    "readerror",
    "reset by peer",
    "unreachable",
    "api error",
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: expected a JSON object")
        rows.append(value)
    return rows


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percent(value: Fraction | float) -> float:
    return round(float(value) * 100.0, 6)


def index_unique(
    rows: list[dict[str, Any]], *, label: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id") or "").strip()
        if not case_id:
            raise ValueError(f"{label}: missing case_id")
        if case_id in indexed:
            raise ValueError(f"{label}: duplicate case_id: {case_id}")
        indexed[case_id] = row
    return indexed


def validate_config(config: dict[str, Any], *, config_path: Path) -> list[dict[str, Any]]:
    if config.get("held_out_or_new_test_allowed") is not False:
        raise ValueError("configuration must explicitly forbid held-out/new-test use")
    case_spec = config.get("cases")
    if not isinstance(case_spec, dict):
        raise ValueError("cases must be an object")
    cases_path = Path(str(case_spec["path"]))
    if sha256(cases_path) != str(case_spec["sha256"]):
        raise ValueError("frozen case SHA-256 mismatch")
    cases = load_jsonl(cases_path)
    if len(cases) != int(case_spec["rows"]):
        raise ValueError("frozen case row-count mismatch")
    index_unique(cases, label=str(cases_path))

    execution = config.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("execution must be an object")
    repetitions = int(execution["repetitions"])
    shards = int(execution["fixed_shards"])
    shard_size = int(execution["cases_per_shard"])
    if repetitions != 3 or shards != 4 or shard_size != 24:
        raise ValueError("n10 requires exactly 3 repetitions and 4x24 fixed shards")
    if shards * shard_size != len(cases):
        raise ValueError("fixed shard geometry does not cover frozen cases")

    protocol = config.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be an object")
    required_protocol = {
        "temperature": 0.0,
        "top_p": 1.0,
        "do_sample": False,
        "seed": 42,
        "max_steps": 3,
        "max_tokens": 320,
        "timeout_seconds": 300,
        "openai_timeout_seconds": 300,
        "omit_model_image_payload": True,
    }
    mismatches = {
        key: (protocol.get(key), expected)
        for key, expected in required_protocol.items()
        if protocol.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"n10 protocol mismatch: {mismatches}")
    if str(config.get("model", {}).get("id")) != "Qwen3.5-35B-A3B":
        raise ValueError("n10 model must be Qwen3.5-35B-A3B")
    if not config_path.is_file():
        raise ValueError("configuration path disappeared during validation")
    return cases


def primary_prediction_path(
    raw_root: Path, *, prefix: str, shard: int, run: int
) -> Path:
    return (
        raw_root
        / f"shard{shard}"
        / f"{prefix}_shard{shard}_run{run}_predictions.jsonl"
    )


def combine_primary_run(
    *,
    cases: list[dict[str, Any]],
    raw_root: Path,
    prefix: str,
    run: int,
    shards: int,
    shard_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    combined: dict[str, dict[str, Any]] = {}
    shard_audit: list[dict[str, Any]] = []
    for shard in range(shards):
        path = primary_prediction_path(raw_root, prefix=prefix, shard=shard, run=run)
        rows = load_jsonl(path)
        indexed = index_unique(rows, label=str(path))
        expected_slice = cases[shard * shard_size : (shard + 1) * shard_size]
        expected_ids = {str(row["case_id"]) for row in expected_slice}
        observed_ids = set(indexed)
        if observed_ids != expected_ids:
            raise ValueError(
                f"run{run}/shard{shard} coverage mismatch: "
                f"missing={sorted(expected_ids - observed_ids)[:3]}, "
                f"extra={sorted(observed_ids - expected_ids)[:3]}"
            )
        overlap = set(combined).intersection(indexed)
        if overlap:
            raise ValueError(f"run{run}: shard overlap: {sorted(overlap)[:3]}")
        combined.update(indexed)
        shard_audit.append(
            {
                "shard": shard,
                "offset": shard * shard_size,
                "rows": len(rows),
                "path": str(path.resolve()),
                "sha256": sha256(path),
            }
        )
    ordered = [combined[str(case["case_id"])] for case in cases]
    return ordered, shard_audit


def runtime_messages(prediction: dict[str, Any]) -> list[str]:
    messages = [
        str(item)
        for item in prediction.get("errors", [])
        if str(item).strip()
    ] if isinstance(prediction.get("errors"), list) else []
    decisions = prediction.get("decisions")
    if isinstance(decisions, list):
        for decision in decisions:
            if not isinstance(decision, dict):
                continue
            metrics = decision.get("_planner_metrics")
            if not isinstance(metrics, dict):
                continue
            error_type = str(metrics.get("error_type") or "").strip()
            error = str(metrics.get("error") or "").strip()
            if error_type or error:
                messages.append(f"{error_type}: {error}".strip())
            retry_error_type = str(metrics.get("retry_error_type") or "").strip()
            retry_error = str(metrics.get("retry_error") or "").strip()
            if retry_error_type or retry_error:
                messages.append(f"{retry_error_type}: {retry_error}".strip())
    return messages


def is_transport_failure(prediction: dict[str, Any]) -> bool:
    messages = runtime_messages(prediction)
    lowered = "\n".join(messages).lower()
    return bool(messages) and any(marker in lowered for marker in TRANSPORT_MARKERS)


def operational_audit(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    empty_ids: list[str] = []
    runtime_ids: list[str] = []
    clipped_ids: list[str] = []
    transport_ids: list[str] = []
    planner_decisions = 0
    for prediction in predictions:
        case_id = str(prediction["case_id"])
        decisions = prediction.get("decisions")
        if not isinstance(decisions, list) or not decisions:
            empty_ids.append(case_id)
            decisions = []
        planner_decisions += len(decisions)
        if runtime_messages(prediction):
            runtime_ids.append(case_id)
        if is_transport_failure(prediction):
            transport_ids.append(case_id)
        if any(
            isinstance(decision, dict)
            and isinstance(decision.get("_planner_metrics"), dict)
            and (
                str(decision["_planner_metrics"].get("first_finish_reason") or "")
                == "length"
                or str(
                    decision["_planner_metrics"].get("retry_finish_reason") or ""
                )
                == "length"
            )
            for decision in decisions
        ):
            clipped_ids.append(case_id)
    return {
        "prediction_rows": len(predictions),
        "planner_decisions": planner_decisions,
        "empty_decision_cases": len(empty_ids),
        "runtime_error_cases": len(runtime_ids),
        "clipped_cases": len(clipped_ids),
        "transport_failure_cases": len(transport_ids),
        "empty_decision_case_ids": empty_ids,
        "runtime_error_case_ids": runtime_ids,
        "clipped_case_ids": clipped_ids,
        "transport_failure_case_ids": transport_ids,
    }


def load_retry_overlay(
    *,
    retry_root: Path,
    retry_plan: dict[str, Any],
    run: int,
    prefix: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    run_plan = retry_plan["runs"][str(run)]
    expected_ids = set(str(item) for item in run_plan["transport_failure_case_ids"])
    if not expected_ids:
        return {}, {"attempted": 0, "path": "", "sha256": ""}
    path = retry_root / f"run{run}" / f"{prefix}_retry_run{run}_run1_predictions.jsonl"
    rows = load_jsonl(path)
    indexed = index_unique(rows, label=str(path))
    if set(indexed) != expected_ids:
        raise ValueError(
            f"run{run} retry coverage is not exactly the predeclared transport failures"
        )
    return indexed, {
        "attempted": len(rows),
        "path": str(path.resolve()),
        "sha256": sha256(path),
    }


def scenario_and_case_scores(
    cases: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, bool]]:
    indexed = index_unique(predictions, label="combined predictions")
    by_scenario: dict[str, list[bool]] = {}
    by_case: dict[str, bool] = {}
    for case in cases:
        case_id = str(case["case_id"])
        result = score_case(case, indexed[case_id], use_expected_when_missing=False)
        passed = result.get("passed") is True
        scenario = str(case.get("scenario_id") or case.get("category") or "unknown")
        by_scenario.setdefault(scenario, []).append(passed)
        by_case[case_id] = passed
    table = {
        scenario: {
            "cases": len(values),
            "passed": sum(values),
            "failed": len(values) - sum(values),
            "strict_pass_rate_percent": percent(Fraction(sum(values), len(values))),
        }
        for scenario, values in sorted(by_scenario.items())
    }
    return table, by_case


def weighted_rate(
    scenario_table: dict[str, dict[str, Any]],
    *,
    metric_scenario: str,
    current_scenario: str,
    metric_weight: int,
    current_weight: int,
) -> Fraction:
    metric = scenario_table[metric_scenario]
    current = scenario_table[current_scenario]
    metric_rate = Fraction(int(metric["passed"]), int(metric["cases"]))
    current_rate = Fraction(int(current["passed"]), int(current["cases"]))
    return (
        metric_weight * metric_rate + current_weight * current_rate
    ) / (metric_weight + current_weight)


def exact_decision_signature(prediction: dict[str, Any]) -> str:
    decisions = prediction.get("decisions")
    if not isinstance(decisions, list):
        decisions = []
    normalized = []
    for decision in decisions:
        if not isinstance(decision, dict):
            normalized.append({"invalid": decision})
            continue
        normalized.append(
            {
                key: decision.get(key)
                for key in (
                    "decision_type",
                    "action",
                    "action_input",
                    "end_reason",
                    "final_answer",
                    "clarification_question",
                )
            }
        )
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def action_signature(prediction: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    decisions = prediction.get("decisions")
    if not isinstance(decisions, list):
        return tuple()
    return tuple(
        (
            str(item.get("decision_type") or "") if isinstance(item, dict) else "",
            str(item.get("action") or "") if isinstance(item, dict) else "",
        )
        for item in decisions
    )


def pairwise_agreement(
    *,
    cases: list[dict[str, Any]],
    predictions_by_run: dict[int, list[dict[str, Any]]],
    passes_by_run: dict[int, dict[str, bool]],
) -> list[dict[str, Any]]:
    case_ids = [str(case["case_id"]) for case in cases]
    indexed = {
        run: index_unique(rows, label=f"run{run}")
        for run, rows in predictions_by_run.items()
    }
    pairs: list[dict[str, Any]] = []
    for left, right in ((1, 2), (1, 3), (2, 3)):
        strict_equal = sum(
            passes_by_run[left][case_id] == passes_by_run[right][case_id]
            for case_id in case_ids
        )
        exact_equal = sum(
            exact_decision_signature(indexed[left][case_id])
            == exact_decision_signature(indexed[right][case_id])
            for case_id in case_ids
        )
        action_equal = sum(
            action_signature(indexed[left][case_id])
            == action_signature(indexed[right][case_id])
            for case_id in case_ids
        )
        pairs.append(
            {
                "runs": f"{left}-{right}",
                "cases": len(case_ids),
                "strict_pass_fail_equal": strict_equal,
                "strict_pass_fail_agreement_percent": percent(
                    Fraction(strict_equal, len(case_ids))
                ),
                "exact_decision_equal": exact_equal,
                "exact_decision_agreement_percent": percent(
                    Fraction(exact_equal, len(case_ids))
                ),
                "action_sequence_equal": action_equal,
                "action_sequence_agreement_percent": percent(
                    Fraction(action_equal, len(case_ids))
                ),
            }
        )
    return pairs


def evaluate(
    *,
    config_path: Path,
    raw_root: Path,
    analysis_dir: Path,
    retry_root: Path | None = None,
    retry_plan_path: Path | None = None,
) -> dict[str, Any]:
    if analysis_dir.exists():
        raise FileExistsError(f"refusing to overwrite analysis directory: {analysis_dir}")
    config = load_json(config_path)
    cases = validate_config(config, config_path=config_path)
    case_spec = config["cases"]
    execution = config["execution"]
    prefix = str(execution["report_prefix"])
    repetitions = int(execution["repetitions"])
    shards = int(execution["fixed_shards"])
    shard_size = int(execution["cases_per_shard"])

    retry_plan = None
    if retry_root is not None:
        if retry_plan_path is None:
            raise ValueError("--retry-root requires --retry-plan")
        retry_plan = load_json(retry_plan_path)
        if retry_plan.get("maximum_retry_attempts") != 1:
            raise ValueError("retry plan must predeclare exactly one maximum attempt")
        if retry_plan.get("cases_sha256") != case_spec["sha256"]:
            raise ValueError("retry plan case hash mismatch")

    primary_by_run: dict[int, list[dict[str, Any]]] = {}
    final_by_run: dict[int, list[dict[str, Any]]] = {}
    shard_audit_by_run: dict[str, Any] = {}
    retry_audit_by_run: dict[str, Any] = {}
    initial_transport_ids: dict[int, list[str]] = {}
    for run in range(1, repetitions + 1):
        primary, shard_audit = combine_primary_run(
            cases=cases,
            raw_root=raw_root,
            prefix=prefix,
            run=run,
            shards=shards,
            shard_size=shard_size,
        )
        primary_by_run[run] = primary
        primary_ops = operational_audit(primary)
        initial_transport_ids[run] = list(primary_ops["transport_failure_case_ids"])
        final_rows = list(primary)
        retry_audit: dict[str, Any] = {"attempted": 0, "path": "", "sha256": ""}
        if retry_plan is not None and retry_root is not None:
            declared_ids = list(
                retry_plan["runs"][str(run)]["transport_failure_case_ids"]
            )
            if declared_ids != initial_transport_ids[run]:
                raise ValueError(f"run{run} retry-plan IDs differ from raw transport failures")
            overlay, retry_audit = load_retry_overlay(
                retry_root=retry_root,
                retry_plan=retry_plan,
                run=run,
                prefix=prefix,
            )
            final_rows = [overlay.get(str(row["case_id"]), row) for row in primary]
        final_by_run[run] = final_rows
        shard_audit_by_run[str(run)] = shard_audit
        retry_audit_by_run[str(run)] = retry_audit

    analysis_dir.mkdir(parents=True)
    combined_hashes: dict[str, Any] = {}
    for run, rows in final_by_run.items():
        path = analysis_dir / f"combined_run{run}_predictions.jsonl"
        write_jsonl(path, rows)
        combined_hashes[str(run)] = {
            "path": str(path.resolve()),
            "rows": len(rows),
            "sha256": sha256(path),
        }

    # Emit an immutable, exact retry plan from primary artifacts.  The runner
    # may consume it once; an empty plan means no retry is authorized or needed.
    retry_plan_output = {
        "schema_version": "1.0",
        "study_id": config["study_id"],
        "cases_path": str(Path(case_spec["path"]).resolve()),
        "cases_sha256": case_spec["sha256"],
        "maximum_retry_attempts": 1,
        "same_protocol_required": True,
        "only_transport_failure_ids": True,
        "runs": {},
    }
    case_by_id = {str(case["case_id"]): case for case in cases}
    for run in range(1, repetitions + 1):
        retry_ids = initial_transport_ids[run]
        retry_cases_path = analysis_dir / f"retry_cases_run{run}.jsonl"
        if retry_ids:
            retry_rows = [case_by_id[case_id] for case_id in retry_ids]
            write_jsonl(retry_cases_path, retry_rows)
            retry_cases_hash = sha256(retry_cases_path)
        else:
            retry_cases_hash = ""
        retry_plan_output["runs"][str(run)] = {
            "transport_failure_case_ids": retry_ids,
            "retry_cases_path": str(retry_cases_path.resolve()) if retry_ids else "",
            "retry_cases_sha256": retry_cases_hash,
            "primary_combined_sha256": sha256(
                analysis_dir / f"combined_run{run}_predictions.jsonl"
            ) if retry_plan is None else "see_input_retry_plan",
        }
    retry_plan_output_path = analysis_dir / "transport_retry_plan.json"
    write_json(retry_plan_output_path, retry_plan_output)

    scenario_spec = config["scenario_strata"]
    mixture_spec = config["weighted_mixture"]
    metric_scenario = str(scenario_spec["metric_veto"])
    current_scenario = str(scenario_spec["current"])
    metric_weight = int(mixture_spec["metric_veto_weight"])
    current_weight = int(mixture_spec["current_weight"])
    run_tables: list[dict[str, Any]] = []
    passes_by_run: dict[int, dict[str, bool]] = {}
    rates: list[float] = []
    final_ops: dict[int, dict[str, Any]] = {}
    primary_ops_by_run: dict[int, dict[str, Any]] = {}
    for run in range(1, repetitions + 1):
        primary_ops_by_run[run] = operational_audit(primary_by_run[run])
        final_ops[run] = operational_audit(final_by_run[run])
        scenarios, by_case = scenario_and_case_scores(cases, final_by_run[run])
        for scenario in (metric_scenario, current_scenario):
            if scenario not in scenarios:
                raise ValueError(f"run{run}: missing scenario {scenario}")
        rate = weighted_rate(
            scenarios,
            metric_scenario=metric_scenario,
            current_scenario=current_scenario,
            metric_weight=metric_weight,
            current_weight=current_weight,
        )
        rate_pct = percent(rate)
        rates.append(rate_pct)
        passes_by_run[run] = by_case
        run_tables.append(
            {
                "run": run,
                "scenario_table": scenarios,
                "weighted_mixture": {
                    "label": mixture_spec["label"],
                    "strict_pass_rate_percent": rate_pct,
                    "margin_above_85_pp": round(rate_pct - 85.0, 6),
                },
                "operations": final_ops[run],
                "artifact": combined_hashes[str(run)],
            }
        )

    pairwise = pairwise_agreement(
        cases=cases,
        predictions_by_run=final_by_run,
        passes_by_run=passes_by_run,
    )
    minimum_agreement = min(
        float(item["strict_pass_fail_agreement_percent"]) for item in pairwise
    )
    rate_range = max(rates) - min(rates)
    gates = config["hard_gates"]
    operational_pass = all(
        int(final_ops[run]["prediction_rows"])
        == int(gates["required_rows_per_repetition"])
        and int(final_ops[run]["empty_decision_cases"])
        <= int(gates["maximum_empty_decision_cases"])
        and int(final_ops[run]["runtime_error_cases"])
        <= int(gates["maximum_runtime_error_cases"])
        and int(final_ops[run]["clipped_cases"])
        <= int(gates["maximum_clipped_cases"])
        for run in range(1, repetitions + 1)
    )
    checks = {
        "exact_96_rows_each_run": all(len(final_by_run[run]) == 96 for run in final_by_run),
        "no_empty_decision_runtime_error_or_clipping": operational_pass,
        "each_weighted_rate_strictly_above_85": all(
            rate > float(gates["minimum_weighted_rate_percent_exclusive"])
            for rate in rates
        ),
        "repetition_range_at_most_2pp": rate_range
        <= float(gates["maximum_repetition_range_pp_inclusive"]),
        "minimum_pairwise_strict_pass_fail_agreement_at_least_95": minimum_agreement
        >= float(
            gates["minimum_pairwise_strict_pass_fail_agreement_percent_inclusive"]
        ),
    }
    dev_components = {
        **{f"run{run}_minus_85": round(rates[run - 1] - 85.0, 6) for run in range(1, 4)},
        "two_minus_range": round(2.0 - rate_range, 6),
        "agreement_minus_95": round(minimum_agreement - 95.0, 6),
    }
    dev_score = min(dev_components.values())

    report = {
        "schema_version": "1.0",
        "study_id": config["study_id"],
        "status": "success" if all(checks.values()) else "failure",
        "hard_success": all(checks.values()),
        "evidence_role": config["evidence_role"],
        "held_out_or_new_test_used": False,
        "model": config["model"],
        "protocol": config["protocol"],
        "coverage_and_hashes": {
            "config": {
                "path": str(config_path.resolve()),
                "sha256": sha256(config_path),
            },
            "cases": {
                "path": str(Path(case_spec["path"]).resolve()),
                "rows": len(cases),
                "sha256": sha256(Path(case_spec["path"])),
            },
            "raw_shards_by_run": shard_audit_by_run,
            "combined_by_run": combined_hashes,
        },
        "run_tables": run_tables,
        "stability": {
            "weighted_rates_percent": rates,
            "max_minus_min_pp": round(rate_range, 6),
            "pairwise_agreement": pairwise,
            "minimum_pairwise_strict_pass_fail_agreement_percent": minimum_agreement,
            "exact_decision_definition": "decision_type/action/action_input/end_reason/final_answer/clarification_question; excludes thought and telemetry",
            "action_sequence_definition": "ordered (decision_type, action) pairs",
        },
        "raw_retry_accounting": {
            "primary_prediction_rows": sum(
                len(primary_by_run[run]) for run in primary_by_run
            ),
            "primary_operations_by_run": {
                str(run): primary_ops_by_run[run] for run in primary_ops_by_run
            },
            "retry_used": retry_plan is not None,
            "retry_by_run": retry_audit_by_run,
            "retry_prediction_rows": sum(
                int(item["attempted"]) for item in retry_audit_by_run.values()
            ),
            "final_operations_by_run": {
                str(run): final_ops[run] for run in final_ops
            },
            "retry_plan": {
                "path": str(retry_plan_output_path.resolve()),
                "sha256": sha256(retry_plan_output_path),
            },
            "silent_expected_decision_fill_used": False,
        },
        "hard_gate_checks": checks,
        "dev_score": {
            "definition": config["dev_score_pp"],
            "components_pp": dev_components,
            "score_pp": round(dev_score, 6),
        },
    }
    report_path = analysis_dir / "stability_report.json"
    write_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--retry-root", type=Path)
    parser.add_argument("--retry-plan", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate(
        config_path=args.config,
        raw_root=args.raw_root,
        analysis_dir=args.analysis_dir,
        retry_root=args.retry_root,
        retry_plan_path=args.retry_plan,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "weighted_rates_percent": report["stability"][
                    "weighted_rates_percent"
                ],
                "range_pp": report["stability"]["max_minus_min_pp"],
                "minimum_pairwise_strict_agreement_percent": report["stability"][
                    "minimum_pairwise_strict_pass_fail_agreement_percent"
                ],
                "dev_score_pp": report["dev_score"]["score_pp"],
                "report": str((args.analysis_dir / "stability_report.json").resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
