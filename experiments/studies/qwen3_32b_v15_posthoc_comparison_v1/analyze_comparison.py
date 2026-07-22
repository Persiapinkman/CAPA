#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[3]
LABELS = {
    "qwen35_4b_base": "Qwen3.5-4B Base",
    "qwen35_4b_sft": "Qwen3.5-4B original SFT",
    "qwen35_35b_a3b": "Qwen3.5-35B-A3B",
    "qwen35_4b_grpo_n64": "Qwen3.5-4B targeted-SFT + one-step GRPO",
    "qwen3_32b": "Qwen3-32B local FP16 TP4 (post-hoc)",
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()

    config = load_json(args.config)
    immutable = config["immutable_inputs"]
    cases_path = resolve(immutable["cases_path"])
    reference_path = resolve(immutable["v15_final_report_path"])
    if sha256(cases_path) != immutable["cases_sha256"]:
        raise ValueError("V15 cases hash mismatch")
    if sha256(reference_path) != immutable["v15_final_report_sha256"]:
        raise ValueError("V15 final report hash mismatch")

    cases = load_jsonl(cases_path)
    expected_ids = [str(row["case_id"]) for row in cases]
    expected_set = set(expected_ids)
    expected_categories = Counter(str(row["category"]) for row in cases)
    if len(cases) != 24 or len(expected_set) != 24:
        raise ValueError("unexpected V15 case geometry")

    run_rows = []
    coverage_errors: list[str] = []
    runtime_error_cases = 0
    length_finishes = 0
    parse_retry_count = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_api_call_ms = 0.0
    for run in (1, 2, 3):
        pred_path = args.output_root / f"qwen3_32b_run{run}_predictions.jsonl"
        reward_path = args.output_root / f"qwen3_32b_run{run}_reward.json"
        predictions = load_jsonl(pred_path)
        rewards = load_json(reward_path)
        pred_ids = [str(row.get("case_id") or "") for row in predictions]
        reward_results = rewards.get("results") if isinstance(rewards.get("results"), list) else []
        reward_ids = [str(row.get("case_id") or "") for row in reward_results]
        if len(predictions) != 24 or set(pred_ids) != expected_set or len(set(pred_ids)) != 24:
            coverage_errors.append(f"run{run}: prediction coverage mismatch")
        if len(reward_results) != 24 or set(reward_ids) != expected_set or len(set(reward_ids)) != 24:
            coverage_errors.append(f"run{run}: reward coverage mismatch")

        prediction_errors_this_run = 0
        for row in predictions:
            row_failed = bool(row.get("errors"))
            for decision in row.get("decisions") or []:
                metrics = decision.get("_planner_metrics") or {}
                if metrics.get("error_type") or metrics.get("first_error_type") or metrics.get("retry_error_type"):
                    row_failed = True
                parse_retry_count += int(metrics.get("retry_count") or 0)
                for prefix in ("first_", "retry_"):
                    finish = metrics.get(f"{prefix}finish_reason")
                    if finish == "length":
                        length_finishes += 1
                    total_input_tokens += int(metrics.get(f"{prefix}input_tokens") or 0)
                    total_output_tokens += int(metrics.get(f"{prefix}output_tokens") or 0)
                total_api_call_ms += float(metrics.get("api_call_ms") or 0.0)
            prediction_errors_this_run += int(row_failed)
        runtime_error_cases += prediction_errors_this_run

        passed_by_category = Counter()
        for result in reward_results:
            if result.get("passed") is True:
                passed_by_category[str(result.get("category") or "")] += 1
        metric_pass = passed_by_category["post_retry_metric_veto_step3"]
        success_pass = passed_by_category["current_success_step2"]
        weighted = 100.0 * (
            111.0 * metric_pass / expected_categories["post_retry_metric_veto_step3"]
            + 14.0 * success_pass / expected_categories["current_success_step2"]
        ) / 125.0
        run_rows.append(
            {
                "run": run,
                "scenario_passed": {
                    "current_success_step2": success_pass,
                    "post_retry_metric_veto_step3": metric_pass,
                },
                "strict_passed": sum(passed_by_category.values()),
                "strict_pass_rate_percent": 100.0 * sum(passed_by_category.values()) / 24.0,
                "weighted_strict_percent": weighted,
                "prediction_error_cases": prediction_errors_this_run,
            }
        )

    scores = [row["weighted_strict_percent"] for row in run_rows]
    qwen32 = {
        "arm": "qwen3_32b",
        "mean_weighted_strict_percent": mean(scores),
        "per_run_scenario_passed": run_rows,
        "run1_weighted_strict_percent": scores[0],
        "run2_weighted_strict_percent": scores[1],
        "run3_weighted_strict_percent": scores[2],
        "run_range_pp": max(scores) - min(scores),
    }

    reference = load_json(reference_path)
    original_table = reference["table"]
    combined = [dict(row) for row in original_table] + [qwen32]
    sorted_table = sorted(combined, key=lambda row: row["mean_weighted_strict_percent"])
    deltas = {
        row["arm"]: qwen32["mean_weighted_strict_percent"] - row["mean_weighted_strict_percent"]
        for row in original_table
    }
    valid = not coverage_errors and runtime_error_cases == 0 and length_finishes == 0
    report = {
        "schema_version": "1.0",
        "study_id": config["study_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "valid" if valid else "invalid",
        "evidence_role": config["evidence_role"],
        "qualification": "Qwen3-32B is a post-hoc read-only extension, not an original preregistered V15 arm.",
        "input_verification": {
            "cases_sha256": sha256(cases_path),
            "v15_final_report_sha256": sha256(reference_path),
            "coverage_errors": coverage_errors,
        },
        "runtime_validity": {
            "top_level_predictions": 72,
            "runtime_error_cases": runtime_error_cases,
            "length_finish_count": length_finishes,
            "parse_retry_count": parse_retry_count,
        },
        "usage": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "api_call_ms_sum": round(total_api_call_ms, 3),
        },
        "qwen3_32b": qwen32,
        "delta_32b_minus_original_arm_mean_pp": deltas,
        "combined_table": combined,
        "sorted_by_mean_ascending": [row["arm"] for row in sorted_table],
        "weighted_formula": config["analysis"]["weighted_formula"],
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Qwen3-32B on V15 comparison",
        "",
        f"Status: **{report['status']}**. Qwen3-32B is a post-hoc read-only extension and does not alter the original V15 confirmation.",
        "",
        "| Model | Run 1 (%) | Run 2 (%) | Run 3 (%) | Mean (%) | Range (pp) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted_table:
        lines.append(
            f"| {LABELS.get(row['arm'], row['arm'])} | "
            f"{row['run1_weighted_strict_percent']:.4f} | "
            f"{row['run2_weighted_strict_percent']:.4f} | "
            f"{row['run3_weighted_strict_percent']:.4f} | "
            f"{row['mean_weighted_strict_percent']:.4f} | "
            f"{row['run_range_pp']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Qwen3-32B scenario counts",
            "",
            "| Run | Current-success passed | Metric-veto passed | Strict passed | Weighted (%) |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in run_rows:
        lines.append(
            f"| {row['run']} | {row['scenario_passed']['current_success_step2']}/12 | "
            f"{row['scenario_passed']['post_retry_metric_veto_step3']}/12 | "
            f"{row['strict_passed']}/24 | {row['weighted_strict_percent']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Runtime errors: {runtime_error_cases}; length finishes: {length_finishes}; parse retries: {parse_retry_count}.",
            "",
            f"Weighted formula: `{config['analysis']['weighted_formula']}`.",
        ]
    )
    args.markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "qwen3_32b": qwen32, "deltas": deltas}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
