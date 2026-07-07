#!/usr/bin/env python3
"""Build side-by-side GRPO model comparison artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = ROOT / "training" / "planner_grpo_seed_v1" / "reports" / "repro_eval"
DEFAULT_CASES = ROOT / "training" / "planner_grpo_seed_v1" / "cases" / "planner_grpo_train_cases.jsonl"

MODEL_SPECS = [
    {
        "label": "Qwen3.5-4B",
        "short": "4B",
        "prefix": "qwen35_4b_grpo_compound245_stateprompt_t60_3x",
    },
    {
        "label": "Qwen3.5-9B",
        "short": "9B",
        "prefix": "qwen35_9b_grpo_compound245_stateprompt_t60_3x",
    },
    {
        "label": "Qwen3.5-35B-A3B",
        "short": "35B-A3B",
        "prefix": "qwen35_35b_a3b_grpo_compound245_stateprompt_t60_3x",
    },
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _compact(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _aggregate_path(out_dir: Path, prefix: str) -> Path:
    return out_dir / f"{prefix}_aggregate.json"


def _reward_path(out_dir: Path, prefix: str, run_idx: int) -> Path:
    return out_dir / f"{prefix}_run{run_idx}_reward.json"


def _prediction_path(out_dir: Path, prefix: str, run_idx: int) -> Path:
    return out_dir / f"{prefix}_run{run_idx}_predictions.jsonl"


def _load_cases_by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("case_id") or ""): row
        for row in load_jsonl(path)
        if isinstance(row, dict) and str(row.get("case_id") or "")
    }


def _load_reward_maps(out_dir: Path, prefix: str, runs: int) -> list[dict[str, dict[str, Any]]]:
    maps: list[dict[str, dict[str, Any]]] = []
    for run_idx in range(1, runs + 1):
        path = _reward_path(out_dir, prefix, run_idx)
        data = load_json(path)
        rows = data.get("results") if isinstance(data.get("results"), list) else []
        maps.append(
            {
                str(row.get("case_id") or ""): row
                for row in rows
                if isinstance(row, dict) and str(row.get("case_id") or "")
            }
        )
    return maps


def _load_prediction_maps(out_dir: Path, prefix: str, runs: int) -> list[dict[str, dict[str, Any]]]:
    maps: list[dict[str, dict[str, Any]]] = []
    for run_idx in range(1, runs + 1):
        path = _prediction_path(out_dir, prefix, run_idx)
        maps.append(
            {
                str(row.get("case_id") or ""): row
                for row in load_jsonl(path)
                if isinstance(row, dict) and str(row.get("case_id") or "")
            }
        )
    return maps


def _failure_text(result: dict[str, Any] | None, prediction: dict[str, Any] | None) -> str:
    failures: list[str] = []
    if isinstance(result, dict):
        for step in result.get("step_scores") if isinstance(result.get("step_scores"), list) else []:
            if not isinstance(step, dict):
                continue
            step_idx = step.get("step")
            for item in step.get("failures") if isinstance(step.get("failures"), list) else []:
                text = str(item or "").strip()
                if text:
                    failures.append(f"step{step_idx}:{text}" if step_idx not in (None, "") else text)
        for item in result.get("forbidden_hit") if isinstance(result.get("forbidden_hit"), list) else []:
            text = str(item or "").strip()
            if text:
                failures.append(f"forbidden:{text}")
    if not failures and isinstance(prediction, dict):
        for item in prediction.get("errors") if isinstance(prediction.get("errors"), list) else []:
            text = str(item or "").strip()
            if text:
                failures.append(text)
    return " | ".join(failures)


def _first_action_and_metrics(prediction: dict[str, Any] | None) -> tuple[str, float | str, float | str]:
    if not isinstance(prediction, dict):
        return "", "", ""
    decisions = prediction.get("decisions") if isinstance(prediction.get("decisions"), list) else []
    first = decisions[0] if decisions and isinstance(decisions[0], dict) else {}
    metrics = first.get("_planner_metrics") if isinstance(first.get("_planner_metrics"), dict) else {}
    return (
        str(first.get("action") or ""),
        metrics.get("planner_total_ms", ""),
        metrics.get("api_call_ms", ""),
    )


def build_summary_rows(out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        aggregate = load_json(_aggregate_path(out_dir, spec["prefix"]))
        by_category = aggregate.get("by_category") if isinstance(aggregate.get("by_category"), dict) else {}
        agg = aggregate.get("aggregate") if isinstance(aggregate.get("aggregate"), dict) else {}
        rows.append(
            {
                "model": spec["label"],
                "short_model": spec["short"],
                "runs": aggregate.get("runs"),
                "mean_score_mean": agg.get("mean_score_mean"),
                "mean_score_stdev": agg.get("mean_score_stdev"),
                "pass_rate_mean": agg.get("pass_rate_mean"),
                "pass_rate_stdev": agg.get("pass_rate_stdev"),
                "pass_all_runs_rate": agg.get("pass_all_runs_rate"),
                "pass_any_run_rate": agg.get("pass_any_run_rate"),
                "single_image_probe_pass_mean": (by_category.get("single_image_probe") or {}).get("pass_rate_mean"),
                "probe_then_migration_pass_mean": (by_category.get("probe_then_migration") or {}).get("pass_rate_mean"),
                "migration_feasibility_pass_mean": (by_category.get("migration_feasibility") or {}).get("pass_rate_mean"),
                "migration_feasibility_with_image_pass_mean": (by_category.get("migration_feasibility_with_image") or {}).get("pass_rate_mean"),
                "full_detection_eval_pass_mean": (by_category.get("full_detection_eval") or {}).get("pass_rate_mean"),
                "general_answer_pass_mean": (by_category.get("general_answer") or {}).get("pass_rate_mean"),
            }
        )
    return rows


def build_case_compare_rows(out_dir: Path, cases_path: Path) -> list[dict[str, Any]]:
    cases_by_id = _load_cases_by_id(cases_path)
    model_data = []
    for spec in MODEL_SPECS:
        aggregate = load_json(_aggregate_path(out_dir, spec["prefix"]))
        runs = int(aggregate.get("runs") or 0)
        model_data.append(
            {
                "spec": spec,
                "runs": runs,
                "reward_maps": _load_reward_maps(out_dir, spec["prefix"], runs),
                "prediction_maps": _load_prediction_maps(out_dir, spec["prefix"], runs),
            }
        )

    rows: list[dict[str, Any]] = []
    for cid, case in cases_by_id.items():
        setup = case.get("setup") if isinstance(case.get("setup"), dict) else {}
        row: dict[str, Any] = {
            "case_id": cid,
            "category": str(case.get("category") or ""),
            "query": str(case.get("user_query") or ""),
            "has_image": bool(setup.get("has_image")),
            "image_fixture": str(setup.get("image_fixture") or ""),
            "expected_decisions": _compact(case.get("expected_decisions")),
        }
        any_failed = False
        for item in model_data:
            spec = item["spec"]
            short = spec["short"]
            pass_values: list[bool] = []
            failures: list[str] = []
            first_actions: list[str] = []
            elapsed_list: list[str] = []
            api_list: list[str] = []
            for idx in range(item["runs"]):
                result = item["reward_maps"][idx].get(cid, {})
                prediction = item["prediction_maps"][idx].get(cid, {})
                passed = bool(result.get("passed") is True)
                pass_values.append(passed)
                failure_text = _failure_text(result, prediction)
                if failure_text:
                    failures.append(f"run{idx + 1}:{failure_text}")
                first_action, elapsed_ms, api_ms = _first_action_and_metrics(prediction)
                if first_action:
                    first_actions.append(f"run{idx + 1}:{first_action}")
                if elapsed_ms != "":
                    elapsed_list.append(f"run{idx + 1}:{elapsed_ms}")
                if api_ms != "":
                    api_list.append(f"run{idx + 1}:{api_ms}")
            pass_count = sum(1 for value in pass_values if value)
            run_count = len(pass_values)
            all_passed = bool(pass_values and all(pass_values))
            any_failed = any_failed or bool(pass_values and not all(pass_values))
            row.update(
                {
                    f"{short}_pass_count": pass_count,
                    f"{short}_run_count": run_count,
                    f"{short}_pass_rate": round(pass_count / max(1, run_count), 6),
                    f"{short}_all_passed": all_passed,
                    f"{short}_first_actions": " | ".join(first_actions),
                    f"{short}_failures": " | ".join(failures),
                    f"{short}_planner_elapsed_ms": " | ".join(elapsed_list),
                    f"{short}_api_call_ms": " | ".join(api_list),
                }
            )
        row["any_model_failed"] = any_failed
        rows.append(row)
    return rows


def build_compare_json(out_dir: Path) -> dict[str, Any]:
    models: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        aggregate = load_json(_aggregate_path(out_dir, spec["prefix"]))
        by_category = aggregate.get("by_category") if isinstance(aggregate.get("by_category"), dict) else {}
        agg = aggregate.get("aggregate") if isinstance(aggregate.get("aggregate"), dict) else {}
        models.append(
            {
                "model": spec["short"],
                "label": spec["label"],
                "runs": aggregate.get("runs"),
                "report": str(_aggregate_path(out_dir, spec["prefix"]).relative_to(ROOT)),
                "mean_score_mean": agg.get("mean_score_mean"),
                "mean_score_stdev": agg.get("mean_score_stdev"),
                "pass_rate_mean": agg.get("pass_rate_mean"),
                "pass_rate_stdev": agg.get("pass_rate_stdev"),
                "pass_all_runs_rate": agg.get("pass_all_runs_rate"),
                "pass_any_run_rate": agg.get("pass_any_run_rate"),
                "single_image_probe_pass_all": (by_category.get("single_image_probe") or {}).get("pass_all_runs_rate"),
                "single_image_probe_pass_mean": (by_category.get("single_image_probe") or {}).get("pass_rate_mean"),
                "probe_then_migration_pass_all": (by_category.get("probe_then_migration") or {}).get("pass_all_runs_rate"),
                "probe_then_migration_pass_mean": (by_category.get("probe_then_migration") or {}).get("pass_rate_mean"),
                "migration_feasibility_pass_all": (by_category.get("migration_feasibility") or {}).get("pass_all_runs_rate"),
                "migration_feasibility_pass_mean": (by_category.get("migration_feasibility") or {}).get("pass_rate_mean"),
                "migration_feasibility_with_image_pass_all": (by_category.get("migration_feasibility_with_image") or {}).get("pass_all_runs_rate"),
                "migration_feasibility_with_image_pass_mean": (by_category.get("migration_feasibility_with_image") or {}).get("pass_rate_mean"),
                "full_detection_eval_pass_all": (by_category.get("full_detection_eval") or {}).get("pass_all_runs_rate"),
                "full_detection_eval_pass_mean": (by_category.get("full_detection_eval") or {}).get("pass_rate_mean"),
                "general_answer_pass_all": (by_category.get("general_answer") or {}).get("pass_all_runs_rate"),
                "general_answer_pass_mean": (by_category.get("general_answer") or {}).get("pass_rate_mean"),
            }
        )
    return {
        "schema_version": "1.1",
        "note": "All models are 3x aggregates generated from repeated GRPO stateprompt evals.",
        "models": models,
    }


def build_compare_md(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Qwen3.5 Planner GRPO Stateprompt Eval Compare",
        "",
        "## Setup",
        "",
        "- Cases: `training/planner_grpo_seed_v1/cases/planner_grpo_train_cases.jsonl`",
        "- Decoding: `temperature=0`, `top_p=1`, `seed=42`, `do_sample=false`",
        "- Timeout: `60s`",
        "- Agent: stateprompt/tool-description cleanup enabled",
        "",
        "## Overall",
        "",
        "| Model | Runs | Mean Score | Pass Rate Mean | Pass All Runs |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['runs']} | {row['mean_score_mean']:.6f} | "
            f"{row['pass_rate_mean']:.6f} | {row['pass_all_runs_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Key Categories",
            "",
            "| Model | Single Image Probe | Probe -> Migration | Migration Text | Migration + Image | Full Eval | General Answer |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['single_image_probe_pass_mean']:.6f} | "
            f"{row['probe_then_migration_pass_mean']:.6f} | {row['migration_feasibility_pass_mean']:.6f} | "
            f"{row['migration_feasibility_with_image_pass_mean']:.6f} | {row['full_detection_eval_pass_mean']:.6f} | "
            f"{row['general_answer_pass_mean']:.6f} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GRPO model comparison artifacts.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--report-prefix", default="qwen35_stateprompt_model_compare_4b9b35b")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    cases_path = Path(args.cases)
    if not cases_path.is_absolute():
        cases_path = ROOT / cases_path

    summary_rows = build_summary_rows(out_dir)
    compare_json = build_compare_json(out_dir)
    compare_md = build_compare_md(summary_rows)
    case_rows = build_case_compare_rows(out_dir, cases_path)

    summary_csv_path = out_dir / f"{args.report_prefix}.csv"
    case_csv_path = out_dir / f"{args.report_prefix}_case_audit.csv"
    failed_csv_path = out_dir / f"{args.report_prefix}_failed_cases.csv"
    json_path = out_dir / f"{args.report_prefix}.json"
    md_path = out_dir / f"{args.report_prefix}.md"

    write_csv(summary_csv_path, summary_rows, list(summary_rows[0].keys()))
    write_csv(case_csv_path, case_rows, list(case_rows[0].keys()))
    write_csv(failed_csv_path, [row for row in case_rows if row.get("any_model_failed") is True], list(case_rows[0].keys()))
    write_json(json_path, compare_json)
    write_text(md_path, compare_md)

    print(f"summary_csv={summary_csv_path}")
    print(f"case_audit_csv={case_csv_path}")
    print(f"failed_cases_csv={failed_csv_path}")
    print(f"json={json_path}")
    print(f"md={md_path}")


if __name__ == "__main__":
    main()
