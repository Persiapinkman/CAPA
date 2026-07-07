#!/usr/bin/env python3
"""Run repeated deterministic Planner GRPO rollouts and aggregate rewards."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CASES = ROOT / "training" / "planner_grpo_seed_v1" / "cases" / "planner_grpo_train_cases.jsonl"
DEFAULT_OUT_DIR = ROOT / "training" / "planner_grpo_seed_v1" / "reports" / "repro_eval"


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


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    mu = sum(values) / len(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def rounded(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def _json_compact(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def prediction_stats(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    action_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    empty_decisions = 0
    decision_count = 0
    for row in rows:
        decisions = row.get("decisions") if isinstance(row.get("decisions"), list) else []
        if not decisions:
            empty_decisions += 1
        for err in row.get("errors") if isinstance(row.get("errors"), list) else []:
            error_counts[str(err)] += 1
        for decision in decisions:
            if not isinstance(decision, dict):
                continue
            decision_count += 1
            action = "clarify" if str(decision.get("decision_type") or "") == "clarify" else str(decision.get("action") or "")
            action_counts[action or ""] += 1
    return {
        "rows": len(rows),
        "decision_count": decision_count,
        "empty_decisions": empty_decisions,
        "errors": dict(error_counts),
        "actions": dict(action_counts),
    }


def _load_cases_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    return {
        str(row.get("case_id") or ""): row
        for row in rows
        if isinstance(row, dict) and str(row.get("case_id") or "")
    }


def _results_by_id(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = report.get("results") if isinstance(report.get("results"), list) else []
    return {
        str(row.get("case_id") or ""): row
        for row in rows
        if isinstance(row, dict) and str(row.get("case_id") or "")
    }


def _predictions_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    return {
        str(row.get("case_id") or ""): row
        for row in rows
        if isinstance(row, dict) and str(row.get("case_id") or "")
    }


def _flatten_failures(result: dict[str, Any]) -> str:
    failures: list[str] = []
    for step in result.get("step_scores") if isinstance(result.get("step_scores"), list) else []:
        if not isinstance(step, dict):
            continue
        step_index = step.get("step")
        step_failures = step.get("failures") if isinstance(step.get("failures"), list) else []
        for failure in step_failures:
            text = str(failure or "").strip()
            if not text:
                continue
            if step_index not in (None, ""):
                failures.append(f"step{step_index}:{text}")
            else:
                failures.append(text)
    forbidden = result.get("forbidden_hit") if isinstance(result.get("forbidden_hit"), list) else []
    for action in forbidden:
        text = str(action or "").strip()
        if text:
            failures.append(f"forbidden:{text}")
    return " | ".join(failures)


def _prediction_metrics(prediction: dict[str, Any] | None) -> tuple[str, str, float | str, float | str, int | str, int | str]:
    if not isinstance(prediction, dict):
        return "", "", "", "", "", ""
    decisions = prediction.get("decisions") if isinstance(prediction.get("decisions"), list) else []
    first = decisions[0] if decisions and isinstance(decisions[0], dict) else {}
    used_actions = []
    for item in decisions:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action") or "").strip()
        if action:
            used_actions.append(action)
    metrics = first.get("_planner_metrics") if isinstance(first.get("_planner_metrics"), dict) else {}
    return (
        str(first.get("action") or ""),
        _json_compact(used_actions),
        metrics.get("planner_total_ms", ""),
        metrics.get("api_call_ms", ""),
        metrics.get("input_tokens", ""),
        metrics.get("output_tokens", ""),
    )


def build_case_audit_rows(*, reward_reports: list[Path], prediction_files: list[Path], cases_path: Path) -> list[dict[str, Any]]:
    loaded_reports = [load_json(path) for path in reward_reports]
    report_rows = [_results_by_id(report) for report in loaded_reports]
    prediction_rows = [_predictions_by_id(path) for path in prediction_files]
    cases_by_id = _load_cases_by_id(cases_path)
    case_ids: list[str] = []
    seen: set[str] = set()
    for cid in cases_by_id:
        if cid not in seen:
            seen.add(cid)
            case_ids.append(cid)
    for report in loaded_reports:
        for row in report.get("results") if isinstance(report.get("results"), list) else []:
            cid = str(row.get("case_id") or "")
            if cid and cid not in seen:
                seen.add(cid)
                case_ids.append(cid)

    rows: list[dict[str, Any]] = []
    for cid in case_ids:
        case = cases_by_id.get(cid, {})
        setup = case.get("setup") if isinstance(case.get("setup"), dict) else {}
        expected = case.get("expected_decisions") if isinstance(case.get("expected_decisions"), list) else []
        pass_values: list[bool] = []
        out: dict[str, Any] = {
            "case_id": cid,
            "category": str(case.get("category") or ""),
            "query": str(case.get("user_query") or ""),
            "has_image": bool(setup.get("has_image")),
            "image_fixture": str(setup.get("image_fixture") or ""),
            "expected_decisions": _json_compact(expected),
            "forbidden_actions": _json_compact(case.get("forbidden_actions")),
            "reward_spec": _json_compact(case.get("reward_spec")),
        }
        for index, (rows_by_id, preds_by_id) in enumerate(zip(report_rows, prediction_rows), start=1):
            row = rows_by_id.get(cid, {})
            pred = preds_by_id.get(cid, {})
            passed = row.get("passed")
            pass_values.append(bool(passed is True))
            actual_action, used_actions, elapsed_ms, api_call_ms, input_tokens, output_tokens = _prediction_metrics(pred)
            out.update(
                {
                    f"run{index}_passed": passed,
                    f"run{index}_score": row.get("score", ""),
                    f"run{index}_used_actions": used_actions,
                    f"run{index}_actual_first_action": actual_action,
                    f"run{index}_decisions": _json_compact(pred.get("decisions")),
                    f"run{index}_failures": _flatten_failures(row) or " | ".join(str(x) for x in pred.get("errors", []) if str(x)),
                    f"run{index}_elapsed_ms": elapsed_ms,
                    f"run{index}_api_call_ms": api_call_ms,
                    f"run{index}_input_tokens": input_tokens,
                    f"run{index}_output_tokens": output_tokens,
                }
            )
        out["pass_count"] = sum(1 for value in pass_values if value)
        out["run_count"] = len(pass_values)
        out["all_passed"] = bool(pass_values and all(pass_values))
        out["any_failed"] = bool(pass_values and not all(pass_values))
        rows.append(out)
    return rows


def case_audit_fieldnames(run_count: int) -> list[str]:
    fields = [
        "case_id",
        "category",
        "query",
        "has_image",
        "image_fixture",
        "expected_decisions",
        "forbidden_actions",
        "reward_spec",
    ]
    for index in range(1, run_count + 1):
        fields.extend(
            [
                f"run{index}_passed",
                f"run{index}_score",
                f"run{index}_used_actions",
                f"run{index}_actual_first_action",
                f"run{index}_decisions",
                f"run{index}_failures",
                f"run{index}_elapsed_ms",
                f"run{index}_api_call_ms",
                f"run{index}_input_tokens",
                f"run{index}_output_tokens",
            ]
        )
    fields.extend(["pass_count", "run_count", "all_passed", "any_failed"])
    return fields


def write_case_audit_csvs(
    *,
    reward_reports: list[Path],
    prediction_files: list[Path],
    cases_path: Path,
    out_dir: Path,
    report_prefix: str,
) -> dict[str, str]:
    rows = build_case_audit_rows(
        reward_reports=reward_reports,
        prediction_files=prediction_files,
        cases_path=cases_path,
    )
    fields = case_audit_fieldnames(len(reward_reports))
    audit_path = out_dir / f"{report_prefix}_case_audit.csv"
    failed_path = out_dir / f"{report_prefix}_failed_cases.csv"
    write_csv(audit_path, rows, fields)
    write_csv(failed_path, [row for row in rows if row.get("any_failed") is True], fields)
    return {
        "case_audit_csv": str(audit_path),
        "failed_cases_csv": str(failed_path),
    }


def aggregate_reports(
    *,
    reward_reports: list[Path],
    prediction_files: list[Path],
    args: argparse.Namespace,
    elapsed_ms: float,
) -> dict[str, Any]:
    loaded = [load_json(path) for path in reward_reports]
    summaries = [item.get("summary") if isinstance(item.get("summary"), dict) else {} for item in loaded]
    mean_scores = [float(s["mean_score"]) for s in summaries if isinstance(s.get("mean_score"), (int, float))]
    pass_rates = [float(s["pass_rate"]) for s in summaries if isinstance(s.get("pass_rate"), (int, float))]
    categories: dict[str, list[float]] = defaultdict(list)
    category_pass_rates: dict[str, list[float]] = defaultdict(list)
    for summary in summaries:
        by_cat = summary.get("by_category") if isinstance(summary.get("by_category"), dict) else {}
        for category, item in by_cat.items():
            if isinstance(item, dict) and isinstance(item.get("mean_score"), (int, float)):
                categories[str(category)].append(float(item["mean_score"]))
            if isinstance(item, dict) and isinstance(item.get("pass_rate"), (int, float)):
                category_pass_rates[str(category)].append(float(item["pass_rate"]))

    case_passes: dict[str, list[bool]] = defaultdict(list)
    case_categories: dict[str, str] = {}
    for report in loaded:
        for result in report.get("results") if isinstance(report.get("results"), list) else []:
            if not isinstance(result, dict):
                continue
            cid = str(result.get("case_id") or "").strip()
            if not cid:
                continue
            case_passes[cid].append(bool(result.get("passed") is True))
            case_categories[cid] = str(result.get("category") or "unknown")
    full_run_count = len(loaded)
    pass_all = [
        cid
        for cid, values in case_passes.items()
        if len(values) == full_run_count and all(values)
    ]
    pass_any = [
        cid
        for cid, values in case_passes.items()
        if len(values) == full_run_count and any(values)
    ]

    stats = [prediction_stats(path) for path in prediction_files]
    empty = [float(item["empty_decisions"]) for item in stats]
    decisions = [float(item["decision_count"]) for item in stats]
    action_total: Counter[str] = Counter()
    error_total: Counter[str] = Counter()
    for item in stats:
        action_total.update(item.get("actions") or {})
        error_total.update(item.get("errors") or {})

    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases_path": str(Path(args.cases).resolve()),
        "model": args.model,
        "api_base": args.api_base,
        "report_prefix": args.report_prefix,
        "runs": len(reward_reports),
        "generation_config": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "do_sample": args.do_sample,
        },
        "aggregate": {
            "mean_score_mean": rounded(mean(mean_scores)),
            "mean_score_stdev": rounded(stdev(mean_scores)),
            "pass_rate_mean": rounded(mean(pass_rates)),
            "pass_rate_stdev": rounded(stdev(pass_rates)),
            "pass_all_runs_rate": rounded(len(pass_all) / max(1, len(case_passes))),
            "pass_any_run_rate": rounded(len(pass_any) / max(1, len(case_passes))),
        },
        "by_category": {
            key: {
                "mean_score_mean": rounded(mean(values)),
                "mean_score_stdev": rounded(stdev(values)),
                "pass_rate_mean": rounded(mean(category_pass_rates.get(key, []))),
                "pass_rate_stdev": rounded(stdev(category_pass_rates.get(key, []))),
                "pass_all_runs_rate": rounded(
                    sum(
                        1
                        for cid, passed_values in case_passes.items()
                        if case_categories.get(cid) == key
                        and len(passed_values) == full_run_count
                        and all(passed_values)
                    )
                    / max(1, sum(1 for cid in case_passes if case_categories.get(cid) == key))
                ),
                "runs": len(values),
            }
            for key, values in sorted(categories.items())
        },
        "prediction_stats": {
            "empty_decisions_mean": rounded(mean(empty)),
            "empty_decisions_stdev": rounded(stdev(empty)),
            "decision_count_mean": rounded(mean(decisions)),
            "decision_count_stdev": rounded(stdev(decisions)),
            "actions_total": dict(action_total),
            "errors_total": dict(error_total),
        },
        "timing": {
            "elapsed_ms": round(elapsed_ms, 3),
        },
        "reports": [
            {
                "reward_report": str(reward),
                "prediction_file": str(pred),
                "summary": summary,
                "prediction_stats": stat,
            }
            for reward, pred, summary, stat in zip(reward_reports, prediction_files, summaries, stats)
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated Planner GRPO eval.")
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--report-prefix", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-key", default="token.sdc@2026")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--openai-timeout-seconds", type=int, default=180)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--do-sample", default="false")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    cases_path = Path(args.cases)
    if not cases_path.is_absolute():
        cases_path = ROOT / cases_path

    rollout_script = ROOT / "training" / "planner_grpo_seed_v1" / "scripts" / "run_planner_grpo_rollout.py"
    reward_script = ROOT / "training" / "planner_grpo_seed_v1" / "scripts" / "reward_planner_grpo.py"
    prediction_files: list[Path] = []
    reward_reports: list[Path] = []
    start = time.perf_counter()
    for run_idx in range(1, max(1, int(args.runs)) + 1):
        pred_path = out_dir / f"{args.report_prefix}_run{run_idx}_predictions.jsonl"
        reward_path = out_dir / f"{args.report_prefix}_run{run_idx}_reward.json"
        if args.resume_existing and pred_path.is_file() and reward_path.is_file():
            print(f"[resume {run_idx}/{args.runs}] reuse existing files", flush=True)
            prediction_files.append(pred_path)
            reward_reports.append(reward_path)
            continue
        rollout_cmd = [
            sys.executable,
            str(rollout_script),
            "--cases",
            str(cases_path),
            "--out",
            str(pred_path),
            "--model",
            args.model,
            "--api-base",
            args.api_base,
            "--api-key",
            args.api_key,
            "--max-steps",
            str(args.max_steps),
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--openai-timeout-seconds",
            str(args.openai_timeout_seconds),
            "--temperature",
            str(args.temperature),
            "--top-p",
            str(args.top_p),
            "--seed",
            str(args.seed),
            "--do-sample",
            str(args.do_sample),
        ]
        if args.limit > 0:
            rollout_cmd.extend(["--limit", str(args.limit)])
        print(f"[run {run_idx}/{args.runs}] {' '.join(rollout_cmd)}", flush=True)
        subprocess.run(rollout_cmd, cwd=str(ROOT), check=True)
        reward_cmd = [
            sys.executable,
            str(reward_script),
            "--cases",
            str(cases_path),
            "--predictions",
            str(pred_path),
            "--out",
            str(reward_path),
        ]
        print(f"[score {run_idx}/{args.runs}] {' '.join(reward_cmd)}", flush=True)
        subprocess.run(reward_cmd, cwd=str(ROOT), check=True)
        prediction_files.append(pred_path)
        reward_reports.append(reward_path)

    elapsed_ms = (time.perf_counter() - start) * 1000
    aggregate = aggregate_reports(
        reward_reports=reward_reports,
        prediction_files=prediction_files,
        args=args,
        elapsed_ms=elapsed_ms,
    )
    audit_paths = write_case_audit_csvs(
        reward_reports=reward_reports,
        prediction_files=prediction_files,
        cases_path=cases_path,
        out_dir=out_dir,
        report_prefix=args.report_prefix,
    )
    agg_path = out_dir / f"{args.report_prefix}_aggregate.json"
    aggregate["artifacts"] = audit_paths
    write_json(agg_path, aggregate)
    print(
        "Repeated planner GRPO eval:",
        f"runs={len(reward_reports)}",
        f"mean_score_mean={aggregate['aggregate'].get('mean_score_mean')}",
        f"mean_score_stdev={aggregate['aggregate'].get('mean_score_stdev')}",
        f"case_audit_csv={audit_paths.get('case_audit_csv')}",
        f"failed_cases_csv={audit_paths.get('failed_cases_csv')}",
        f"out={agg_path}",
    )


if __name__ == "__main__":
    main()
