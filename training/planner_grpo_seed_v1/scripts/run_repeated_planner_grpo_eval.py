#!/usr/bin/env python3
"""Run repeated deterministic Planner GRPO rollouts and aggregate rewards."""

from __future__ import annotations

import argparse
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


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    mu = sum(values) / len(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def rounded(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


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
    agg_path = out_dir / f"{args.report_prefix}_aggregate.json"
    write_json(agg_path, aggregate)
    print(
        "Repeated planner GRPO eval:",
        f"runs={len(reward_reports)}",
        f"mean_score_mean={aggregate['aggregate'].get('mean_score_mean')}",
        f"mean_score_stdev={aggregate['aggregate'].get('mean_score_stdev')}",
        f"out={agg_path}",
    )


if __name__ == "__main__":
    main()
