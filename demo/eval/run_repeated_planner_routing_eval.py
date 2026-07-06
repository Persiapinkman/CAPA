#!/usr/bin/env python3
"""Run repeated deterministic Planner routing evals and aggregate results."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = ROOT / "training" / "planner_dpo_train_seed_v1" / "eval" / "planner_routing_eval_90cases.json"
DEFAULT_OUT_DIR = ROOT / "results" / "planner_routing_eval"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    mu = sum(values) / len(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def rounded(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def aggregate_reports(*, reports: list[Path], args: argparse.Namespace, elapsed_ms: float) -> dict[str, Any]:
    loaded = [load_json(path) for path in reports]
    summaries = [item.get("summary") if isinstance(item.get("summary"), dict) else {} for item in loaded]
    accuracies = [float(s["accuracy"]) for s in summaries if isinstance(s.get("accuracy"), (int, float))]
    passed = [float(s["passed"]) for s in summaries if isinstance(s.get("passed"), (int, float))]
    failed = [float(s["failed"]) for s in summaries if isinstance(s.get("failed"), (int, float))]
    timing_rows = [
        s.get("timing") for s in summaries if isinstance(s.get("timing"), dict)
    ]
    case_avg = [
        float(t["case_elapsed_ms_avg"])
        for t in timing_rows
        if isinstance(t.get("case_elapsed_ms_avg"), (int, float))
    ]
    api_avg = [
        float(t["api_call_ms_avg"])
        for t in timing_rows
        if isinstance(t.get("api_call_ms_avg"), (int, float))
    ]
    timeout_counts = [
        float(t["timeout_count"])
        for t in timing_rows
        if isinstance(t.get("timeout_count"), (int, float))
    ]
    slow_case_counts = [
        float(t["slow_case_count"])
        for t in timing_rows
        if isinstance(t.get("slow_case_count"), (int, float))
    ]
    run_elapsed = [
        float(t["elapsed_ms"])
        for t in timing_rows
        if isinstance(t.get("elapsed_ms"), (int, float))
    ]
    usage_rows = [
        s.get("usage") for s in summaries if isinstance(s.get("usage"), dict)
    ]
    input_avg = [
        float(u["input_tokens_avg"])
        for u in usage_rows
        if isinstance(u.get("input_tokens_avg"), (int, float))
    ]
    output_avg = [
        float(u["output_tokens_avg"])
        for u in usage_rows
        if isinstance(u.get("output_tokens_avg"), (int, float))
    ]
    retry_rows = [
        s.get("retry") for s in summaries if isinstance(s.get("retry"), dict)
    ]
    retry_total = [
        float(r["retry_count_total"])
        for r in retry_rows
        if isinstance(r.get("retry_count_total"), (int, float))
    ]
    retry_case_count = [
        float(r["retry_case_count"])
        for r in retry_rows
        if isinstance(r.get("retry_case_count"), (int, float))
    ]
    error_counter: dict[str, int] = {}
    for summary in summaries:
        errors = summary.get("errors") if isinstance(summary.get("errors"), dict) else {}
        by_type = errors.get("by_error_type") if isinstance(errors.get("by_error_type"), dict) else {}
        for key, value in by_type.items():
            try:
                error_counter[str(key)] = error_counter.get(str(key), 0) + int(value)
            except (TypeError, ValueError):
                continue

    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases_path": str(Path(args.cases).resolve()),
        "model": args.model,
        "api_base": args.api_base,
        "report_prefix": args.report_prefix,
        "runs": len(reports),
        "generation_config": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "do_sample": args.do_sample,
        },
        "aggregate": {
            "accuracy_mean": rounded(mean(accuracies)),
            "accuracy_stdev": rounded(stdev(accuracies)),
            "passed_mean": rounded(mean(passed)),
            "passed_stdev": rounded(stdev(passed)),
            "failed_mean": rounded(mean(failed)),
            "failed_stdev": rounded(stdev(failed)),
        },
        "timing": {
            "source": "measured",
            "elapsed_ms": round(elapsed_ms, 3),
            "run_elapsed_ms_mean": rounded(mean(run_elapsed)),
            "run_elapsed_ms_stdev": rounded(stdev(run_elapsed)),
            "case_elapsed_ms_avg_mean": rounded(mean(case_avg)),
            "case_elapsed_ms_avg_stdev": rounded(stdev(case_avg)),
            "api_call_ms_avg_mean": rounded(mean(api_avg)),
            "api_call_ms_avg_stdev": rounded(stdev(api_avg)),
            "slow_case_count_mean": rounded(mean(slow_case_counts)),
            "slow_case_count_stdev": rounded(stdev(slow_case_counts)),
            "timeout_count_mean": rounded(mean(timeout_counts)),
            "timeout_count_stdev": rounded(stdev(timeout_counts)),
        },
        "usage": {
            "input_tokens_avg_mean": rounded(mean(input_avg)),
            "input_tokens_avg_stdev": rounded(stdev(input_avg)),
            "output_tokens_avg_mean": rounded(mean(output_avg)),
            "output_tokens_avg_stdev": rounded(stdev(output_avg)),
        },
        "retry": {
            "retry_count_total_mean": rounded(mean(retry_total)),
            "retry_count_total_stdev": rounded(stdev(retry_total)),
            "retry_case_count_mean": rounded(mean(retry_case_count)),
            "retry_case_count_stdev": rounded(stdev(retry_case_count)),
        },
        "errors": {
            "by_error_type_total": error_counter,
        },
        "reports": [
            {
                "path": str(path),
                "summary": summary,
            }
            for path, summary in zip(reports, summaries)
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated Planner routing evals.")
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--report-prefix", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-key", default="token.sdc@2026")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=180)
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

    reports: list[Path] = []
    start = time.perf_counter()
    for run_idx in range(1, max(1, int(args.runs)) + 1):
        out_path = out_dir / f"{args.report_prefix}_run{run_idx}.json"
        cmd = [
            sys.executable,
            str(Path(__file__).with_name("run_planner_routing_eval.py")),
            "--cases",
            str(cases_path),
            "--out",
            str(out_path),
            "--model",
            args.model,
            "--api-base",
            args.api_base,
            "--api-key",
            args.api_key,
            "--timeout-seconds",
            str(args.timeout_seconds),
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
            cmd.extend(["--limit", str(args.limit)])
        print(f"[run {run_idx}/{args.runs}] {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, cwd=str(ROOT), check=True)
        reports.append(out_path)

    elapsed_ms = (time.perf_counter() - start) * 1000
    aggregate = aggregate_reports(reports=reports, args=args, elapsed_ms=elapsed_ms)
    agg_path = out_dir / f"{args.report_prefix}_aggregate.json"
    write_json(agg_path, aggregate)
    agg = aggregate["aggregate"]
    timing = aggregate["timing"]
    print(
        "Repeated planner routing eval:",
        f"runs={len(reports)}",
        f"accuracy_mean={agg.get('accuracy_mean')}",
        f"accuracy_stdev={agg.get('accuracy_stdev')}",
        f"case_avg_ms_mean={timing.get('case_elapsed_ms_avg_mean')}",
        f"out={agg_path}",
    )


if __name__ == "__main__":
    main()
