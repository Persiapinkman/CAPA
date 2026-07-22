#!/usr/bin/env python3
"""Validate and summarize trial-level results from the frozen throughput study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260721


def percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def stable_seed(label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return BOOTSTRAP_SEED + int.from_bytes(digest[:4], "big")


def bootstrap_mean_ci(values: list[float], label: str) -> tuple[float, float]:
    rng = random.Random(stable_seed(label))
    size = len(values)
    draws = [statistics.fmean(rng.choice(values) for _ in range(size)) for _ in range(BOOTSTRAP_REPLICATES)]
    return percentile(draws, 0.025), percentile(draws, 0.975)


def bootstrap_ratio_ci(numerator: list[float], denominator: list[float], label: str) -> tuple[float, float]:
    rng = random.Random(stable_seed(label))
    numerator_size = len(numerator)
    denominator_size = len(denominator)
    draws: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        numerator_mean = statistics.fmean(rng.choice(numerator) for _ in range(numerator_size))
        denominator_mean = statistics.fmean(rng.choice(denominator) for _ in range(denominator_size))
        draws.append(numerator_mean / denominator_mean)
    return percentile(draws, 0.025), percentile(draws, 0.975)


def load_complete(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete":
        raise ValueError(f"{path} has status {payload.get('status')!r}, expected 'complete'")
    invalid = [trial for trial in payload["trials"] if not trial["valid"]]
    if invalid:
        raise ValueError(f"{path} contains {len(invalid)} invalid formal trials")
    expected = payload["blocks"] * len(payload["conditions"])
    if len(payload["trials"]) != expected:
        raise ValueError(f"{path} contains {len(payload['trials'])} trials, expected {expected}")
    return payload


def group_trials(payload: dict[str, Any]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for trial in payload["trials"]:
        grouped[(trial["input_length"], trial["concurrency"])].append(trial)
    for key, trials in grouped.items():
        blocks = sorted(trial["block"] for trial in trials)
        expected_blocks = list(range(1, payload["blocks"] + 1))
        if blocks != expected_blocks:
            raise ValueError(f"condition {key} has blocks {blocks}, expected {expected_blocks}")
    return dict(grouped)


def trial_metric(trials: list[dict[str, Any]], key: str) -> list[float]:
    return [float(trial["summary"][key]) for trial in trials]


def request_metric(trials: list[dict[str, Any]], key: str) -> list[float]:
    return [
        float(request[key])
        for trial in trials
        for request in trial["requests"]
        if request["success"] and request.get(key) is not None
    ]


def optional_percentile(values: list[float], probability: float) -> float | None:
    return percentile(values, probability) if values else None


def telemetry_metric(trials: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for trial in trials:
        value = trial["telemetry_summary"].get(key)
        if value is not None:
            values.append(float(value))
    return values


def total_peak_memory_mib(trial: dict[str, Any]) -> float:
    return sum(
        float(row["memory_peak_mib"])
        for row in trial["telemetry_summary"]["per_gpu"].values()
        if row["memory_peak_mib"] is not None
    )


def make_model_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    grouped = group_trials(payload)
    rows: list[dict[str, Any]] = []
    gpu_count = int(payload["gpu_count"])
    for (input_length, concurrency), trials in sorted(grouped.items()):
        output_rates = trial_metric(trials, "output_tok_per_s")
        prompt_rates = trial_metric(trials, "prompt_tok_per_s")
        total_rates = trial_metric(trials, "total_tok_per_s")
        request_rates = trial_metric(trials, "request_per_s")
        powers = telemetry_metric(trials, "service_power_mean_w")
        latency = request_metric(trials, "latency_s")
        ttft = request_metric(trials, "ttft_s")
        tpot = request_metric(trials, "tpot_s")
        ci_low, ci_high = bootstrap_mean_ci(
            output_rates, f"{payload['model_label']}-{input_length}-{concurrency}-output"
        )
        output_mean = statistics.fmean(output_rates)
        power_mean = statistics.fmean(powers) if powers else None
        rows.append({
            "model_label": payload["model_label"],
            "served_model_name": payload["served_model_name"],
            "backend": payload.get("backend", "vllm_openai_service"),
            "execution_semantics": payload.get("execution_semantics", "concurrent_http_requests"),
            "gpu_count": gpu_count,
            "input_length": input_length,
            "output_length": int(payload["output_length"]),
            "concurrency": concurrency,
            "trial_n": len(trials),
            "request_n": sum(len(trial["requests"]) for trial in trials),
            "error_n": sum(trial["summary"]["error_count"] for trial in trials),
            "output_tok_s_mean": output_mean,
            "output_tok_s_sd": statistics.stdev(output_rates),
            "output_tok_s_cv_pct": statistics.stdev(output_rates) / output_mean * 100,
            "output_tok_s_ci95_low": ci_low,
            "output_tok_s_ci95_high": ci_high,
            "output_tok_s_per_gpu": output_mean / gpu_count,
            "prompt_tok_s_mean": statistics.fmean(prompt_rates),
            "total_tok_s_mean": statistics.fmean(total_rates),
            "request_s_mean": statistics.fmean(request_rates),
            "latency_p50_s": percentile(latency, 0.50),
            "latency_p95_s": percentile(latency, 0.95),
            "ttft_p50_s": optional_percentile(ttft, 0.50),
            "ttft_p95_s": optional_percentile(ttft, 0.95),
            "tpot_p50_ms": None if not tpot else percentile(tpot, 0.50) * 1000,
            "tpot_p95_ms": None if not tpot else percentile(tpot, 0.95) * 1000,
            "service_power_mean_w": power_mean,
            "output_tok_s_per_w": None if power_mean is None else output_mean / power_mean,
            "energy_kwh_per_million_output_tokens": (
                None if power_mean is None else power_mean / output_mean * 1_000_000 / 3_600_000
            ),
            "allocated_memory_peak_gib_mean": statistics.fmean(total_peak_memory_mib(trial) for trial in trials) / 1024,
        })
    return rows


def make_comparison_rows(
    four_b_payload: dict[str, Any],
    thirty_two_b_payload: dict[str, Any],
    four_b_rows: list[dict[str, Any]],
    thirty_two_b_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    four_groups = group_trials(four_b_payload)
    thirty_two_groups = group_trials(thirty_two_b_payload)
    four_index = {(row["input_length"], row["concurrency"]): row for row in four_b_rows}
    thirty_two_index = {(row["input_length"], row["concurrency"]): row for row in thirty_two_b_rows}
    if set(four_index) != set(thirty_two_index):
        raise ValueError("model condition sets differ")
    rows: list[dict[str, Any]] = []
    for key in sorted(four_index):
        four_row = four_index[key]
        thirty_two_row = thirty_two_index[key]
        four_rates = trial_metric(four_groups[key], "output_tok_per_s")
        thirty_two_rates = trial_metric(thirty_two_groups[key], "output_tok_per_s")
        ratio = four_row["output_tok_s_mean"] / thirty_two_row["output_tok_s_mean"]
        ratio_low, ratio_high = bootstrap_ratio_ci(
            four_rates, thirty_two_rates, f"ratio-{key[0]}-{key[1]}"
        )
        rows.append({
            "input_length": key[0],
            "output_length": four_row["output_length"],
            "concurrency": key[1],
            "four_b_output_tok_s": four_row["output_tok_s_mean"],
            "four_b_ci95_low": four_row["output_tok_s_ci95_low"],
            "four_b_ci95_high": four_row["output_tok_s_ci95_high"],
            "thirty_two_b_output_tok_s": thirty_two_row["output_tok_s_mean"],
            "thirty_two_b_ci95_low": thirty_two_row["output_tok_s_ci95_low"],
            "thirty_two_b_ci95_high": thirty_two_row["output_tok_s_ci95_high"],
            "four_b_over_thirty_two_b_total_ratio": ratio,
            "ratio_ci95_low": ratio_low,
            "ratio_ci95_high": ratio_high,
            "four_b_over_thirty_two_b_per_gpu_ratio": (
                four_row["output_tok_s_per_gpu"] / thirty_two_row["output_tok_s_per_gpu"]
            ),
            "four_b_latency_p95_s": four_row["latency_p95_s"],
            "thirty_two_b_latency_p95_s": thirty_two_row["latency_p95_s"],
            "four_b_ttft_p95_s": four_row["ttft_p95_s"],
            "thirty_two_b_ttft_p95_s": thirty_two_row["ttft_p95_s"],
            "four_b_service_power_w": four_row["service_power_mean_w"],
            "thirty_two_b_service_power_w": thirty_two_row["service_power_mean_w"],
            "four_b_energy_kwh_per_million_output_tokens": four_row["energy_kwh_per_million_output_tokens"],
            "thirty_two_b_energy_kwh_per_million_output_tokens": thirty_two_row["energy_kwh_per_million_output_tokens"],
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def model_audit(payload: dict[str, Any]) -> dict[str, Any]:
    trials = payload["trials"]
    requests = [request for trial in trials for request in trial["requests"]]
    return {
        "model_label": payload["model_label"],
        "status": payload["status"],
        "formal_trial_count": len(trials),
        "valid_trial_count": sum(trial["valid"] for trial in trials),
        "request_count": len(requests),
        "successful_request_count": sum(request["success"] for request in requests),
        "http_error_count": sum(not request["success"] for request in requests),
        "prompt_token_mismatch_count": sum(
            request["prompt_tokens"] != trial["input_length"]
            for trial in trials
            for request in trial["requests"]
            if request["success"]
        ),
        "completion_token_mismatch_count": sum(
            request["completion_tokens"] != trial["output_length"]
            for trial in trials
            for request in trial["requests"]
            if request["success"]
        ),
        "completion_tokens_total": sum(
            request["completion_tokens"] for request in requests if request["success"]
        ),
        "prompt_tokens_total": sum(request["prompt_tokens"] for request in requests if request["success"]),
        "telemetry_error_count": sum(len(trial["telemetry_errors"]) for trial in trials),
        "backend": payload.get("backend", "vllm_openai_service"),
        "execution_semantics": payload.get("execution_semantics", "concurrent_http_requests"),
        "server_version_response": payload.get("server_version_response"),
        "model_class": payload.get("model_class"),
        "sanity": payload.get("sanity"),
        "prompt_pool_sha256": payload["prompt_pool_sha256"],
        "client_script_sha256": payload["client_script_sha256"],
    }


def format_ci(row: dict[str, Any], prefix: str) -> str:
    return (
        f"{row[prefix]:.2f} "
        f"[{row[prefix + '_ci95_low']:.2f}, {row[prefix + '_ci95_high']:.2f}]"
    )


def markdown_fragment(
    model_rows: list[dict[str, Any]], comparison_rows: list[dict[str, Any]]
) -> str:
    lines = [
        "# 自动生成的吞吐统计摘要",
        "",
        f"Bootstrap：trial 级、{BOOTSTRAP_REPLICATES:,} 次、percentile 95% CI；seed={BOOTSTRAP_SEED}。",
        "",
        "## 模型结果",
        "",
        "| 模型 | ISL/OSL | 并发 | output tok/s（95% CI） | tok/s/GPU | P95 latency | P95 TTFT | 平均功率 | kWh/M output tok |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in model_rows:
        ttft = "N/A" if row["ttft_p95_s"] is None else f"{row['ttft_p95_s']:.2f}s"
        lines.append(
            f"| {row['model_label']} | {row['input_length']}/{row['output_length']} | {row['concurrency']} | "
            f"{row['output_tok_s_mean']:.2f} [{row['output_tok_s_ci95_low']:.2f}, {row['output_tok_s_ci95_high']:.2f}] | "
            f"{row['output_tok_s_per_gpu']:.2f} | {row['latency_p95_s']:.2f}s | {ttft} | "
            f"{row['service_power_mean_w']:.1f}W | {row['energy_kwh_per_million_output_tokens']:.3f} |"
        )
    lines.extend([
        "",
        "## 4B / 32B 比较",
        "",
        "| ISL/OSL | 并发 | 4B output tok/s | 32B output tok/s | 4B/32B 总吞吐比（95% CI） | 每 GPU 吞吐比 |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for row in comparison_rows:
        lines.append(
            f"| {row['input_length']}/{row['output_length']} | {row['concurrency']} | "
            f"{row['four_b_output_tok_s']:.2f} | {row['thirty_two_b_output_tok_s']:.2f} | "
            f"{row['four_b_over_thirty_two_b_total_ratio']:.2f}× "
            f"[{row['ratio_ci95_low']:.2f}, {row['ratio_ci95_high']:.2f}] | "
            f"{row['four_b_over_thirty_two_b_per_gpu_ratio']:.2f}× |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--four-b", type=Path, required=True)
    parser.add_argument("--thirty-two-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    four_b = load_complete(args.four_b)
    thirty_two_b = load_complete(args.thirty_two_b)
    four_b_rows = make_model_rows(four_b)
    thirty_two_b_rows = make_model_rows(thirty_two_b)
    model_rows = four_b_rows + thirty_two_b_rows
    comparisons = make_comparison_rows(four_b, thirty_two_b, four_b_rows, thirty_two_b_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "model_condition_summary.csv", model_rows)
    write_csv(args.output_dir / "comparison_summary.csv", comparisons)
    audit = {
        "schema_version": 1,
        "bootstrap": {
            "unit": "trial",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "interval": "percentile_95",
        },
        "models": [model_audit(four_b), model_audit(thirty_two_b)],
    }
    (args.output_dir / "validation_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    summary = {
        "model_conditions": model_rows,
        "comparisons": comparisons,
        "audit": audit,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "SUMMARY_TABLES.md").write_text(markdown_fragment(model_rows, comparisons))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
