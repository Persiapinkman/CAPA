#!/usr/bin/env python3
"""Paired entity-cluster bootstrap for a completed model-separation eval."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def pass_all_by_case(aggregate: dict[str, Any]) -> dict[str, bool]:
    report_paths = [Path(item["reward_report"]) for item in aggregate.get("reports", [])]
    if len(report_paths) != int(aggregate.get("runs") or 0) or not report_paths:
        raise ValueError("aggregate report paths do not match run count")
    by_run: list[dict[str, bool]] = []
    for path in report_paths:
        report = load_json(path)
        by_run.append(
            {
                str(row["case_id"]): bool(row.get("passed") is True)
                for row in report.get("results", [])
            }
        )
    case_ids = set(by_run[0])
    if any(set(run) != case_ids for run in by_run[1:]):
        raise ValueError("reward reports do not cover identical cases")
    return {case_id: all(run[case_id] for run in by_run) for case_id in case_ids}


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take percentile of empty values")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(probability * (len(ordered) - 1))))
    return ordered[index]


def analyze(
    *,
    cases: list[dict[str, Any]],
    base_pass: dict[str, bool],
    reference_pass: dict[str, bool],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    case_ids = [str(row.get("case_id") or "") for row in cases]
    if not all(case_ids) or len(case_ids) != len(set(case_ids)):
        raise ValueError("cases require unique non-empty case IDs")
    if set(case_ids) != set(base_pass) or set(case_ids) != set(reference_pass):
        raise ValueError("case coverage mismatch")
    clusters: dict[str, list[str]] = {}
    for row in cases:
        cluster = str(row.get("entity_id") or row.get("group_id") or "")
        if not cluster:
            raise ValueError("every case requires entity_id/group_id")
        clusters.setdefault(cluster, []).append(str(row["case_id"]))
    cluster_ids = sorted(clusters)

    def rates(sampled_clusters: list[str]) -> tuple[float, float, float]:
        sampled_cases = [case_id for cluster in sampled_clusters for case_id in clusters[cluster]]
        base_rate = sum(base_pass[case_id] for case_id in sampled_cases) / len(sampled_cases)
        reference_rate = sum(reference_pass[case_id] for case_id in sampled_cases) / len(sampled_cases)
        return base_rate, reference_rate, reference_rate - base_rate

    base_rate, reference_rate, gap = rates(cluster_ids)
    rng = random.Random(seed)
    base_draws: list[float] = []
    reference_draws: list[float] = []
    gap_draws: list[float] = []
    for _ in range(bootstrap_samples):
        sampled = [rng.choice(cluster_ids) for _ in cluster_ids]
        draw_base, draw_reference, draw_gap = rates(sampled)
        base_draws.append(draw_base)
        reference_draws.append(draw_reference)
        gap_draws.append(draw_gap)

    paired = {"reference_only": 0, "base_only": 0, "both": 0, "neither": 0}
    for case_id in case_ids:
        base_value = base_pass[case_id]
        reference_value = reference_pass[case_id]
        key = (
            "both"
            if base_value and reference_value
            else "reference_only"
            if reference_value
            else "base_only"
            if base_value
            else "neither"
        )
        paired[key] += 1

    def interval(draws: list[float]) -> list[float]:
        return [percentile(draws, 0.025), percentile(draws, 0.975)]

    return {
        "schema_version": "1.0",
        "experimental_unit": "case_id",
        "bootstrap_cluster": "entity_id",
        "cases": len(case_ids),
        "clusters": len(cluster_ids),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "metrics": {
            "base_pass_all": base_rate,
            "base_cluster_bootstrap_95ci": interval(base_draws),
            "reference_pass_all": reference_rate,
            "reference_cluster_bootstrap_95ci": interval(reference_draws),
            "reference_minus_base_gap": gap,
            "gap_cluster_bootstrap_95ci": interval(gap_draws),
        },
        "paired_case_counts": paired,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-aggregate", type=Path, required=True)
    parser.add_argument("--reference-aggregate", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    payload = analyze(
        cases=load_jsonl(args.cases),
        base_pass=pass_all_by_case(load_json(args.base_aggregate)),
        reference_pass=pass_all_by_case(load_json(args.reference_aggregate)),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
