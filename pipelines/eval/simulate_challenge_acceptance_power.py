#!/usr/bin/env python3
"""Monte Carlo sensitivity for the model-separation confirmation gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def beta_parameters(mean: float, icc: float) -> tuple[float, float] | None:
    if icc <= 0:
        return None
    concentration = (1.0 / icc) - 1.0
    return mean * concentration, (1.0 - mean) * concentration


def clustered_success_totals(
    *,
    rng: np.random.Generator,
    simulations: int,
    clusters: int,
    cluster_size: int,
    mean: float,
    icc: float,
) -> np.ndarray:
    params = beta_parameters(mean, icc)
    if params is None:
        probabilities = np.full((simulations, clusters), mean, dtype=np.float64)
    else:
        probabilities = rng.beta(params[0], params[1], size=(simulations, clusters))
    successes = rng.binomial(cluster_size, probabilities)
    return successes.sum(axis=1)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def simulate(
    *,
    simulations: int,
    clusters: int,
    cluster_size: int,
    reference_rate: float,
    base_rate: float,
    icc: float,
    reference_threshold: float,
    base_threshold: float,
    gap_threshold: float,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    reference_success = clustered_success_totals(
        rng=rng,
        simulations=simulations,
        clusters=clusters,
        cluster_size=cluster_size,
        mean=reference_rate,
        icc=icc,
    )
    base_success = clustered_success_totals(
        rng=rng,
        simulations=simulations,
        clusters=clusters,
        cluster_size=cluster_size,
        mean=base_rate,
        icc=icc,
    )
    total_cases = clusters * cluster_size
    reference_observed = reference_success / total_cases
    base_observed = base_success / total_cases
    accepted = (
        (reference_observed >= reference_threshold)
        & (base_observed <= base_threshold)
        & ((reference_observed - base_observed) >= gap_threshold)
    )
    accepted_count = int(accepted.sum())
    ci_low, ci_high = wilson_interval(accepted_count, simulations)
    return {
        "clusters": clusters,
        "cluster_size": cluster_size,
        "total_cases": total_cases,
        "assumed_reference_rate": reference_rate,
        "assumed_base_rate": base_rate,
        "icc": icc,
        "simulations": simulations,
        "acceptance_probability": accepted_count / simulations,
        "monte_carlo_95ci": [ci_low, ci_high],
        "median_observed_reference": float(np.median(reference_observed)),
        "median_observed_base": float(np.median(base_observed)),
    }


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_designs(value: str) -> list[tuple[int, int]]:
    designs: list[tuple[int, int]] = []
    for item in value.split(","):
        clusters, cluster_size = item.lower().strip().split("x", 1)
        designs.append((int(clusters), int(cluster_size)))
    return designs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--designs", default="100x6,75x8,60x10")
    parser.add_argument("--reference-rates", default="0.96,0.97,0.98")
    parser.add_argument("--base-rates", default="0.50,0.60")
    parser.add_argument("--iccs", default="0.02,0.05,0.10,0.15")
    parser.add_argument("--simulations", type=int, default=5000)
    parser.add_argument("--reference-threshold", type=float, default=0.95)
    parser.add_argument("--base-threshold", type=float, default=0.60)
    parser.add_argument("--gap-threshold", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.simulations < 1000:
        raise ValueError("at least 1000 simulations are required")
    results: list[dict[str, Any]] = []
    index = 0
    for clusters, cluster_size in parse_designs(args.designs):
        for reference_rate in parse_float_list(args.reference_rates):
            for base_rate in parse_float_list(args.base_rates):
                for icc in parse_float_list(args.iccs):
                    results.append(
                        simulate(
                            simulations=args.simulations,
                            clusters=clusters,
                            cluster_size=cluster_size,
                            reference_rate=reference_rate,
                            base_rate=base_rate,
                            icc=icc,
                            reference_threshold=args.reference_threshold,
                            base_threshold=args.base_threshold,
                            gap_threshold=args.gap_threshold,
                            seed=args.seed + index,
                        )
                    )
                    index += 1
    payload = {
        "schema_version": "1.0",
        "method": "beta-binomial clustered Monte Carlo; model arms simulated independently (conservative for a positively paired comparison)",
        "thresholds": {
            "reference_min": args.reference_threshold,
            "base_max": args.base_threshold,
            "gap_min": args.gap_threshold,
        },
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"scenarios": len(results), "out": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
