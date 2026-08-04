#!/usr/bin/env python3
"""Diagnose where GRPO headroom actually is, for a set of 3x eval reward reports.

Motivation
----------
`mean_score` on a saturated SFT initializer hides the real residual: v7 SFT
ckpt-100 reaches softbnd_dev mean_score=0.9704 while pass_all_runs is only
~0.51.  A GRPO run optimised against `mean_score` therefore chases ~3% of
headroom and collapses to zero within-group reward variance (observed:
frac_reward_zero_std=0.994, grad_norm=0 on 21/22 steps).

This tool re-reads existing reward reports (no model inference, no training)
and reports:

*per-case pass consistency across repeats (pass_all_runs / pass_any_run)
*   which reward component is responsible for every non-perfect case
*   occurrence counts of the safety / process failure signatures
*   the concrete "mean headroom" vs "pass headroom" split per category

Usage
-----
    python pipelines/eval/diagnose_grpo_headroom.py \
        --arm sft=/path/to/eval/<stamp>_sft/softbnd_dev \
        --arm base_4b=/path/to/eval/base_4b_v7_final_3run \
        --out reports/grpo_headroom_<date>.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Per-step reward components produced by reward_planner_grpo.score_expected_step.
STEP_COMPONENTS = (
    "json_valid",
    "decision_type_valid",
    "action_match",
    "argument_match",
    "finish_after_tool",
)

# Case-level boolean failure signatures (safety / process rewards).
CASE_SIGNATURES = (
    "premature_stop_hit",
    "repeated_tool_hit",
    "skip_required_probe_hit",
    "final_tool_not_finished_hit",
)


def load_runs(arm_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    """Load every `*_run<N>_reward.json` report inside an arm directory."""
    runs: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(arm_dir.glob("*_run*_reward.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or "results" not in payload:
            raise ValueError(f"{path}: not a reward report (missing 'results')")
        runs.append((path.name, payload))
    if not runs:
        raise FileNotFoundError(f"{arm_dir}: no *_run*_reward.json found")
    return runs


def failure_reasons(result: dict[str, Any]) -> list[str]:
    """Attribute a non-perfect case to concrete reward components."""
    reasons: list[str] = []
    if result.get("parse_ok") is False:
        reasons.append("parse_failed")
    if result.get("missing_prediction"):
        reasons.append("missing_prediction")
    for step in result.get("step_scores", []) or []:
        detail = step.get("detail") or {}
        index = step.get("step")
        for component in STEP_COMPONENTS:
            value = detail.get(component)
            if value is None:
                continue
            if float(value) < 1.0:
                reasons.append(f"step{index}:{component}")
    if result.get("forbidden_hit"):
        for action in result["forbidden_hit"]:
            reasons.append(f"forbidden:{action}")
    for signature in CASE_SIGNATURES:
        if result.get(signature):
            reasons.append(signature)
    return reasons


def analyse_arm(name: str, arm_dir: Path) -> dict[str, Any]:
    runs = load_runs(arm_dir)
    n_runs = len(runs)

    # case_id -> list of per-run results
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    category_of: dict[str, str] = {}
    for _, payload in runs:
        for result in payload["results"]:
            cid = str(result.get("case_id"))
            by_case[cid].append(result)
            category_of[cid] = str(result.get("category") or "unknown")

    reason_counter: Counter[str] = Counter()
    # reason -> number of distinct cases affected (not run-inflated)
    reason_cases: dict[str, set[str]] = defaultdict(set)
    signature_occurrence: Counter[str] = Counter()

    per_category: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "cases": 0,
            "mean_score_sum": 0.0,
            "pass_all_runs": 0,
            "pass_any_run": 0,
            "flaky": 0,
        }
    )

    total_mean = 0.0
    pass_all = 0
    pass_any = 0
    flaky = 0
    incomplete = 0

    for cid, results in sorted(by_case.items()):
        if len(results) != n_runs:
            incomplete += 1
        scores = [float(r.get("score") or 0.0) for r in results]
        passes = [bool(r.get("passed")) for r in results]
        case_mean = sum(scores) / max(1, len(scores))
        all_pass = all(passes)
        any_pass = any(passes)

        total_mean += case_mean
        pass_all += int(all_pass)
        pass_any += int(any_pass)
        is_flaky = any_pass and not all_pass
        flaky += int(is_flaky)

        cat = category_of[cid]
        bucket = per_category[cat]
        bucket["cases"] += 1
        bucket["mean_score_sum"] += case_mean
        bucket["pass_all_runs"] += int(all_pass)
        bucket["pass_any_run"] += int(any_pass)
        bucket["flaky"] += int(is_flaky)

        for result in results:
            for reason in failure_reasons(result):
                reason_counter[reason] += 1
                reason_cases[reason].add(cid)
            for signature in CASE_SIGNATURES:
                if result.get(signature):
                    signature_occurrence[signature] += 1
            for action in result.get("forbidden_hit") or []:
                signature_occurrence[f"forbidden:{action}"] += 1

    n_cases = len(by_case)
    mean_score = total_mean / max(1, n_cases)
    pass_all_rate = pass_all / max(1, n_cases)

    categories = {}
    for cat, bucket in sorted(per_category.items()):
        count = bucket["cases"]
        cat_mean = bucket["mean_score_sum"] / max(1, count)
        cat_pass = bucket["pass_all_runs"] / max(1, count)
        categories[cat] = {
            "cases": count,
            "mean_score": round(cat_mean, 6),
            "pass_all_runs_rate": round(cat_pass, 6),
            "pass_any_run_rate": round(bucket["pass_any_run"] / max(1, count), 6),
            "flaky_cases": bucket["flaky"],
            "mean_headroom": round(1.0 - cat_mean, 6),
            "pass_headroom": round(1.0 - cat_pass, 6),
        }

    return {
        "arm": name,
        "dir": str(arm_dir),
        "runs": [name for name, _ in runs],
        "n_runs": n_runs,
        "cases": n_cases,
        "incomplete_cases": incomplete,
        "mean_score": round(mean_score, 6),
        "pass_all_runs_rate": round(pass_all_rate, 6),
        "pass_any_run_rate": round(pass_any / max(1, n_cases), 6),
        "flaky_cases": flaky,
        "mean_headroom": round(1.0 - mean_score, 6),
        "pass_headroom": round(1.0 - pass_all_rate, 6),
        "headroom_ratio_pass_over_mean": (
            round((1.0 - pass_all_rate) / (1.0 - mean_score), 3)
            if mean_score < 1.0
            else None
        ),
        "failure_reasons_by_occurrence": dict(reason_counter.most_common()),
        "failure_reasons_by_distinct_case": {
            reason: len(cases) for reason, cases in sorted(
                reason_cases.items(), key=lambda kv: (-len(kv[1]), kv[0])
            )
        },
        "safety_signature_occurrence": dict(signature_occurrence.most_common()),
        "by_category": categories,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="Arm label and directory containing *_run<N>_reward.json reports.",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    arms = []
    for spec in args.arm:
        if "=" not in spec:
            raise SystemExit(f"--arm expects NAME=DIR, got {spec!r}")
        name, _, raw_dir = spec.partition("=")
        arms.append(analyse_arm(name.strip(), Path(raw_dir.strip())))

    report = {
        "tool": "diagnose_grpo_headroom",
        "purpose": "locate real GRPO headroom (pass_all_runs) vs illusory headroom (mean_score)",
        "arms": arms,
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    for arm in arms:
        print(f"\n===== {arm['arm']}  ({arm['cases']} cases x {arm['n_runs']} runs)")
        print(
            f"  mean_score={arm['mean_score']:.4f}  "
            f"pass_all_runs={arm['pass_all_runs_rate']:.4f}  "
            f"pass_any_run={arm['pass_any_run_rate']:.4f}  flaky={arm['flaky_cases']}"
        )
        print(
            f"  mean_headroom={arm['mean_headroom']:.4f}  "
            f"pass_headroom={arm['pass_headroom']:.4f}  "
            f"ratio={arm['headroom_ratio_pass_over_mean']}"
        )
        print("  top failure reasons (distinct cases):")
        for reason, count in list(arm["failure_reasons_by_distinct_case"].items())[:12]:
            print(f"    {count:4d}  {reason}")
        if arm["safety_signature_occurrence"]:
            print("  safety signature occurrence:", arm["safety_signature_occurrence"])
        print("  per-category (mean / pass_all / pass_headroom):")
        for cat, stats in arm["by_category"].items():
            print(
                f"    {cat:28s} {stats['mean_score']:.4f}  "
                f"{stats['pass_all_runs_rate']:.4f}  {stats['pass_headroom']:.4f}"
                f"  flaky={stats['flaky_cases']}"
            )

    if args.out:
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
