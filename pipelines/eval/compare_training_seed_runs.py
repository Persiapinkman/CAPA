#!/usr/bin/env python3
"""Compare a baseline with multiple GRPO training seeds and their mean policy effect."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.eval.compare_generation_runs import load_run, paired_comparison, summary


def parse_mapping(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError(f"expected LABEL=PATH, got {value!r}")
    return label, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=parse_mapping, required=True)
    parser.add_argument("--candidate", type=parse_mapping, action="append", required=True)
    parser.add_argument("--cluster-key", default="entity_id")
    parser.add_argument("--filter-category", default="")
    parser.add_argument("--filter-step", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--title", default="GRPO Multi-Seed Confirmation")
    return parser.parse_args()


def mean_candidate_rows(candidates: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    row_sets = [candidate["rows"] for candidate in candidates]
    keys = sorted(set.intersection(*(set(rows) for rows in row_sets)))
    averaged = {}
    for key in keys:
        source = row_sets[0][key]
        averaged[key] = {
            **source,
            "score": statistics.mean(float(rows[key]["score"]) for rows in row_sets),
            "action_match": statistics.mean(float(rows[key]["action_match"]) for rows in row_sets),
            "json_valid": statistics.mean(float(rows[key]["json_valid"]) for rows in row_sets),
            "extra_text": statistics.mean(float(rows[key]["extra_text"]) for rows in row_sets),
        }
    return averaged


def main() -> None:
    args = parse_args()
    baseline_label, baseline_path = args.baseline
    baseline = load_run(baseline_path)
    candidates = [(label, load_run(path)) for label, path in args.candidate]
    if args.filter_category or args.filter_step:
        for run in [baseline, *(candidate for _, candidate in candidates)]:
            run["rows"] = {
                key: row
                for key, row in run["rows"].items()
                if (not args.filter_category or row["category"] == args.filter_category)
                and (not args.filter_step or int(row["step_index"]) == args.filter_step)
            }
            if not run["rows"]:
                raise ValueError("comparison filter removed all rows from a run")
    seed_comparisons = {
        label: paired_comparison(
            baseline["rows"],
            candidate["rows"],
            seed=args.seed,
            samples=args.bootstrap_samples,
            cluster_key=args.cluster_key,
        )
        for label, candidate in candidates
    }
    averaged_rows = mean_candidate_rows([candidate for _, candidate in candidates])
    mean_comparison = paired_comparison(
        baseline["rows"],
        averaged_rows,
        seed=args.seed,
        samples=args.bootstrap_samples,
        cluster_key=args.cluster_key,
    )
    baseline_summary = summary(baseline["rows"])
    candidate_summaries = {label: summary(candidate["rows"]) for label, candidate in candidates}
    mean_summary = summary(averaged_rows)
    category_deltas = {
        category: mean_summary["categories"][category] - baseline_summary["categories"][category]
        for category in baseline_summary["categories"]
        if category in mean_summary["categories"]
    }
    payload = {
        "filters": {
            "category": args.filter_category or None,
            "step": args.filter_step or None,
        },
        "baseline": {"label": baseline_label, "summary": baseline_summary},
        "training_seeds": candidate_summaries,
        "seed_comparisons": seed_comparisons,
        "mean_policy": {
            "summary": mean_summary,
            "comparison": mean_comparison,
            "category_deltas": category_deltas,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        f"# {args.title}",
        "",
        "| Arm | Case macro | Action match | Score delta | Score 95% CI | Action delta | Action 95% CI | Conclusion |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
        f"| `{baseline_label}` | {baseline_summary['case_macro_mean']:.6f} | "
        f"{baseline_summary.get('action_match_rate', float('nan')):.6f} | - | - | - | - | baseline |",
    ]
    for label, candidate_summary in candidate_summaries.items():
        comparison = seed_comparisons[label]
        lines.append(
            f"| `{label}` | {candidate_summary['case_macro_mean']:.6f} | "
            f"{candidate_summary.get('action_match_rate', float('nan')):.6f} | "
            f"{comparison['case_macro_delta']:+.6f} | "
            f"[{comparison['case_macro_ci95'][0]:+.6f}, {comparison['case_macro_ci95'][1]:+.6f}] | "
            f"{comparison.get('action_match_delta', float('nan')):+.6f} | "
            f"[{comparison.get('action_match_ci95', [float('nan'), float('nan')])[0]:+.6f}, "
            f"{comparison.get('action_match_ci95', [float('nan'), float('nan')])[1]:+.6f}] | "
            f"{comparison['conclusion']} |"
        )
    lines.append(
        f"| `mean_policy` | {mean_summary['case_macro_mean']:.6f} | "
        f"{mean_summary.get('action_match_rate', float('nan')):.6f} | "
        f"{mean_comparison['case_macro_delta']:+.6f} | "
        f"[{mean_comparison['case_macro_ci95'][0]:+.6f}, {mean_comparison['case_macro_ci95'][1]:+.6f}] | "
        f"{mean_comparison.get('action_match_delta', float('nan')):+.6f} | "
        f"[{mean_comparison.get('action_match_ci95', [float('nan'), float('nan')])[0]:+.6f}, "
        f"{mean_comparison.get('action_match_ci95', [float('nan'), float('nan')])[1]:+.6f}] | "
        f"{mean_comparison['conclusion']} |"
    )
    lines.extend(
        [
            "",
            "## Mean Category Deltas",
            "",
            "| Category | Delta |",
            "|---|---:|",
        ]
    )
    for category, delta in sorted(category_deltas.items()):
        lines.append(f"| `{category}` | {delta:+.6f} |")
    lines.extend(
        [
            "",
            f"Intervals use a paired bootstrap clustered by `{args.cluster_key}`.",
            "",
        ]
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "compared", "training_seeds": len(candidates)}))


if __name__ == "__main__":
    main()
