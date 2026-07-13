#!/usr/bin/env python3
"""Compare repeated generation runs with case-clustered bootstrap intervals."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_run(path: Path) -> dict[str, Any]:
    record = read_json(path)
    eval_path = Path(record["data"]["files"]["eval"])
    expected_by_key = {
        (str(row["case_id"]), int(row["step_index"])): decision_action(row["expected_step"])
        for row in read_jsonl(eval_path)
    }
    prediction_paths = record.get("artifacts", {}).get("predictions", [])
    repeats = [read_jsonl(Path(item)) for item in prediction_paths]
    if not repeats:
        raise ValueError(f"{path}: no predictions")
    by_repeat = [
        {(str(row["case_id"]), int(row["step_index"])): row for row in rows}
        for rows in repeats
    ]
    keys = sorted(set.intersection(*(set(rows) for rows in by_repeat)))
    averaged: dict[tuple[str, int], dict[str, Any]] = {}
    for key in keys:
        source = by_repeat[0][key]
        averaged[key] = {
            "case_id": key[0],
            "step_index": key[1],
            "category": source["category"],
            "entity_id": str(source.get("entity_id") or ""),
            "group_id": str(source.get("group_id") or key[0]),
            "score": statistics.mean(float(rows[key]["score"]) for rows in by_repeat),
            "action_match": statistics.mean(
                float(decision_action(rows[key].get("scored_completion", "")) == expected_by_key.get(key, ""))
                for rows in by_repeat
            ),
            "json_valid": statistics.mean(float(bool(rows[key]["json_valid"])) for rows in by_repeat),
            "extra_text": statistics.mean(
                float(bool(rows[key]["extra_text_after_json"])) for rows in by_repeat
            ),
        }
    return {"record": record, "rows": averaged}


def decision_action(value: str | dict[str, Any]) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return ""
    if not isinstance(value, dict):
        return ""
    decision_type = str(value.get("decision_type") or "")
    if decision_type in {"clarify", "end"}:
        return decision_type
    return str(value.get("action") or "")


def summary(rows: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    values = list(rows.values())
    by_case: dict[str, list[float]] = defaultdict(list)
    by_category: dict[str, list[float]] = defaultdict(list)
    action_by_category: dict[str, list[float]] = defaultdict(list)
    by_category_step: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in values:
        by_case[row["case_id"]].append(float(row["score"]))
        by_category[row["category"]].append(float(row["score"]))
        if row.get("action_match") is not None:
            action_by_category[row["category"]].append(float(row["action_match"]))
        by_category_step[(row["category"], int(row["step_index"]))].append(row)
    case_means = [statistics.mean(items) for items in by_case.values()]
    category_means = {key: statistics.mean(items) for key, items in sorted(by_category.items())}
    result = {
        "steps": len(values),
        "cases": len(by_case),
        "step_mean": statistics.mean(float(row["score"]) for row in values),
        "case_macro_mean": statistics.mean(case_means),
        "category_macro_mean": statistics.mean(category_means.values()),
        "exact_pass_rate": statistics.mean(float(float(row["score"]) >= 1.0) for row in values),
        "json_valid_rate": statistics.mean(float(row["json_valid"]) for row in values),
        "extra_text_rate": statistics.mean(float(row["extra_text"]) for row in values),
        "categories": category_means,
        "category_steps": {
            f"{category}#step{step}": {
                "count": len(items),
                "mean_score": statistics.mean(float(row["score"]) for row in items),
                "action_match_rate": (
                    statistics.mean(float(row["action_match"]) for row in items)
                    if all(row.get("action_match") is not None for row in items)
                    else None
                ),
            }
            for (category, step), items in sorted(by_category_step.items())
        },
    }
    if action_by_category:
        result["action_match_rate"] = statistics.mean(
            float(row["action_match"]) for row in values if row.get("action_match") is not None
        )
        result["category_action_match"] = {
            key: statistics.mean(items) for key, items in sorted(action_by_category.items())
        }
    return result


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(fraction * len(ordered))))
    return ordered[index]


def paired_comparison(
    left: dict[tuple[str, int], dict[str, Any]],
    right: dict[tuple[str, int], dict[str, Any]],
    *,
    seed: int,
    samples: int,
    cluster_key: str = "case_id",
) -> dict[str, Any]:
    keys = sorted(set(left) & set(right))
    step_deltas_by_case: dict[str, list[float]] = defaultdict(list)
    action_deltas_by_case: dict[str, list[float]] = defaultdict(list)
    has_action_match = bool(keys) and all(
        left[key].get("action_match") is not None
        and right[key].get("action_match") is not None
        for key in keys
    )
    case_clusters: dict[str, str] = {}
    for key in keys:
        case_id = key[0]
        step_deltas_by_case[case_id].append(
            float(right[key]["score"]) - float(left[key]["score"])
        )
        if has_action_match:
            action_deltas_by_case[case_id].append(
                float(right[key]["action_match"]) - float(left[key]["action_match"])
            )
        if cluster_key == "case_id":
            case_clusters[case_id] = case_id
        else:
            case_clusters[case_id] = str(
                right[key].get(cluster_key) or left[key].get(cluster_key) or case_id
            )
    case_deltas = {
        case_id: statistics.mean(values) for case_id, values in step_deltas_by_case.items()
    }
    action_case_deltas = {
        case_id: statistics.mean(values) for case_id, values in action_deltas_by_case.items()
    }
    cluster_cases: dict[str, list[str]] = defaultdict(list)
    for case_id, cluster_id in case_clusters.items():
        cluster_cases[cluster_id].append(case_id)
    cluster_ids = sorted(cluster_cases)
    rng = random.Random(seed)
    step_bootstrap: list[float] = []
    case_bootstrap: list[float] = []
    action_step_bootstrap: list[float] = []
    action_case_bootstrap: list[float] = []
    for _ in range(samples):
        sampled_clusters = [rng.choice(cluster_ids) for _ in cluster_ids]
        sampled_cases_ids = [
            case_id for cluster_id in sampled_clusters for case_id in cluster_cases[cluster_id]
        ]
        sampled_steps = [
            delta for case_id in sampled_cases_ids for delta in step_deltas_by_case[case_id]
        ]
        sampled_cases = [case_deltas[case_id] for case_id in sampled_cases_ids]
        step_bootstrap.append(statistics.mean(sampled_steps))
        case_bootstrap.append(statistics.mean(sampled_cases))
        if has_action_match:
            sampled_action_steps = [
                delta for case_id in sampled_cases_ids for delta in action_deltas_by_case[case_id]
            ]
            sampled_action_cases = [action_case_deltas[case_id] for case_id in sampled_cases_ids]
            action_step_bootstrap.append(statistics.mean(sampled_action_steps))
            action_case_bootstrap.append(statistics.mean(sampled_action_cases))
    step_deltas = [delta for values in step_deltas_by_case.values() for delta in values]
    case_delta_values = list(case_deltas.values())
    step_ci = [percentile(step_bootstrap, 0.025), percentile(step_bootstrap, 0.975)]
    case_ci = [percentile(case_bootstrap, 0.025), percentile(case_bootstrap, 0.975)]
    if case_ci[0] > 0:
        conclusion = "supported"
    elif case_ci[1] < 0:
        conclusion = "regressed"
    else:
        conclusion = "inconclusive"
    result = {
        "steps": len(step_deltas),
        "cases": len(case_delta_values),
        "clusters": len(cluster_ids),
        "step_weighted_delta": statistics.mean(step_deltas),
        "step_weighted_ci95": step_ci,
        "case_macro_delta": statistics.mean(case_delta_values),
        "case_macro_ci95": case_ci,
        "improved_steps": sum(value > 1e-12 for value in step_deltas),
        "unchanged_steps": sum(abs(value) <= 1e-12 for value in step_deltas),
        "regressed_steps": sum(value < -1e-12 for value in step_deltas),
        "conclusion": conclusion,
        "bootstrap": {"cluster": cluster_key, "samples": samples, "seed": seed},
    }
    if has_action_match:
        action_step_deltas = [
            delta for values in action_deltas_by_case.values() for delta in values
        ]
        action_case_delta_values = list(action_case_deltas.values())
        result.update(
            {
                "action_match_delta": statistics.mean(action_step_deltas),
                "action_match_ci95": [
                    percentile(action_step_bootstrap, 0.025),
                    percentile(action_step_bootstrap, 0.975),
                ],
                "action_match_case_macro_delta": statistics.mean(action_case_delta_values),
                "action_match_case_macro_ci95": [
                    percentile(action_case_bootstrap, 0.025),
                    percentile(action_case_bootstrap, 0.975),
                ],
            }
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="LABEL=run_record.json")
    parser.add_argument("--pair", action="append", required=True, help="LEFT=RIGHT")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cluster-key", default="case_id")
    parser.add_argument("--filter-category", default="")
    parser.add_argument("--filter-step", type=int, default=0)
    parser.add_argument("--title", default="Planner ChatML Training-Effect Study")
    parser.add_argument("--update-run-records", action="store_true")
    return parser.parse_args()


def parse_mapping(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key or not item:
            raise ValueError(f"invalid mapping: {value}")
        result[key] = item
    return result


def main() -> None:
    args = parse_args()
    run_paths = parse_mapping(args.run)
    runs = {label: load_run(Path(path)) for label, path in run_paths.items()}
    if args.filter_category or args.filter_step:
        for run in runs.values():
            run["rows"] = {
                key: row
                for key, row in run["rows"].items()
                if (not args.filter_category or row["category"] == args.filter_category)
                and (not args.filter_step or int(row["step_index"]) == args.filter_step)
            }
            if not run["rows"]:
                raise ValueError("comparison filter removed all rows from a run")
    summaries = {label: summary(run["rows"]) for label, run in runs.items()}
    comparisons: dict[str, Any] = {}
    parsed_pairs: list[tuple[str, str, str]] = []
    for pair in args.pair:
        left, separator, right = pair.partition("=")
        if not separator or left not in runs or right not in runs:
            raise ValueError(f"invalid pair: {pair}")
        comparison_key = f"{left}_to_{right}"
        comparisons[comparison_key] = paired_comparison(
            runs[left]["rows"],
            runs[right]["rows"],
            seed=args.seed,
            samples=args.bootstrap_samples,
            cluster_key=args.cluster_key,
        )
        parsed_pairs.append((left, right, comparison_key))
    payload = {
        "filters": {
            "category": args.filter_category or None,
            "step": args.filter_step or None,
        },
        "runs": summaries,
        "comparisons": comparisons,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# {args.title}",
        "",
        "## Run Summary",
        "",
        "| Run | Step mean | Case macro | Category macro | Action match | Exact pass | JSON valid | Extra text |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, item in summaries.items():
        lines.append(
            f"| `{label}` | {item['step_mean']:.6f} | {item['case_macro_mean']:.6f} | "
            f"{item['category_macro_mean']:.6f} | {item.get('action_match_rate', float('nan')):.6f} | "
            f"{item['exact_pass_rate']:.6f} | "
            f"{item['json_valid_rate']:.6f} | {item['extra_text_rate']:.6f} |"
        )
    category_names = sorted(
        {category for item in summaries.values() for category in item["categories"]}
    )
    category_header = "| Category | " + " | ".join(
        f"{label} score | {label} action" for label in summaries
    ) + " |"
    category_separator = "|---|" + "|".join("---:|---:" for _ in summaries) + "|"
    lines.extend(["", "## Category Results", "", category_header, category_separator])
    for category in category_names:
        cells = []
        for item in summaries.values():
            cells.extend(
                [
                    f"{item['categories'][category]:.6f}",
                    f"{item.get('category_action_match', {}).get(category, float('nan')):.6f}",
                ]
            )
        lines.append(f"| `{category}` | " + " | ".join(cells) + " |")

    category_step_names = sorted(
        {key for item in summaries.values() for key in item["category_steps"]}
    )
    step_header = "| Category step | " + " | ".join(
        f"{label} score | {label} action" for label in summaries
    ) + " |"
    step_separator = "|---|" + "|".join("---:|---:" for _ in summaries) + "|"
    lines.extend(["", "## Step Results", "", step_header, step_separator])
    for key in category_step_names:
        cells = []
        for item in summaries.values():
            step = item["category_steps"][key]
            cells.extend(
                [
                    f"{step['mean_score']:.6f}",
                    f"{step.get('action_match_rate', float('nan')):.6f}",
                ]
            )
        lines.append(f"| `{key}` | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Paired Comparisons",
            "",
            "| Comparison | Step delta [95% CI] | Case-macro delta [95% CI] | Action delta [95% CI] | +/=/- steps | Conclusion |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for label, item in comparisons.items():
        action_delta = item.get("action_match_delta", float("nan"))
        action_ci = item.get("action_match_ci95", [float("nan"), float("nan")])
        lines.append(
            f"| `{label}` | {item['step_weighted_delta']:+.6f} "
            f"[{item['step_weighted_ci95'][0]:+.6f}, {item['step_weighted_ci95'][1]:+.6f}] | "
            f"{item['case_macro_delta']:+.6f} "
            f"[{item['case_macro_ci95'][0]:+.6f}, {item['case_macro_ci95'][1]:+.6f}] | "
            f"{action_delta:+.6f} [{action_ci[0]:+.6f}, {action_ci[1]:+.6f}] | "
            f"{item['improved_steps']}/{item['unchanged_steps']}/{item['regressed_steps']} | "
            f"{item['conclusion']} |"
        )
    lines.extend(
        [
            "",
            f"Intervals use a paired bootstrap clustered by `{args.cluster_key}`. Claim scope follows the registered study split.",
            "",
        ]
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines), encoding="utf-8")

    if args.update_run_records:
        compared_labels = {right for _, right, _ in parsed_pairs}
        for label, path in run_paths.items():
            record = runs[label]["record"]
            record.setdefault("metrics", {})["primary"] = {
                "name": "case_macro_mean_verifier_score",
                "value": summaries[label]["case_macro_mean"],
                "higher_is_better": True,
            }
            record["metrics"]["study_summary"] = summaries[label]
            record.setdefault("artifacts", {})["study_comparison"] = {
                "json": str(args.output_json),
                "markdown": str(args.output_md),
            }
            prediction_paths = record["artifacts"].get("predictions", [])
            record["artifacts"]["prediction_sha256"] = {
                str(path): sha256_file(Path(path)) for path in prediction_paths
            }
            if label not in compared_labels:
                record["decision"] = {
                    "outcome": "baseline",
                    "rationale": "Reference arm for the paired study comparisons.",
                }
            Path(path).write_text(
                json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        for left, right, comparison_key in parsed_pairs:
            path = Path(run_paths[right])
            record = read_json(path)
            result = comparisons[comparison_key]
            outcome = {
                "supported": "promote_development",
                "regressed": "reject",
                "inconclusive": "inconclusive",
            }[result["conclusion"]]
            record.setdefault("metrics", {})["paired_comparison"] = {
                "reference_arm": left,
                **result,
            }
            record["decision"] = {
                "outcome": outcome,
                "rationale": (
                    f"Paired case-clustered comparison against {left}: "
                    f"case-macro delta={result['case_macro_delta']:+.6f}, "
                    f"95% CI=[{result['case_macro_ci95'][0]:+.6f}, "
                    f"{result['case_macro_ci95'][1]:+.6f}]."
                ),
            }
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "compared", "runs": len(runs), "pairs": len(comparisons)}))


if __name__ == "__main__":
    main()
