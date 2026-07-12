#!/usr/bin/env python3
"""Compare Planner eval prediction JSONL files and surface failure modes."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "demo"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from training.planner_grpo_seed_v1.scripts import reward_planner_grpo as rewardlib  # noqa: E402
from training.planner_grpo_seed_v1.scripts.train_planner_grpo import parse_completion  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Planner eval predictions against case specs.")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row must be a JSON object")
            rows.append(row)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row.get("case_id") or ""), int(row.get("step_index") or 0)


def expected_action(expected: dict[str, Any]) -> str:
    if str(expected.get("decision_type") or "tool") == "clarify":
        return "clarify"
    return rewardlib.normalize_action(str(expected.get("action") or ""))


def actual_action(actual: dict[str, Any] | None) -> str:
    if not isinstance(actual, dict):
        return ""
    if str(actual.get("decision_type") or "") == "clarify":
        return "clarify"
    if str(actual.get("decision_type") or "") == "end":
        return "final_answer"
    return rewardlib.normalize_action(str(actual.get("action") or ""))


def prediction_detail(
    pred: dict[str, Any],
    case: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    reward_spec = dict(rewardlib.DEFAULT_REWARD_SPEC)
    case_spec = case.get("reward_spec")
    if isinstance(case_spec, dict):
        reward_spec.update(case_spec)
    actual = parse_completion(pred.get("scored_completion") or pred.get("completion") or "", first_json_only=True)
    _, info = rewardlib.score_expected_step(expected=expected, actual=actual, reward_spec=reward_spec)
    action = actual_action(actual)
    forbidden = {
        rewardlib.normalize_action(str(item))
        for item in case.get("forbidden_actions", [])
        if str(item).strip()
    }
    return {
        "score": float(pred.get("score") or 0.0),
        "expected_action": expected_action(expected),
        "actual_action": action,
        "actual_decision_type": str(actual.get("decision_type") or "") if isinstance(actual, dict) else "",
        "forbidden_hit": action in forbidden,
        "detail": info.get("detail", {}),
        "failures": info.get("failures", []),
        "completion": pred.get("scored_completion") or pred.get("completion") or "",
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(entries: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in entries:
        by_category[item["category"]].append(item)
    return {
        "overall": {
            "count": len(entries),
            "mean_score": mean([float(item[score_key]["score"]) for item in entries]),
        },
        "categories": {
            category: {
                "count": len(rows),
                "mean_score": mean([float(item[score_key]["score"]) for item in rows]),
                "low_score_count_lt_0_8": sum(float(item[score_key]["score"]) < 0.8 for item in rows),
            }
            for category, rows in sorted(by_category.items())
        },
    }


def truncate(text: str, limit: int = 180) -> str:
    single = " ".join(str(text).split())
    return single if len(single) <= limit else single[: limit - 3] + "..."


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# Planner GRPO Eval Audit",
        "",
        f"- Baseline: `{report['baseline_name']}`",
        f"- Candidate: `{report['candidate_name']}`",
        f"- Compared rows: {report['summary']['rows_compared']}",
        f"- Mean score delta: {report['summary']['candidate_mean'] - report['summary']['baseline_mean']:.6f}",
        "",
        "## Category Delta",
        "",
        "| Category | N | Baseline | Candidate | Delta | Low<0.8 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["category_delta"]:
        lines.append(
            f"| {row['category']} | {row['count']} | {row['baseline_mean']:.6f} | "
            f"{row['candidate_mean']:.6f} | {row['delta']:.6f} | {row['candidate_low_lt_0_8']} |"
        )

    lines.extend(["", "## Main Regressions", ""])
    for item in report["regressions"][:10]:
        lines.append(
            f"- `{item['case_id']}` step {item['step_index']} `{item['category']}`: "
            f"{item['baseline']['score']:.4f} -> {item['candidate']['score']:.4f}; "
            f"expected `{item['candidate']['expected_action']}`, got `{item['candidate']['actual_action']}`; "
            f"failures: {', '.join(item['candidate']['failures']) or 'none'}"
        )

    lines.extend(["", "## Persistent Low Scores", ""])
    for item in report["persistent_low_scores"][:15]:
        lines.append(
            f"- `{item['case_id']}` step {item['step_index']} `{item['category']}`: "
            f"{item['candidate']['score']:.4f}; expected `{item['candidate']['expected_action']}`, "
            f"got `{item['candidate']['actual_action']}`; forbidden_hit={item['candidate']['forbidden_hit']}; "
            f"sample: {truncate(item['candidate']['completion'])}"
        )

    lines.extend(["", "## Action Confusions", ""])
    for name, count in report["candidate_action_confusions"].items():
        lines.append(f"- `{name}`: {count}")

    lines.extend(["", "## Recommendation", ""])
    lines.extend(f"- {line}" for line in report["recommendations"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    cases = {str(row.get("case_id") or ""): row for row in load_jsonl(resolve(args.cases))}
    baseline = {key(row): row for row in load_jsonl(resolve(args.baseline_predictions))}
    candidate = {key(row): row for row in load_jsonl(resolve(args.candidate_predictions))}
    common_keys = sorted(set(baseline) & set(candidate))

    entries: list[dict[str, Any]] = []
    confusions: Counter[str] = Counter()
    for item_key in common_keys:
        case_id, step_index = item_key
        case = cases.get(case_id)
        if not case:
            continue
        expected_list = case.get("expected_decisions") if isinstance(case.get("expected_decisions"), list) else []
        if step_index < 1 or step_index > len(expected_list):
            continue
        expected = expected_list[step_index - 1]
        b_detail = prediction_detail(baseline[item_key], case, expected)
        c_detail = prediction_detail(candidate[item_key], case, expected)
        category = str(case.get("category") or baseline[item_key].get("category") or candidate[item_key].get("category") or "")
        confusions[f"{c_detail['expected_action']} -> {c_detail['actual_action']}"] += 1
        entries.append(
            {
                "case_id": case_id,
                "category": category,
                "step_index": step_index,
                "baseline": b_detail,
                "candidate": c_detail,
                "delta": c_detail["score"] - b_detail["score"],
            }
        )

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_category[entry["category"]].append(entry)
    category_delta = []
    for category, rows in sorted(by_category.items()):
        category_delta.append(
            {
                "category": category,
                "count": len(rows),
                "baseline_mean": mean([row["baseline"]["score"] for row in rows]),
                "candidate_mean": mean([row["candidate"]["score"] for row in rows]),
                "delta": mean([row["candidate"]["score"] - row["baseline"]["score"] for row in rows]),
                "candidate_low_lt_0_8": sum(row["candidate"]["score"] < 0.8 for row in rows),
            }
        )

    regressions = sorted([row for row in entries if row["delta"] < -1e-9], key=lambda row: row["delta"])
    improvements = sorted([row for row in entries if row["delta"] > 1e-9], key=lambda row: row["delta"], reverse=True)
    persistent_low = sorted(
        [row for row in entries if row["candidate"]["score"] < 0.8],
        key=lambda row: (row["candidate"]["score"], row["category"], row["case_id"], row["step_index"]),
    )
    candidate_mean = mean([row["candidate"]["score"] for row in entries])
    baseline_mean = mean([row["baseline"]["score"] for row in entries])
    report = {
        "baseline_name": args.baseline_name,
        "candidate_name": args.candidate_name,
        "inputs": {
            "cases": str(resolve(args.cases)),
            "baseline_predictions": str(resolve(args.baseline_predictions)),
            "candidate_predictions": str(resolve(args.candidate_predictions)),
        },
        "summary": {
            "rows_compared": len(entries),
            "baseline_mean": baseline_mean,
            "candidate_mean": candidate_mean,
            "delta_mean": candidate_mean - baseline_mean,
            "regression_count": len(regressions),
            "improvement_count": len(improvements),
            "candidate_low_score_count_lt_0_8": len(persistent_low),
        },
        "baseline_summary": summarize(entries, "baseline"),
        "candidate_summary": summarize(entries, "candidate"),
        "category_delta": category_delta,
        "regressions": regressions[:50],
        "improvements": improvements[:50],
        "persistent_low_scores": persistent_low[:100],
        "candidate_action_confusions": dict(sorted(confusions.items())),
        "recommendations": [
            "Do not extend GRPO v4 blindly: clarify samples show no action-level improvement, so equal wrong rewards can yield no GRPO advantage.",
            "Create a hard-case refresh path that first raises clarify/pipeline parameter behavior with SFT or high-exploration GRPO, then run GRPO on a balanced hard subset.",
            "Track parameter regressions separately from action regressions; full_detection_eval kept pipeline_eval but degraded task_text quality.",
        ],
    }
    write_json(resolve(args.json_out), report)
    write_text(resolve(args.md_out), render_md(report))
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
