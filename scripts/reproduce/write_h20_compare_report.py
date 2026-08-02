#!/usr/bin/env python3
"""Aggregate H20 3-scenario 3x eval outputs into a single markdown comparison.

Reads every ``<ART_ROOT>/repro_h20/eval/<STAMP>_<arm>/summary.json`` and produces
a wide-format table per scenario. Historical gateway aggregates from
``results/planner_routing_eval/`` are pulled in as an optional reference row.

Zero external dependencies beyond the stdlib; works with any Python >=3.8.

Usage:
  scripts/reproduce/write_h20_compare_report.py \
      --repro-root /apdcephfs_hzlf/share_1227201/zkq/capa_h20/artifacts/CAPA/repro_h20 \
      --out projects/CAPA/reports/H20_THREE_SCENARIO_COMPARE.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any


SCENARIOS = ("routing90", "multistep", "softbnd_dev")
ARMS_ORDER = ("base_4b", "sft", "grpo42", "grpo43", "grpo44", "base_35b")


def load_summary(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def latest_arm_dirs(eval_root: Path) -> dict[str, Path]:
    """Map arm-tag -> latest <STAMP>_<arm> directory."""
    result: dict[str, Path] = {}
    for arm in ARMS_ORDER:
        candidates = sorted(
            (p for p in eval_root.glob(f"*_{arm}") if p.is_dir()),
            key=lambda p: p.name,
        )
        if candidates:
            result[arm] = candidates[-1]
    return result


def score_for(summary: dict[str, Any] | None, scenario: str) -> dict[str, Any]:
    if not summary:
        return {}
    for row in summary.get("results", []):
        if row.get("scenario") == scenario:
            return row
    return {}


def fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def load_historical() -> dict[str, dict[str, Any]]:
    """Optional gateway 3x aggregates for the 90-case scenario."""
    repo = Path(__file__).resolve().parents[2]
    files = {
        "base_4b": repo
        / "results/planner_routing_eval/qwen35_4b_stateprompt_zip90_3x_aggregate.json",
        "base_35b": repo
        / "results/planner_routing_eval/qwen35_35b_a3b_stateprompt_zip90_3x_aggregate.json",
    }
    out: dict[str, dict[str, Any]] = {}
    for arm, path in files.items():
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            agg = payload.get("aggregate", {})
            out[arm] = {
                "accuracy_mean": agg.get("accuracy_mean"),
                "accuracy_stdev": agg.get("accuracy_stdev"),
            }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repro-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    eval_root = args.repro_root / "eval"
    if not eval_root.exists():
        raise SystemExit(f"eval root missing: {eval_root}")

    arm_dirs = latest_arm_dirs(eval_root)
    hist = load_historical()

    lines: list[str] = []
    lines.append("# H20 三场景 3x 评测对照\n")
    lines.append(f"_Repro root_: `{args.repro_root}`\n")
    lines.append(f"_Eval root_: `{eval_root}`\n")
    lines.append("")
    lines.append("Deterministic settings: `temperature=0 top_p=1 seed=42 runs=3`; each row is 3-run mean.\n")

    for scenario in SCENARIOS:
        title = {
            "routing90": "S1. 单步工具路由 (`planner_routing_eval_90cases`, 90 case)",
            "multistep": "S2. 多步工具路由 (`planner_grpo_focused_val_v3`, 31 case)",
            "softbnd_dev": "S3. 软边界状态 (`planner_retry_migrate_v6_grpo_dev`, 225 case)",
        }[scenario]
        lines.append(f"\n## {title}\n")
        lines.append("| Arm | case_macro_mean | case_pass_rate | runs | source |")
        lines.append("|---|---:|---:|---:|---|")
        for arm in ARMS_ORDER:
            arm_dir = arm_dirs.get(arm)
            summary = load_summary(arm_dir / "summary.json") if arm_dir else None
            row = score_for(summary, scenario)
            src = f"`{arm_dir.name}/{scenario}`" if arm_dir else "—"
            lines.append(
                f"| {arm} | {fmt(row.get('case_macro_mean'))} | {fmt(row.get('case_pass_rate'))} | {fmt(row.get('runs'))} | {src} |"
            )
            if scenario == "routing90" and arm in hist:
                h = hist[arm]
                lines.append(
                    f"| {arm} _(gateway historical)_ | — | {fmt(h.get('accuracy_mean'))} ± {fmt(h.get('accuracy_stdev'))} | 3 | `results/planner_routing_eval/*_3x_aggregate.json` |"
                )

    lines.append("")
    lines.append("## Notes")
    lines.append("- softbnd_dev 请按 `entity_id` / `counterfactual_bundle_id` 聚合再解读；此表按 case-macro 呈现。")
    lines.append("- routing90 的 gateway 行是 V100 时代远端网关的 3x 基线，仅供噪声范围参考，不是 pass/fail 判据。")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
