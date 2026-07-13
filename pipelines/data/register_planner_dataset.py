#!/usr/bin/env python3
"""Audit and register the current Planner focused dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from capa.evaluation.dataset_audit import build_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    base = Path("training/planner_grpo_seed_v1")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default="planner_focused_v3")
    parser.add_argument("--output-dir", type=Path, default=Path("data/datasets/planner_focused_v3"))
    parser.add_argument("--source-cases", type=Path, default=base / "cases/planner_grpo_focused_4b_cases.jsonl")
    parser.add_argument("--train-cases", type=Path, default=base / "cases/planner_grpo_focused_train_v3_cases.jsonl")
    parser.add_argument("--dev-cases", type=Path, default=base / "cases/planner_grpo_focused_val_v3_cases.jsonl")
    parser.add_argument("--regression-cases", type=Path, default=base / "cases/planner_grpo_compound245_eval_cases.jsonl")
    parser.add_argument("--train-steps", type=Path, default=base / "sft_data_v3_chatml/train.jsonl")
    parser.add_argument("--dev-steps", type=Path, default=base / "sft_data_v3_chatml/val.jsonl")
    parser.add_argument("--hard-v4-steps", type=Path, default=base / "sft_data_v4_hard_chatml/train.jsonl")
    parser.add_argument("--hard-v5-steps", type=Path, default=base / "sft_data_v5_mixed_hard_chatml/train.jsonl")
    return parser.parse_args()


def render_card(manifest: dict) -> str:
    stats = manifest["stats"]
    splits = manifest["splits"]
    integrity = manifest["integrity"]
    lines = [
        f"# Dataset Card: {manifest['dataset_id']}",
        "",
        manifest["description"],
        "",
        "## Intended Use",
        "",
        "Training and development diagnostics for CAPA Planner routing. This dataset must not support final generalization claims.",
        "",
        "## Composition",
        "",
        f"- Source: {stats['cases']} cases and {stats['steps']} expected decisions.",
        f"- Train: {splits['train_cases']['cases']} cases and {splits['train_steps']['rows']} step rows.",
        f"- Dev: {splits['dev_cases']['cases']} cases and {splits['dev_steps']['rows']} step rows.",
        f"- Regression: {splits['regression_cases']['cases']} cases.",
        "",
        "## Integrity",
        "",
        f"- Status: `{integrity['status']}`",
        f"- Train/dev case ID overlap: {integrity['train_dev_overlap']['case_id_overlap']}",
        f"- Train/regression case ID overlap: {integrity['train_regression_overlap']['case_id_overlap']}",
        f"- Dev nearest-template median similarity: {integrity['train_dev_similarity']['median']:.4f}",
        "",
        "## Known Limitations",
        "",
    ]
    lines.extend(f"- {issue}" for issue in integrity["issues"])
    lines.extend(["", "Hashes and complete distributions are in `manifest.json`.", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    manifest = build_manifest(
        root=ROOT,
        dataset_id=args.dataset_id,
        source_cases=args.source_cases,
        train_cases=args.train_cases,
        dev_cases=args.dev_cases,
        regression_cases=args.regression_cases,
        train_steps=args.train_steps,
        dev_steps=args.dev_steps,
        hard_v4_steps=args.hard_v4_steps,
        hard_v5_steps=args.hard_v5_steps,
    )
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "DATASET_CARD.md").write_text(render_card(manifest), encoding="utf-8")
    print(json.dumps({"status": "registered", "dataset_id": args.dataset_id}, ensure_ascii=False))


if __name__ == "__main__":
    main()
