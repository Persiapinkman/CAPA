#!/usr/bin/env python3
"""Audit and register the stateful retrieval GRPO dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from capa.evaluation.dataset_audit import (  # noqa: E402
    case_stats,
    load_jsonl,
    nearest_same_category_similarity,
    overlap,
    step_stats,
)
from capa.experiments.registry import sha256_file  # noqa: E402


BASE = Path("training/planner_grpo_seed_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default="planner_stateful_retrieval_v1")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/datasets/planner_stateful_retrieval_v1")
    )
    for split in ("train", "dev", "test"):
        parser.add_argument(
            f"--{split}-cases",
            type=Path,
            default=BASE / f"cases/planner_stateful_retrieval_v1_{split}_cases.jsonl",
        )
        parser.add_argument(
            f"--{split}-steps",
            type=Path,
            default=BASE / f"sft_data_stateful_retrieval_v1_chatml/{split}.jsonl",
        )
    parser.add_argument(
        "--support-audit",
        type=Path,
        help="Optional fixed development subset used only for sampling-support auditing.",
    )
    return parser.parse_args()


def resolved(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def metadata_overlap(
    left: list[dict[str, Any]], right: list[dict[str, Any]], key: str
) -> int:
    return len(
        {str(row.get(key) or "") for row in left}
        & {str(row.get(key) or "") for row in right}
    )


def render_card(manifest: dict[str, Any]) -> str:
    splits = manifest["splits"]
    integrity = manifest["integrity"]
    lines = [
        f"# Dataset Card: {manifest['dataset_id']}",
        "",
        "State-conditioned Planner routing cases for retrieval, rewrite, retry, memory reuse, and guardrails.",
        "",
        "## Research Question",
        "",
        "Can GRPO improve multi-step state transitions when the policy has non-saturated sampled rewards?",
        "",
        "## Composition",
        "",
    ]
    for split in ("train", "dev", "test"):
        item = splits[split]
        lines.append(
            f"- {split.title()}: {item['cases']['cases']} cases, "
            f"{item['steps']['rows']} step rows, and {item['entity_groups']} entity groups."
        )
    lines.extend(
        [
            "",
            "## Isolation",
            "",
            f"- Train/dev entity overlap: {integrity['train_dev']['entity_overlap']}",
            f"- Train/test entity overlap: {integrity['train_test']['entity_overlap']}",
            f"- Dev/test entity overlap: {integrity['dev_test']['entity_overlap']}",
            f"- Train/test template overlap: {integrity['train_test']['template_overlap']}",
            "- Test status: `sealed_until_hyperparameters_are_locked`",
            "",
            "## Scenario Families",
            "",
            "- `coref_rewrite_then_rag`: resolve an entity from history before retrieval.",
            "- `rag_miss_rewrite_then_rag`: retrieve, rewrite after a miss, then retrieve once more.",
            "- `memory_hit_end`: reuse sufficient prior evidence without another tool call.",
            "- `direct_rag_guardrail`: retrieve directly when no rewrite is needed.",
            "- `general_answer_guardrail`: answer general knowledge without private retrieval.",
            "",
            "## Human Review",
            "",
            "Review the source case JSONL before derived ChatML rows. Audit all development cases and stratify training samples by category, step index, entity, and template. Do not inspect model predictions on the test split until hyperparameters and training seeds are locked.",
            "Use `HUMAN_REVIEW.md` for the label checklist and rejection conditions.",
            "",
            "## Known Limitations",
            "",
            "- Cases are curated templates rather than production traffic.",
            "- Tool observations are deterministic mocks; this isolates Planner policy learning, not tool reliability.",
            "- Derived prompts contain fixed opaque ledger identifiers. Training reads the registered step file directly so these nuisance tokens remain byte-identical across ranks and seeds.",
            "- Entity grouping must be respected in confidence intervals.",
            "- A positive result applies to this stateful retrieval task family until replicated on real traffic.",
            "",
            "Hashes, category distributions, and similarity diagnostics are in `manifest.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    files: dict[str, Path] = {}
    cases: dict[str, list[dict[str, Any]]] = {}
    steps: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "dev", "test"):
        case_path = resolved(getattr(args, f"{split}_cases"))
        step_path = resolved(getattr(args, f"{split}_steps"))
        files[f"{split}_cases"] = case_path
        files[f"{split}_steps"] = step_path
        cases[split] = load_jsonl(case_path)
        steps[split] = load_jsonl(step_path)
    if args.support_audit is not None:
        files["support_audit"] = resolved(args.support_audit)

    pair_integrity: dict[str, Any] = {}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        pair_integrity[f"{left}_{right}"] = {
            **overlap(cases[left], cases[right]),
            "entity_overlap": metadata_overlap(cases[left], cases[right], "entity_id"),
            "template_overlap": metadata_overlap(cases[left], cases[right], "template_id"),
        }

    manifest = {
        "schema_version": "1.0",
        "dataset_id": args.dataset_id,
        "role": "train_dev_and_sealed_test",
        "experimental_unit": "case_id",
        "bootstrap_cluster": "entity_id",
        "description": "Stateful retrieval and memory routing benchmark for GRPO learnability.",
        "splits": {
            split: {
                "cases": case_stats(cases[split]),
                "steps": step_stats(steps[split]),
                "entity_groups": len({str(row.get("entity_id") or "") for row in cases[split]}),
                "template_ids": sorted({str(row.get("template_id") or "") for row in cases[split]}),
            }
            for split in ("train", "dev", "test")
        },
        "files": {
            key: str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
            for key, path in files.items()
        },
        "sha256": {key: sha256_file(path) for key, path in files.items()},
        "integrity": {
            "status": "sealed_test_created_unopened",
            **pair_integrity,
            "dev_similarity_to_train": nearest_same_category_similarity(cases["train"], cases["dev"]),
            "test_similarity_to_train": nearest_same_category_similarity(cases["train"], cases["test"]),
        },
    }
    output_dir = resolved(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "DATASET_CARD.md").write_text(render_card(manifest), encoding="utf-8")
    print(json.dumps({"status": "registered", "dataset_id": args.dataset_id}, ensure_ascii=False))


if __name__ == "__main__":
    main()
