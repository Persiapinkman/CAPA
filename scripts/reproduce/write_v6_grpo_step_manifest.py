#!/usr/bin/env python3
"""Write the `.manifest.json` sidecar for planner_retry_migrate_v6 GRPO step data.

The upstream trainer (``scripts/run_qwen35_4b_grpo_v5_train_v1.sh``) refuses to
launch unless ``<step_data>.manifest.json`` exists. The V6 dataset builder emits
the step-data JSONL but not the per-file manifest sidecar (that was V5-only).

We re-derive the sidecar from committed evidence:

- rows/prompt token stats from the JSONL itself,
- dataset_id + file sha256 from ``data/datasets/planner_retry_migrate_v6/manifest.json``,
- tokenizer / config / chat_template sha256 from the local model directory
  (defaults to the H20 model location under this repo tree).

Rerunning the script is idempotent: if the sidecar already exists and matches,
it is left untouched; if it disagrees, the script exits non-zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL = Path(
    "/apdcephfs_hzlf/share_1227201/zkq/capa_h20/models/Qwen3.5-4B"
)
DEFAULT_DATASET_MANIFEST = ROOT / "data/datasets/planner_retry_migrate_v6/manifest.json"

DEFAULT_STEP_FILES = {
    "grpo_train": ROOT
    / "training/planner_grpo_seed_v1/step_data/planner_retry_migrate_v6_grpo_train_qwen35_4b_nothinking_step2.jsonl",
    "grpo_dev": ROOT
    / "training/planner_grpo_seed_v1/step_data/planner_retry_migrate_v6_grpo_dev_qwen35_4b_nothinking_step2.jsonl",
}

EXPECTED_DATASET_ID = "planner_retry_migrate_v6"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def percentile(sorted_values: list[int], q: float) -> int:
    if not sorted_values:
        return 0
    idx = min(len(sorted_values) - 1, int((len(sorted_values) - 1) * q))
    return sorted_values[idx]


def build_manifest(
    *,
    step_jsonl: Path,
    cases_jsonl: Path,
    model_dir: Path,
    dataset_manifest: dict[str, Any],
    max_prompt_tokens: int = 4608,
) -> dict[str, Any]:
    rows = load_jsonl(step_jsonl)
    if not rows:
        raise ValueError(f"empty step data: {step_jsonl}")

    for row in rows:
        if row.get("dataset_id") != EXPECTED_DATASET_ID:
            raise ValueError(
                f"row dataset_id mismatch in {step_jsonl.name}: "
                f"{row.get('dataset_id')} != {EXPECTED_DATASET_ID}"
            )

    prompt_tokens = sorted(int(r["prompt_token_count"]) for r in rows)
    over_limit = sum(1 for t in prompt_tokens if t > max_prompt_tokens)

    action_counts: Counter[str] = Counter()
    scenario_counts: Counter[str] = Counter()
    for row in rows:
        try:
            expected_step = json.loads(row["expected_step"])
            action_counts[str(expected_step.get("action") or expected_step.get("decision_type") or "")] += 1
        except (KeyError, json.JSONDecodeError):
            pass
        scenario_counts[str(row.get("scenario_id") or "")] += 1

    tokenizer_config = model_dir / "tokenizer_config.json"
    config_json = model_dir / "config.json"
    chat_template = model_dir / "chat_template.jinja"
    for required in (tokenizer_config, config_json, chat_template):
        if not required.exists():
            raise FileNotFoundError(f"model asset missing: {required}")

    return {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": EXPECTED_DATASET_ID,
        "role": "optimization_only_qwen35_step_data",
        "model_name_or_path": str(model_dir),
        "cases": str(cases_jsonl),
        "output": str(step_jsonl),
        "rows": len(rows),
        "target_step": 2,
        "prompt_contract": {
            "chat_template": "native_qwen35",
            "enable_thinking": False,
            "suffix": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
            "eos_token_id": 248046,
            "pad_token_id": 248044,
            "max_prompt_tokens_hard_gate": max_prompt_tokens,
            "token_stats": {
                "min": prompt_tokens[0],
                "mean": sum(prompt_tokens) / len(prompt_tokens),
                "p50": percentile(prompt_tokens, 0.50),
                "p95": percentile(prompt_tokens, 0.95),
                "p99": percentile(prompt_tokens, 0.99),
                "max": prompt_tokens[-1],
                "over_limit": over_limit,
            },
        },
        "distribution": {
            "expected_actions": dict(sorted(action_counts.items())),
            "scenarios": dict(sorted(scenario_counts.items())),
        },
        "sha256": {
            "cases": sha256_file(cases_jsonl),
            "step_data": sha256_file(step_jsonl),
            "tokenizer_config": sha256_file(tokenizer_config),
            "config": sha256_file(config_json),
            "chat_template": sha256_file(chat_template),
        },
        "dataset_manifest_dataset_id": dataset_manifest.get("dataset_id"),
    }


def compare_or_write(manifest: dict[str, Any], path: Path) -> str:
    """Idempotent write. Returns 'wrote' | 'kept' | 'mismatch'."""

    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        # Compare on the fields that gate the trainer.
        keys = ("dataset_id", "rows", "target_step", "sha256")
        if all(existing.get(k) == manifest.get(k) for k in keys):
            return "kept"
        return "mismatch"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return "wrote"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=DEFAULT_DATASET_MANIFEST,
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=4608,
    )
    parser.add_argument(
        "--split",
        choices=("grpo_train", "grpo_dev", "both"),
        default="both",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing sidecar even if it matches on gate fields.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    if dataset_manifest.get("dataset_id") != EXPECTED_DATASET_ID:
        raise ValueError(
            f"dataset manifest is for {dataset_manifest.get('dataset_id')!r}, "
            f"expected {EXPECTED_DATASET_ID!r}"
        )

    splits = ("grpo_train", "grpo_dev") if args.split == "both" else (args.split,)
    results: dict[str, str] = {}
    for split in splits:
        step_path = DEFAULT_STEP_FILES[split]
        files = dataset_manifest.get("files", {})
        cases_key = "cases_" + ("grpo_train" if split == "grpo_train" else "grpo_dev")
        cases_path = ROOT / files[cases_key]
        if not step_path.exists():
            raise FileNotFoundError(f"step data missing: {step_path}")
        if not cases_path.exists():
            raise FileNotFoundError(f"cases missing: {cases_path}")

        manifest = build_manifest(
            step_jsonl=step_path,
            cases_jsonl=cases_path,
            model_dir=args.model_dir,
            dataset_manifest=dataset_manifest,
            max_prompt_tokens=args.max_prompt_tokens,
        )
        sidecar = step_path.with_suffix(".manifest.json")

        if args.force and sidecar.exists():
            sidecar.unlink()
        status = compare_or_write(manifest, sidecar)
        results[split] = f"{status}: {sidecar}"
        if status == "mismatch":
            print(f"[write_v6_grpo_step_manifest] {split}: existing sidecar disagrees; "
                  f"rerun with --force to overwrite. path={sidecar}", file=sys.stderr)
            return 2

    for split, msg in results.items():
        print(f"[write_v6_grpo_step_manifest] {split}: {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
