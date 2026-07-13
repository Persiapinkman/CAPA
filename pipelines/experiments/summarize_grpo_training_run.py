#!/usr/bin/env python3
"""Create a versioned training run record from a completed TRL GRPO output."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from capa.experiments.registry import current_git_commit, sha256_file  # noqa: E402
from training.planner_grpo_seed_v1.scripts.train_planner_grpo_trl import (  # noqa: E402
    summarize_log_history,
)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def latest_trainer_state(output_dir: Path) -> Path:
    checkpoints = []
    for path in output_dir.glob("checkpoint-*/trainer_state.json"):
        try:
            step = int(path.parent.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        checkpoints.append((step, path))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint trainer_state.json under {output_dir}")
    return max(checkpoints)[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--parent-run-id", default="")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    run_dir = resolve(args.run_dir)
    config_path = output_dir / "capa_trl_grpo_config.json"
    result_path = output_dir / "capa_trl_grpo_result.json"
    trainer_state_path = latest_trainer_state(output_dir)
    config = load(config_path)
    trainer_state = load(trainer_state_path)
    log_summary = summarize_log_history(list(trainer_state.get("log_history") or []))
    if result_path.is_file():
        result = load(result_path)
    else:
        step_rows = [
            row
            for row in trainer_state.get("log_history") or []
            if isinstance(row, dict) and row.get("reward") is not None
        ]
        result = {
            "started_at": config["started_at"],
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "train_metrics": {
                "train_runtime": sum(float(row.get("step_time") or 0.0) for row in step_rows),
                "train_loss": (
                    sum(float(row.get("loss") or 0.0) for row in step_rows) / len(step_rows)
                    if step_rows
                    else None
                ),
                "checkpoint_step": int(trainer_state_path.parent.name.rsplit("-", 1)[1]),
            },
        }
    cases_path = Path(config["cases"])
    if not cases_path.is_absolute():
        cases_path = resolve(cases_path)
    step_data_path = Path(config["step_data"]) if config.get("step_data") else None
    if step_data_path is not None and not step_data_path.is_absolute():
        step_data_path = resolve(step_data_path)
    data_files = {"cases": str(cases_path)}
    data_hashes = {"cases": sha256_file(cases_path)}
    if step_data_path is not None:
        data_files["steps"] = str(step_data_path)
        data_hashes["steps"] = sha256_file(step_data_path)
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "world_size": args.world_size,
        "precision": config["training"]["precision"],
        "attention": config["training"]["attention_implementation"],
    }
    record = {
        "schema_version": "2.0",
        "run_id": args.run_id,
        "study_id": args.study_id,
        "date": str(result["finished_at"])[:10],
        "kind": "train_grpo_lora_checkpoint",
        "status": "completed",
        "purpose": args.purpose,
        "hypothesis": args.hypothesis,
        "parent_run_id": args.parent_run_id or None,
        "provenance": {
            "git_commit": current_git_commit(ROOT),
            "command": config["command"],
            "seed": config["seed"],
            "environment": environment,
            "started_at": result["started_at"],
            "finished_at": result["finished_at"],
        },
        "data": {
            "dataset_id": args.dataset_id,
            "split": "train",
            "files": data_files,
            "sha256": data_hashes,
            "rows": config["length_stats"]["rows_after"],
        },
        "method": {
            "model": config["model_name_or_path"],
            "adapter_path": str(trainer_state_path.parent),
            "training": config["training"],
            "generation": {
                "num_generations": config["num_generations"],
                "generation_batch_size": config["generation_batch_size"],
                "max_completion_length": config["max_completion_length"],
            },
            "reward_weights": config["reward_weights"],
            "lora": config["lora"],
        },
        "metrics": {
            "primary": {
                "name": "train_reward_mean",
                "value": log_summary["mean_reward"],
                "higher_is_better": True,
            },
            "train": result["train_metrics"],
            "log_summary": log_summary,
        },
        "artifacts": {
            "output_dir": str(output_dir),
            "config": str(config_path),
            "trainer_state": str(trainer_state_path),
        },
        "decision": {
            "outcome": "pending_comparison",
            "rationale": "Training completed; held-out development evaluation has not yet been applied.",
        },
    }
    if result_path.is_file():
        record["artifacts"]["result"] = str(result_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "training_summary.json").write_text(
        json.dumps(log_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "summarized", "run_record": str(run_dir / 'run_record.json')}))


if __name__ == "__main__":
    main()
