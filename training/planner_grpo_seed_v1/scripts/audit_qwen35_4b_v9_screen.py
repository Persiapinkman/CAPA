#!/usr/bin/env python3
"""Audit a Qwen3.5-4B GRPO screen before any checkpoint is evaluated."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


REQUIRED_LOG_KEYS = (
    "reward",
    "reward_std",
    "reward_min",
    "reward_max",
    "policy_entropy",
    "grad_norm",
    "advantage/mean",
    "advantage/abs_mean",
    "advantage/std",
    "advantage/positive_fraction",
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_screen(
    *,
    result: dict[str, Any],
    config: dict[str, Any],
    trainer_state: dict[str, Any],
    telemetry: list[dict[str, Any]],
    run_dir: Path,
    expected_steps: int = 40,
    world_size: int = 4,
    candidate_checkpoints: tuple[int, ...] = (10, 20, 40),
) -> dict[str, Any]:
    step_logs = [
        row for row in trainer_state.get("log_history", [])
        if isinstance(row, dict) and row.get("reward") is not None
    ]
    missing_metric_values = [
        {"step": row.get("step"), "metric": key}
        for row in step_logs
        for key in REQUIRED_LOG_KEYS
        if row.get(key) is None
    ]
    nonfinite_metric_values = [
        {"step": row.get("step"), "metric": key, "value": row.get(key)}
        for row in step_logs
        for key in REQUIRED_LOG_KEYS
        if row.get(key) is not None and not math.isfinite(float(row[key]))
    ]
    gradient_events = [row for row in telemetry if row.get("event") == "pre_optimizer_finite_gradient"]
    optimizer_events = [row for row in telemetry if row.get("event") == "optimizer_step_end"]
    generation_events = [row for row in telemetry if row.get("event") == "g3_distribution"]
    memory_values = [
        row["memory"] for row in telemetry if isinstance(row.get("memory"), dict)
    ]
    checkpoint_paths = {
        checkpoint: run_dir / f"checkpoint-{checkpoint}/adapter_model.safetensors"
        for checkpoint in candidate_checkpoints
    }
    checkpoints = {
        checkpoint: path.is_file() for checkpoint, path in checkpoint_paths.items()
    }
    checks = {
        "result_completed": result.get("status") == "completed",
        "optimizer_steps": int(result.get("optimizer_steps") or 0) == expected_steps,
        "logged_steps": len(step_logs) == expected_steps,
        "required_metrics_complete": not missing_metric_values,
        "metrics_finite": not nonfinite_metric_values,
        "gradient_event_count": len(gradient_events) == expected_steps * world_size,
        "optimizer_event_count": len(optimizer_events) == expected_steps * world_size,
        "generation_event_count": len(generation_events) == expected_steps * 8 * world_size,
        "no_missing_gradients": not any(int(row.get("missing_gradient_tensors") or 0) for row in gradient_events),
        "no_nonfinite_gradients": not any(int(row.get("nonfinite_gradient_tensors") or 0) for row in gradient_events),
        "memory_peak": max((float(row["max_allocated_gib"]) for row in memory_values), default=0.0) <= 28.0,
        "memory_free": min((float(row["device_free_gib"]) for row in memory_values), default=0.0) >= 2.0,
        "no_completion_clipping": not any(float(row.get("completions/clipped_ratio") or 0) > 0.01 for row in step_logs),
        "candidate_checkpoints_present": all(checkpoints.values()),
        "source_commit_clean": config.get("source_control", {}).get("tracked_worktree_dirty") is False,
        "wandb_enabled": config.get("observability", {}).get("wandb", {}).get("enabled") is True,
    }
    return {
        "schema_version": "1.0",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "observed": {
            "logged_steps": len(step_logs),
            "gradient_events": len(gradient_events),
            "optimizer_events": len(optimizer_events),
            "generation_events": len(generation_events),
            "diverse_generation_groups": sum(int(row.get("diverse_output_groups") or 0) for row in generation_events),
            "maximum_allocated_gib": max((float(row["max_allocated_gib"]) for row in memory_values), default=0.0),
            "minimum_free_gib": min((float(row["device_free_gib"]) for row in memory_values), default=0.0),
            "zero_gradient_steps_per_rank": sum(float(row.get("grad_norm") or 0) == 0 for row in step_logs),
            "missing_metric_values": missing_metric_values,
            "nonfinite_metric_values": nonfinite_metric_values,
            "checkpoints": checkpoints,
            "checkpoint_adapter_sha256": {
                checkpoint: sha256_file(path)
                for checkpoint, path in checkpoint_paths.items()
                if path.is_file()
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=40)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument(
        "--candidate-checkpoint",
        type=int,
        action="append",
        dest="candidate_checkpoints",
        help="Required saved checkpoint; repeat for every selection candidate",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_checkpoints = tuple(args.candidate_checkpoints or (10, 20, 40))
    if args.expected_steps <= 0:
        raise ValueError("expected steps must be positive")
    if args.world_size <= 0:
        raise ValueError("world size must be positive")
    if not candidate_checkpoints or any(step <= 0 for step in candidate_checkpoints):
        raise ValueError("candidate checkpoints must be positive")
    if len(set(candidate_checkpoints)) != len(candidate_checkpoints):
        raise ValueError("candidate checkpoints must be unique")
    final_checkpoint = max(candidate_checkpoints)
    telemetry = [
        row
        for path in sorted((args.run_dir / "telemetry").glob("rank*.jsonl"))
        for row in load_jsonl(path)
    ]
    payload = audit_screen(
        result=load_json(args.run_dir / "capa_qwen35_grpo_result.json"),
        config=load_json(args.run_dir / "capa_qwen35_grpo_config.json"),
        trainer_state=load_json(
            args.run_dir / f"checkpoint-{final_checkpoint}/trainer_state.json"
        ),
        telemetry=telemetry,
        run_dir=args.run_dir,
        expected_steps=args.expected_steps,
        world_size=args.world_size,
        candidate_checkpoints=candidate_checkpoints,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
