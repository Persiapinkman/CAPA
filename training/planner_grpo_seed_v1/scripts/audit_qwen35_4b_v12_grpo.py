#!/usr/bin/env python3
"""Audit V12 GRPO runtime health, recipe integrity, and safety telemetry."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.planner_grpo_seed_v1.scripts import (  # noqa: E402
    audit_qwen35_4b_v9_screen as base,
)


EXPECTED_REWARD_WEIGHTS = {
    "task_reward": 0.75,
    "format_reward": 0.05,
    "route_exact": 0.0,
    "argument_exact": 0.0,
    "stop_exact": 0.0,
    "no_forbidden_action": 0.2,
}
SAFETY_KEYS = (
    "rewards/no_forbidden_action/mean",
    "rewards/no_forbidden_action/std",
)


def audit_v12(
    *,
    run_dir: Path,
    expected_steps: int,
    world_size: int,
    candidate_checkpoints: tuple[int, ...],
) -> dict:
    result = base.load_json(run_dir / "capa_qwen35_grpo_result.json")
    config = base.load_json(run_dir / "capa_qwen35_grpo_config.json")
    final_checkpoint = max(candidate_checkpoints)
    trainer_state = base.load_json(
        run_dir / f"checkpoint-{final_checkpoint}/trainer_state.json"
    )
    telemetry = [
        row
        for path in sorted((run_dir / "telemetry").glob("rank*.jsonl"))
        for row in base.load_jsonl(path)
    ]
    payload = base.audit_screen(
        result=result,
        config=config,
        trainer_state=trainer_state,
        telemetry=telemetry,
        run_dir=run_dir,
        expected_steps=expected_steps,
        world_size=world_size,
        candidate_checkpoints=candidate_checkpoints,
    )
    step_logs = [
        row
        for row in trainer_state.get("log_history", [])
        if isinstance(row, dict) and row.get("reward") is not None
    ]
    safety_missing = [
        {"step": row.get("step"), "metric": key}
        for row in step_logs
        for key in SAFETY_KEYS
        if row.get(key) is None
    ]
    safety_nonfinite = [
        {"step": row.get("step"), "metric": key, "value": row.get(key)}
        for row in step_logs
        for key in SAFETY_KEYS
        if row.get(key) is not None and not math.isfinite(float(row[key]))
    ]
    weights = {
        key: float(config.get("reward_weights", {}).get(key, float("nan")))
        for key in EXPECTED_REWARD_WEIGHTS
    }
    recipe_checks = {
        "v12_dataset": config.get("dataset", {}).get("dataset_id")
        == "planner_retry_optimizer_matched_v12",
        "reward_weights_frozen": weights == EXPECTED_REWARD_WEIGHTS,
        "learning_rate_frozen": float(
            config.get("optimization", {}).get("learning_rate") or 0.0
        )
        == 1e-6,
        "warmup_frozen": int(config.get("optimization", {}).get("warmup_steps") or 0)
        == 1,
        "sampling_frozen": (
            float(config.get("generation", {}).get("temperature") or 0.0) == 0.9
            and float(config.get("generation", {}).get("top_p") or 0.0) == 0.9
        ),
        "safety_metrics_complete": not safety_missing,
        "safety_metrics_finite": not safety_nonfinite,
    }
    payload["checks"].update(recipe_checks)
    payload["observed"]["reward_weights"] = weights
    payload["observed"]["safety_metric_missing"] = safety_missing
    payload["observed"]["safety_metric_nonfinite"] = safety_nonfinite
    payload["observed"]["mean_safety_reward"] = (
        sum(float(row[SAFETY_KEYS[0]]) for row in step_logs) / len(step_logs)
        if step_logs and not safety_missing
        else None
    )
    payload["observed"]["mean_safety_reward_std"] = (
        sum(float(row[SAFETY_KEYS[1]]) for row in step_logs) / len(step_logs)
        if step_logs and not safety_missing
        else None
    )
    payload["status"] = "pass" if all(payload["checks"].values()) else "fail"
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument(
        "--candidate-checkpoint",
        type=int,
        action="append",
        dest="candidate_checkpoints",
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = tuple(args.candidate_checkpoints)
    if args.expected_steps <= 0 or args.world_size <= 0:
        raise ValueError("expected steps and world size must be positive")
    if not candidates or any(step <= 0 for step in candidates):
        raise ValueError("candidate checkpoints must be positive")
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate checkpoints must be unique")
    payload = audit_v12(
        run_dir=args.run_dir,
        expected_steps=args.expected_steps,
        world_size=args.world_size,
        candidate_checkpoints=candidates,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
