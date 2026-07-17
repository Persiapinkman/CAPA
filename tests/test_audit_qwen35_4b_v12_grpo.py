import json
from pathlib import Path

from training.planner_grpo_seed_v1.scripts.audit_qwen35_4b_v12_grpo import (
    EXPECTED_REWARD_WEIGHTS,
    audit_v12,
)


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path, *, safety_weight=0.2, include_safety=True):
    run = tmp_path / "run"
    checkpoint = run / "checkpoint-2"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    log = {
        "step": 1,
        "reward": 0.7,
        "reward_std": 0.2,
        "reward_min": 0.2,
        "reward_max": 1.0,
        "policy_entropy": 0.1,
        "grad_norm": 0.5,
        "advantage/mean": 0.0,
        "advantage/abs_mean": 0.4,
        "advantage/std": 0.5,
        "advantage/positive_fraction": 0.5,
        "completions/clipped_ratio": 0.0,
    }
    if include_safety:
        log["rewards/no_forbidden_action/mean"] = 0.8
        log["rewards/no_forbidden_action/std"] = 0.2
    weights = dict(EXPECTED_REWARD_WEIGHTS)
    weights["no_forbidden_action"] = safety_weight
    _write_json(run / "capa_qwen35_grpo_result.json", {"status": "completed", "optimizer_steps": 1})
    _write_json(
        run / "capa_qwen35_grpo_config.json",
        {
            "dataset": {"dataset_id": "planner_retry_optimizer_matched_v12"},
            "reward_weights": weights,
            "optimization": {"learning_rate": 1e-6, "warmup_steps": 1},
            "generation": {"temperature": 0.9, "top_p": 0.9},
            "source_control": {"tracked_worktree_dirty": False},
            "observability": {"wandb": {"enabled": True}},
        },
    )
    _write_json(checkpoint / "trainer_state.json", {"log_history": [log]})
    telemetry = []
    for rank in range(4):
        telemetry.extend(
            [
                {"event": "pre_optimizer_finite_gradient", "missing_gradient_tensors": 0, "nonfinite_gradient_tensors": 0},
                {"event": "optimizer_step_end"},
                {"event": "memory", "memory": {"max_allocated_gib": 10, "device_free_gib": 20}},
            ]
        )
        telemetry.extend({"event": "g3_distribution", "diverse_output_groups": 1} for _ in range(8))
    path = run / "telemetry/rank0.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in telemetry), encoding="utf-8")
    return run


def test_v12_audit_accepts_frozen_recipe_and_safety_telemetry(tmp_path):
    run = _fixture(tmp_path)
    result = audit_v12(
        run_dir=run, expected_steps=1, world_size=4, candidate_checkpoints=(2,)
    )
    assert result["status"] == "pass"
    assert result["observed"]["mean_safety_reward"] == 0.8


def test_v12_audit_rejects_missing_metric_or_changed_weight(tmp_path):
    run = _fixture(tmp_path, safety_weight=0.0, include_safety=False)
    result = audit_v12(
        run_dir=run, expected_steps=1, world_size=4, candidate_checkpoints=(2,)
    )
    assert result["status"] == "fail"
    assert not result["checks"]["reward_weights_frozen"]
    assert not result["checks"]["safety_metrics_complete"]
