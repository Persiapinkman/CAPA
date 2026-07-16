from training.planner_grpo_seed_v1.scripts.audit_qwen35_4b_v9_screen import (
    REQUIRED_LOG_KEYS,
    audit_screen,
)


def test_screen_audit_requires_metrics_events_and_checkpoints(tmp_path):
    steps = 2
    world = 2
    for checkpoint in (10, 20, 40):
        path = tmp_path / f"checkpoint-{checkpoint}/adapter_model.safetensors"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"x")
    logs = [
        {"step": step, "completions/clipped_ratio": 0, **{key: 0.1 for key in REQUIRED_LOG_KEYS}}
        for step in range(1, steps + 1)
    ]
    telemetry = []
    for rank in range(world):
        for step in range(steps):
            telemetry.extend(
                [
                    {"event": "pre_optimizer_finite_gradient", "missing_gradient_tensors": 0, "nonfinite_gradient_tensors": 0},
                    {"event": "optimizer_step_end", "memory": {"max_allocated_gib": 12, "device_free_gib": 16}},
                ]
            )
            telemetry.extend({"event": "g3_distribution", "diverse_output_groups": 1} for _ in range(8))
    result = audit_screen(
        result={"status": "completed", "optimizer_steps": steps},
        config={"source_control": {"tracked_worktree_dirty": False}, "observability": {"wandb": {"enabled": True}}},
        trainer_state={"log_history": logs},
        telemetry=telemetry,
        run_dir=tmp_path,
        expected_steps=steps,
        world_size=world,
    )
    assert result["status"] == "pass"
    assert all(result["checks"].values())
