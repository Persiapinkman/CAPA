import json
import os
from collections import defaultdict
from pathlib import Path

import pytest

from scripts.setup_wandb_dashboard import load_spec
from training.wandb_observability import (
    GRPO_REQUIRED_METRICS,
    advantage_statistics,
    configure_wandb_environment,
    metric_contract,
    mirror_policy_entropy_metric,
    parse_report_to,
    weighted_reward_statistics,
)


ROOT = Path(__file__).resolve().parents[1]


def test_parse_report_to_normalizes_and_deduplicates() -> None:
    assert parse_report_to("wandb,tensorboard,wandb") == ["wandb", "tensorboard"]
    assert parse_report_to(["none", " wandb "]) == ["wandb"]
    assert parse_report_to(None) == []


def test_all_reporting_backend_enables_wandb_environment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    settings = configure_wandb_environment(
        report_to="all",
        output_dir=tmp_path,
        stage="sft",
        run_name="all-integrations",
    )
    assert settings["enabled"] is True
    assert settings["job_type"] == "sft"
    assert settings["local_dir"] == str(tmp_path / "wandb")
    assert os.environ["WANDB_DIR"] == str(tmp_path)


def test_wandb_environment_is_safe_to_persist(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "secret-that-must-not-be-returned")
    settings = configure_wandb_environment(
        report_to="wandb",
        output_dir=tmp_path,
        stage="grpo",
        run_name="unit-run",
        project="unit-project",
        entity="unit-entity",
        group="unit-group",
        tags="one,two",
        mode="offline",
    )

    assert settings["api_key_present"] is True
    assert "secret-that-must-not-be-returned" not in json.dumps(settings)
    assert settings["tags"] == ["one", "two", "capa", "post-training", "grpo"]
    assert settings["local_dir"] == str(tmp_path / "wandb")
    assert settings["step_metric"] == "train/global_step"
    assert settings["enabled"] is True


def test_policy_entropy_is_mirrored_before_trl_flush() -> None:
    metrics: defaultdict[str, list[float]] = defaultdict(list)
    metrics["entropy"].extend([0.1, 0.2])
    mirror_policy_entropy_metric(metrics)
    mirror_policy_entropy_metric(metrics)
    assert metrics["policy_entropy"] == [0.1, 0.2]


def test_advantage_statistics_expose_center_and_signal_size() -> None:
    metrics = advantage_statistics([-2.0, 0.0, 2.0, 4.0])
    assert metrics["advantage/mean"] == pytest.approx(1.0)
    assert metrics["advantage/abs_mean"] == pytest.approx(2.0)
    assert metrics["advantage/std"] == pytest.approx(5**0.5)
    assert metrics["advantage/positive_fraction"] == pytest.approx(0.5)


def test_weighted_reward_statistics_match_trl_weighted_sum() -> None:
    metrics = weighted_reward_statistics(
        {
            "task_reward": [1.0, 0.5, None],
            "format_reward": [0.0, 1.0, 0.5],
        },
        {"task_reward": 0.95, "format_reward": 0.05},
    )
    assert metrics == pytest.approx({"reward_min": 0.025, "reward_max": 0.95})


def test_dashboard_contains_required_training_sections_and_metrics() -> None:
    spec = load_spec(ROOT / "configs/wandb/post_training_v1.json")
    sections = {section["name"]: section for section in spec["sections"]}
    assert {
        "Train Reward Statistics",
        "Policy Entropy",
        "Gradient Norm",
        "Mean Advantage Estimate",
    }.issubset(sections)

    dashboard_metrics = {
        metric
        for section in spec["sections"]
        for panel in section["panels"]
        for metric in panel["y"]
    }
    assert set(GRPO_REQUIRED_METRICS).issubset(dashboard_metrics)
    assert set(metric_contract("grpo")["required_metrics"]).issubset(dashboard_metrics)


def test_metric_contract_rejects_unknown_stage() -> None:
    with pytest.raises(ValueError, match="unsupported post-training stage"):
        metric_contract("pretrain")
