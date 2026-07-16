"""Shared W&B metric and workspace contracts for CAPA post-training.

This module deliberately has no dependency on :mod:`wandb`. Training scripts
can therefore validate configs and run in ``--mode dry-run`` before the W&B
optional dependency or credentials are present.
"""

from __future__ import annotations

import math
import os
import statistics
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any


DEFAULT_WANDB_PROJECT = "capa-planner-post-training"
DEFAULT_WANDB_GROUP = "planner-retry-migrate-v6"

WANDB_STEP_METRIC = "train/global_step"
SFT_REQUIRED_METRICS = (
    "train/loss",
    "eval/loss",
    "train/policy_entropy",
    "eval/policy_entropy",
    "train/grad_norm",
    "train/mean_token_accuracy",
    "eval/mean_token_accuracy",
)
GRPO_REQUIRED_METRICS = (
    "train/reward",
    "train/reward_std",
    "train/reward_min",
    "train/reward_max",
    "train/policy_entropy",
    "train/grad_norm",
    "train/advantage/mean",
    "train/advantage/abs_mean",
    "train/advantage/std",
    "train/advantage/positive_fraction",
)


def parse_report_to(value: str | Sequence[str] | None) -> list[str]:
    """Normalize a CLI/HF reporting value into explicit backend names."""

    if value is None:
        return []
    raw = [value] if isinstance(value, str) else list(value)
    backends: list[str] = []
    for item in raw:
        for candidate in str(item).split(","):
            normalized = candidate.strip().lower()
            if not normalized or normalized in {"none", "null", "disabled"}:
                continue
            if normalized not in backends:
                backends.append(normalized)
    return backends


def configure_wandb_environment(
    *,
    report_to: str | Sequence[str] | None,
    output_dir: Path,
    stage: str,
    run_name: str,
    project: str = DEFAULT_WANDB_PROJECT,
    entity: str = "",
    group: str = DEFAULT_WANDB_GROUP,
    tags: str | Sequence[str] = (),
    mode: str = "online",
) -> dict[str, Any]:
    """Configure W&B through documented environment variables.

    The return value is safe to persist in a run config; it never includes an
    API key. Explicit CLI values take precedence over inherited environment
    settings so an immutable run record describes the effective destination.
    """

    backends = parse_report_to(report_to)
    enabled = "wandb" in backends or "all" in backends
    if isinstance(tags, str):
        tag_values = [part.strip() for part in tags.split(",") if part.strip()]
    else:
        tag_values = [str(part).strip() for part in tags if str(part).strip()]
    for required in ("capa", "post-training", stage):
        if required not in tag_values:
            tag_values.append(required)

    settings = {
        "enabled": enabled,
        "report_to": backends,
        "project": project,
        "entity": entity,
        "group": group,
        "run_name": run_name,
        "job_type": stage,
        "mode": mode,
        "tags": tag_values,
        "local_dir": str(output_dir / "wandb"),
        "step_metric": WANDB_STEP_METRIC,
        "api_key_present": bool(os.environ.get("WANDB_API_KEY")),
    }
    if not enabled:
        return settings

    os.environ["WANDB_PROJECT"] = project
    if entity:
        os.environ["WANDB_ENTITY"] = entity
    os.environ["WANDB_RUN_GROUP"] = group
    os.environ["WANDB_NAME"] = run_name
    os.environ["WANDB_JOB_TYPE"] = stage
    os.environ["WANDB_TAGS"] = ",".join(tag_values)
    os.environ["WANDB_MODE"] = mode
    # W&B creates its own ``wandb/`` child below WANDB_DIR.
    os.environ["WANDB_DIR"] = str(output_dir)
    # Checkpoints remain in the audited artifact store; do not duplicate them
    # into W&B unless a future protocol explicitly opts in.
    os.environ.setdefault("WANDB_LOG_MODEL", "false")
    os.environ.setdefault("WANDB_WATCH", "false")
    return settings


def mirror_policy_entropy_metric(
    metrics: MutableMapping[str, list[float]],
) -> None:
    """Mirror TRL's buffered ``entropy`` metric before its log flush.

    TRL adds its internally accumulated metrics inside ``Trainer.log``.  The
    alias therefore has to be inserted into the metric buffer before calling
    the parent implementation, rather than added to the incoming log record.
    """

    entropy = metrics.get("entropy")
    if entropy and not metrics.get("policy_entropy"):
        metrics["policy_entropy"] = list(entropy)


def advantage_statistics(values: Sequence[float]) -> dict[str, float]:
    """Return interpretable GRPO advantage diagnostics.

    Group-normalized GRPO centers the raw mean close to zero by construction;
    absolute mean and standard deviation carry most of the signal.
    """

    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {}
    return {
        "advantage/mean": statistics.fmean(finite),
        "advantage/abs_mean": statistics.fmean(abs(value) for value in finite),
        "advantage/std": statistics.pstdev(finite),
        "advantage/positive_fraction": sum(value > 0 for value in finite) / len(finite),
    }


def weighted_reward_statistics(
    rewards_by_name: Mapping[str, Sequence[float | None]],
    weights_by_name: Mapping[str, float],
) -> dict[str, float]:
    """Compute aggregate reward range using the same weighted-sum semantics as TRL."""

    if not rewards_by_name:
        return {}
    lengths = {len(values) for values in rewards_by_name.values()}
    if len(lengths) != 1:
        raise ValueError(f"reward columns have inconsistent lengths: {sorted(lengths)}")
    row_count = next(iter(lengths), 0)
    totals: list[float] = []
    for index in range(row_count):
        total = 0.0
        scorable = False
        for name, values in rewards_by_name.items():
            value = values[index]
            if value is None or not math.isfinite(float(value)):
                continue
            scorable = True
            total += float(weights_by_name.get(name, 0.0)) * float(value)
        if scorable:
            totals.append(total)
    if not totals:
        return {}
    return {
        "reward_min": min(totals),
        "reward_max": max(totals),
    }


def metric_contract(stage: str) -> dict[str, Any]:
    """Return the persisted W&B metric contract for a training stage."""

    if stage not in {"sft", "grpo"}:
        raise ValueError(f"unsupported post-training stage: {stage}")
    required = SFT_REQUIRED_METRICS if stage == "sft" else GRPO_REQUIRED_METRICS
    return {
        "stage": stage,
        "step_metric": WANDB_STEP_METRIC,
        "required_metrics": list(required),
        "advantage_note": (
            "not defined for supervised fine-tuning"
            if stage == "sft"
            else "raw group-normalized mean is expected to remain near zero; inspect abs_mean and std"
        ),
    }
