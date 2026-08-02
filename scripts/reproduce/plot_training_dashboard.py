#!/usr/bin/env python3
"""Render an offline training dashboard (SFT + GRPO) from HF Trainer
``trainer_state.json`` files. No wandb, no server; PNGs only.

Usage:
    python scripts/reproduce/plot_training_dashboard.py \
        --sft   capa_h20/artifacts/CAPA/repro_h20/sft/20260802_155804_qwen35_4b_planner_v6_sft/checkpoint-400/trainer_state.json \
        --grpo  capa_h20/artifacts/CAPA/repro_h20/grpo/20260801_grpo_v7_seed42_r3/checkpoint-100/trainer_state.json \
        --out   reports/figures/train_dashboard_20260802
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(p: str) -> list[dict]:
    return json.load(open(p, encoding="utf-8")).get("log_history", [])


def _series(logs: list[dict], key: str) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    for r in logs:
        if key in r and r[key] is not None and "step" in r:
            xs.append(r["step"])
            ys.append(float(r[key]))
    return xs, ys


def plot_sft(sft_logs: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle("SFT training — Qwen3.5-4B on planner_retry_migrate_v7_longobs",
                 fontsize=13)

    ax = axes[0, 0]
    x, y = _series(sft_logs, "loss")
    ax.plot(x, y, color="tab:blue", lw=1.2, label="train loss")
    x2, y2 = _series(sft_logs, "eval_loss")
    if x2:
        ax.plot(x2, y2, color="tab:red", lw=1.4, marker="o", ms=4, label="eval loss")
    ax.set_yscale("log")
    ax.set_title("Loss (log-scale)")
    ax.set_xlabel("step"); ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[0, 1]
    x, y = _series(sft_logs, "mean_token_accuracy")
    ax.plot(x, y, color="tab:blue", lw=1.2, label="train token-acc")
    x2, y2 = _series(sft_logs, "eval_mean_token_accuracy")
    if x2:
        ax.plot(x2, y2, color="tab:red", lw=1.4, marker="o", ms=4, label="eval token-acc")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Token accuracy")
    ax.set_xlabel("step"); ax.set_ylabel("accuracy")
    ax.grid(True, alpha=0.3); ax.legend(loc="lower right")

    ax = axes[1, 0]
    x, y = _series(sft_logs, "grad_norm")
    ax.plot(x, y, color="tab:purple", lw=1.0)
    ax.set_yscale("log")
    ax.set_title("Gradient norm (log-scale)")
    ax.set_xlabel("step"); ax.set_ylabel("|grad|")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    x, y = _series(sft_logs, "learning_rate")
    ax.plot(x, y, color="tab:green", lw=1.2, label="lr")
    ax2 = ax.twinx()
    xe, ye = _series(sft_logs, "entropy")
    ax2.plot(xe, ye, color="tab:orange", lw=1.0, alpha=0.8, label="entropy")
    ax.set_title("Learning rate  &  policy entropy")
    ax.set_xlabel("step"); ax.set_ylabel("lr", color="tab:green")
    ax2.set_ylabel("entropy", color="tab:orange")
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_grpo(grpo_logs: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle("GRPO training — Qwen3.5-4B SFT+GRPO seed=42 (100 opt-steps)",
                 fontsize=13)

    # (0,0) Total reward + std
    ax = axes[0, 0]
    x, y = _series(grpo_logs, "reward")
    ax.plot(x, y, color="tab:blue", lw=1.4, label="reward")
    xs, ys = _series(grpo_logs, "reward_std")
    ax.fill_between(x, [a - b for a, b in zip(y, ys)],
                    [a + b for a, b in zip(y, ys)],
                    color="tab:blue", alpha=0.15, label="±1 std")
    ax.set_title("Total reward (± std)"); ax.set_xlabel("step"); ax.set_ylabel("reward")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # (0,1) Reward components
    ax = axes[0, 1]
    for k, col in [
        ("rewards/task_reward/mean", "tab:blue"),
        ("rewards/route_exact/mean", "tab:orange"),
        ("rewards/argument_exact/mean", "tab:green"),
        ("rewards/stop_exact/mean", "tab:red"),
        ("rewards/no_forbidden_action/mean", "tab:purple"),
        ("rewards/format_reward/mean", "tab:gray"),
    ]:
        x, y = _series(grpo_logs, k)
        if x:
            ax.plot(x, y, color=col, lw=1.1, label=k.replace("rewards/", "").replace("/mean", ""))
    ax.set_ylim(0.0, 1.05); ax.set_title("Reward components (mean)")
    ax.set_xlabel("step"); ax.set_ylabel("mean")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7, loc="lower right")

    # (0,2) frac_reward_zero_std  (support-gate proxy)
    ax = axes[0, 2]
    x, y = _series(grpo_logs, "frac_reward_zero_std")
    ax.plot(x, y, color="tab:red", lw=1.4)
    ax.axhline(0.5, color="k", ls=":", lw=0.8, alpha=0.6)
    ax.set_ylim(0.0, 1.02)
    ax.set_title("frac_reward_zero_std\n(lower = better; >0.9 kills GRPO)")
    ax.set_xlabel("step"); ax.set_ylabel("fraction")
    ax.grid(True, alpha=0.3)

    # (1,0) advantage stats
    ax = axes[1, 0]
    x, y = _series(grpo_logs, "advantage/abs_mean")
    ax.plot(x, y, color="tab:blue", lw=1.2, label="|advantage| mean")
    x2, y2 = _series(grpo_logs, "advantage/std")
    ax.plot(x2, y2, color="tab:orange", lw=1.2, label="advantage std")
    ax.set_title("Advantage magnitude"); ax.set_xlabel("step"); ax.set_ylabel("value")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # (1,1) advantage positive fraction
    ax = axes[1, 1]
    x, y = _series(grpo_logs, "advantage/positive_fraction")
    ax.plot(x, y, color="tab:green", lw=1.2)
    ax.axhline(0.5, color="k", ls=":", lw=0.8, alpha=0.6)
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Advantage positive fraction")
    ax.set_xlabel("step"); ax.set_ylabel("fraction")
    ax.grid(True, alpha=0.3)

    # (1,2) policy entropy
    ax = axes[1, 2]
    x, y = _series(grpo_logs, "entropy")
    ax.plot(x, y, color="tab:orange", lw=1.2)
    ax.set_title("Policy entropy")
    ax.set_xlabel("step"); ax.set_ylabel("H")
    ax.grid(True, alpha=0.3)

    # (2,0) loss + grad_norm
    ax = axes[2, 0]
    x, y = _series(grpo_logs, "loss")
    ax.plot(x, y, color="tab:blue", lw=1.2, label="loss")
    ax2 = ax.twinx()
    xg, yg = _series(grpo_logs, "grad_norm")
    ax2.plot(xg, yg, color="tab:red", lw=1.0, alpha=0.7, label="grad_norm")
    ax.set_title("Loss & grad norm")
    ax.set_xlabel("step"); ax.set_ylabel("loss", color="tab:blue")
    ax2.set_ylabel("grad_norm", color="tab:red")
    ax.grid(True, alpha=0.3)

    # (2,1) completion length
    ax = axes[2, 1]
    x_m, y_m = _series(grpo_logs, "completions/mean_length")
    x_a, y_a = _series(grpo_logs, "completions/max_length")
    x_i, y_i = _series(grpo_logs, "completions/min_length")
    ax.plot(x_m, y_m, color="tab:blue", lw=1.2, label="mean")
    ax.fill_between(x_m, y_i, y_a, color="tab:blue", alpha=0.15, label="[min, max]")
    ax.set_title("Completion length (tokens)")
    ax.set_xlabel("step"); ax.set_ylabel("tokens")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # (2,2) step time
    ax = axes[2, 2]
    x, y = _series(grpo_logs, "step_time")
    ax.plot(x, y, color="tab:gray", lw=1.0)
    ax.set_title("Step time (s)")
    ax.set_xlabel("step"); ax.set_ylabel("seconds")
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True, help="SFT trainer_state.json")
    ap.add_argument("--grpo", required=True, help="GRPO trainer_state.json")
    ap.add_argument("--out", required=True, help="output dir (created if missing)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sft_logs = _load(args.sft)
    grpo_logs = _load(args.grpo)
    print(f"SFT log entries : {len(sft_logs)}")
    print(f"GRPO log entries: {len(grpo_logs)}")

    sft_png = out_dir / "sft_dashboard.png"
    grpo_png = out_dir / "grpo_dashboard.png"
    plot_sft(sft_logs, sft_png)
    plot_grpo(grpo_logs, grpo_png)
    print("wrote", sft_png)
    print("wrote", grpo_png)


if __name__ == "__main__":
    main()
