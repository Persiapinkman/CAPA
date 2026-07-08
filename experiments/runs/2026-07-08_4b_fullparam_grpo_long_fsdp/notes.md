# Qwen3.5-4B Full-Parameter GRPO Long Run

- Date: 2026-07-08
- Purpose: start the full-parameter FSDP GRPO long run after smoke validation.
- Base model: `/mnt/zkq/models/Qwen3.5-4B`
- Base backup: `/mnt/zkq/models/Qwen3.5-4B.backup-20260708`
- Environment: `.venv-train-cu124` (`torch 2.6.0+cu124`)
- GPUs: `CUDA_VISIBLE_DEVICES=4,5,6,7`
- Dataset: `training/planner_grpo_seed_v1/cases/planner_grpo_focused_4b_cases.jsonl`
- Output: `outputs/planner-grpo-qwen35-4b-focused-full-fsdp-long-20260708`
- Log: `experiments/runs/2026-07-08_4b_fullparam_grpo_long_fsdp/logs/grpo_fullparam_fsdp.log`

## Launch Settings

- Full parameters, no LoRA: `--use-lora false`
- Distributed strategy: Accelerate FSDP full-shard
- Precision: fp16
- FSDP activation checkpointing: enabled
- Trainer/model gradient checkpointing: disabled
- Epochs: 1
- Max steps: `-1` (full epoch)
- Per-device train batch size: 1
- Gradient accumulation steps: 8
- Number of GRPO generations: 4
- Max prompt length: 3072
- Max completion length: 384
- Learning rate: `1e-6`
- Save steps: `100000` to avoid expensive intermediate full-state checkpoints

## Rationale

The preceding smoke run showed full-parameter GRPO can run on 4x V100-32GB under `.venv-train-cu124`, while `.venv-train` failed NCCL due to a CUDA runtime/driver mismatch. Intermediate FSDP full-state saving is expensive and caused exit/sync friction after checkpoint write, so this long run avoids periodic checkpointing and saves at the end.

## Runtime Status

- Started at: `2026-07-08T03:46:31Z`
- Launcher PID: `1819747`
- Accelerate PID: `1819753`
- Worker PIDs: `1819811`, `1819812`, `1819813`, `1819814`
- Start check: process detached under `setsid`; GPU 4-7 active; progress initialized at `0/30` optimizer steps.
