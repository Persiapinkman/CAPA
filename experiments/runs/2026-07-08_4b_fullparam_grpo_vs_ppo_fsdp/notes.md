# Qwen3.5-4B Full-Parameter GRPO vs PPO FSDP

- Date: 2026-07-08
- Goal: compare GRPO and PPO training behavior for CAPA Planner routing without LoRA.
- Base model: `/mnt/zkq/models/Qwen3.5-4B`
- Base backup: `/mnt/zkq/models/Qwen3.5-4B.backup-20260708`
- GPUs: default training allocation `CUDA_VISIBLE_DEVICES=4,5,6,7`
- Dataset: `training/planner_grpo_seed_v1/cases/planner_grpo_focused_4b_cases.jsonl`
- Run logs: `experiments/runs/2026-07-08_4b_fullparam_grpo_vs_ppo_fsdp/logs/`

## Resource state

- The previous `qwen3-32b` vLLM service on GPUs 4-7 was stopped before training.
- Existing 4B services on GPUs 0-2 were left running.
- Hardware snapshot is stored at `experiments/resource_snapshot_2026-07-08.json`.

## Technical choice

Use PyTorch FSDP full shard over 4 V100-32GB cards.

Reasoning:

- Full-parameter 4B training with Adam states is too large for one 32GB V100.
- DeepSpeed is not installed in `.venv-train`; compiling it during an experiment adds avoidable risk.
- FSDP is already available through PyTorch/Accelerate and shards parameters, gradients, and optimizer state.
- V100 does not support bf16 efficiently, so the runs use fp16.
- FSDP activation checkpointing and short completions are enabled to reduce activation memory; Trainer/model gradient checkpointing is disabled to avoid duplicate FSDP all-gathers.

## Algorithms

GRPO:

- Uses TRL `GRPOTrainer`.
- Reuses the existing CAPA step-level prompt builder and verifier reward.
- Full-parameter mode is forced with `--use-lora false`.

PPO:

- Uses a project-local clipped PPO policy objective because the installed TRL exposes GRPO but no PPO trainer.
- The smoke/default launcher uses fixed expected Planner JSON rollouts, scored by the same verifier reward, then optimized with old-logprob ratio clipping.
- No value head/critic is used in this first comparison to keep memory close to GRPO and avoid another full 4B module.
- Online PPO rollout directly through an FSDP-wrapped model hit sharded embedding shape issues; the production path should use a separate rollout worker, for example vLLM, then train PPO on the collected rollout batch.

## Commands

GRPO smoke:

```bash
MAX_STEPS=1 SAVE_STEPS=1000 REPORT_TO=none bash scripts/run_qwen35_4b_grpo_fullparam_fsdp.sh
```

PPO smoke:

```bash
MAX_UPDATES=1 SAVE_STEPS=1000 bash scripts/run_qwen35_4b_ppo_fullparam_fsdp.sh
```

Longer comparable runs should keep the same dataset, GPUs, seed, max prompt/completion lengths, and eval protocol, then evaluate both saved full models with the existing single-step and compound Planner evals.

## Smoke Results

See `smoke_results.json` for machine-readable details.

GRPO smoke:

- Environment: `.venv-train-cu124` (`torch 2.6.0+cu124`).
- Completed 1 optimizer step on GPUs 4-7 without OOM.
- Metrics: loss `0.1494`, reward mean `0.4062`, reward std `0.499`, entropy `0.2704`, step time `230.3s`.
- Output: `outputs/planner-grpo-qwen35-4b-focused-full-fsdp/checkpoint-1`.
- Caveat: final distributed save/exit hung after checkpoint write; process was terminated after checkpoint verification.

PPO smoke:

- Environment: `.venv-train-cu124`.
- Online generation through the custom FSDP-wrapped model failed with a sharded embedding shape error, so the smoke uses fixed expected Planner JSON rollouts.
- Completed 1 PPO update and wrote metrics: reward mean `1.0`, loss `-0.5`, approx KL `0.0`, max CUDA memory `14708.07 MB`.
- Caveat: the custom PPO save path did not emit a full model checkpoint; fix before long PPO runs.

Environment issue found:

- `.venv-train` uses `torch 2.8.0+cu128`, while the host driver is `550.163.01` / CUDA `12.4`. Multi-GPU NCCL failed with `CUDA driver version is insufficient for CUDA runtime version`.
- `.venv-train-cu124` was created from the existing cu124 vLLM env and is the correct environment for multi-GPU training on this host.
