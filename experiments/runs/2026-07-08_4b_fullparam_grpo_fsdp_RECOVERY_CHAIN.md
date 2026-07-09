# 2026-07-08 4B Full-Parameter GRPO FSDP Recovery Chain

Purpose: preserve the training/recovery timeline without moving or deleting large artifacts.

## Current State

- Active training: none.
- Base model: `/mnt/zkq/models/Qwen3.5-4B`.
- Base backup: `/mnt/zkq/models/Qwen3.5-4B.backup-20260708`.
- Best full-parameter checkpoint: `outputs/planner-grpo-qwen35-4b-focused-full-fsdp-long-20260708-recovery-v5/checkpoint-2`.
- Best LoRA checkpoint: `outputs/planner-grpo-qwen35-4b-focused-lora/checkpoint-50`.

## Timeline

| Run | GPUs | Key settings | Outcome | Artifact |
|---|---:|---|---|---|
| `2026-07-08_4b_fullparam_grpo_long_fsdp` | physical 4,5,6,7 | full-param FSDP, `max_completion_length=384`, high `save_steps` | OOM at optimizer step 17; no checkpoint | none |
| `2026-07-08_4b_fullparam_grpo_long_fsdp_recovery` | physical 4,5,6,7 | lowered `max_completion_length=256`, `save_steps=5` | OOM after 2 optimizer steps; no checkpoint | none |
| `2026-07-08_4b_fullparam_grpo_long_fsdp_recovery_v2` | physical 4,5,6,7 | `num_generations=2`, `max_completion_length=192`, `gradient_accumulation_steps=16` | superseded before useful step count because `max_steps` was wrong | none |
| `2026-07-08_4b_fullparam_grpo_long_fsdp_recovery_v3` | physical 4,5,6,7 | corrected `max_steps=30`, `save_steps=2` | wrote `checkpoint-2`, then OOM in online GRPO generation | `outputs/planner-grpo-qwen35-4b-focused-full-fsdp-long-20260708-recovery-v3/checkpoint-2` |
| `2026-07-08_4b_fullparam_grpo_long_fsdp_recovery_v4` | physical 3,4,5,6 | resumed from v3 checkpoint including trainer state | failed during FSDP optimizer/scaler restore; not an OOM | v3 checkpoint remains usable as model weights |
| `2026-07-08_4b_fullparam_grpo_long_fsdp_recovery_v5` | physical 3,4,5,6 | loaded v3 model weights only, optimizer reinitialized, `max_completion_length=160` | wrote own `checkpoint-2`, then OOM in Qwen3.5 linear-attention generation | `outputs/planner-grpo-qwen35-4b-focused-full-fsdp-long-20260708-recovery-v5/checkpoint-2` |

## Interpretation

The repeated OOM is not caused by optimizer/model state size under FSDP. The failing stage is online GRPO generation, where each rank still carries local activation/KV/linear-attention peak memory. FSDP does not shard that generation-time peak.

Traceback GPU ids are local ids after `CUDA_VISIBLE_DEVICES`; v5 used physical GPUs `3,4,5,6`, not physical `0,1,2`.

## Recommended Next Direction

For correctness and progress, the best next change is architectural: decouple rollout/generation from the FSDP training process, or use a rollout cache so the trainer does not run peak-memory generation inside each FSDP rank.

If a short salvage run is needed before refactor, use only verified empty physical GPUs, load v5 checkpoint model weights only, reinitialize optimizer state, use `max_completion_length <= 128`, keep `num_generations=2`, and save every 1-2 optimizer steps.
