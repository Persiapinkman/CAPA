# Qwen3.5-4B LoRA GRPO Focused Training

- Date started: 2026-07-07
- Goal: train a lightweight LoRA adapter for compound Planner GRPO on focused 4B cases.
- Base model: `/mnt/zkq/models/Qwen3.5-4B`
- GPU: physical `3` (`CUDA_VISIBLE_DEVICES=3`, single V100-32GB)
- Dataset: `training/planner_grpo_seed_v1/cases/planner_grpo_focused_4b_cases.jsonl`
- Script: `scripts/run_qwen35_4b_grpo_focused.sh`
- Output: `outputs/planner-grpo-qwen35-4b-focused-lora`

## Configuration

- LoRA: `r=16`, `alpha=32`, `dropout=0.05`, target `q_proj,k_proj,v_proj,o_proj`
- GRPO: `num_generations=4`, `max_completion_length=512`, `gradient_accumulation_steps=8`, `learning_rate=5e-6`
- Precision: fp16
- Environment: `.venv-train` (torch 2.8.0+cu128)

## Attempts

1. Initial run (`logs/grpo_lora_initial.log`): completed optimizer steps 1-66, wrote `checkpoint-25` and `checkpoint-50`, OOM at step 67 during TRL `generate()` SDPA prefill.
2. Resume run (`logs/grpo_lora_resume50_mc384.log`, 2026-07-08): resumed from `checkpoint-50` with `max_completion_length=384`, completed steps 51-66 again, OOM again at step 67.

## Outcome

- Status: failed OOM, but this is currently the furthest 4B GRPO training progress with a usable artifact.
- Best checkpoint: `checkpoint-50` (~56M adapter + optimizer state)
- No formal single-step 90 or compound 245 eval has been run on this adapter yet.

## Relation to full-param FSDP runs

LoRA uses much less memory for trainable weights and checkpoints, so it progressed to 66/122 optimizer steps before hitting the same online-generation memory peak. Full-param FSDP runs later failed much earlier (around step 2-17) on the same TRL generate path.
