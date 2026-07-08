# Qwen3.5-4B Full-Parameter GRPO Long Run Recovery

- Date: 2026-07-08
- Purpose: restart the long FSDP GRPO run after step-17 generation OOM.
- Base model: `/mnt/zkq/models/Qwen3.5-4B`
- Base backup: `/mnt/zkq/models/Qwen3.5-4B.backup-20260708`
- Environment: `.venv-train-cu124` (`torch 2.6.0+cu124`)
- GPUs: `CUDA_VISIBLE_DEVICES=4,5,6,7`
- Output: `outputs/planner-grpo-qwen35-4b-focused-full-fsdp-long-20260708-recovery`

## Failure Analysis

The failed long run reached step 16/30 and OOMed when preparing step 17. The traceback shows the OOM inside `unwrapped_model.generate()` during full-attention SDPA prefill:

- failing rank: local rank 2
- free memory at failure: about `2.00 GiB`
- requested allocation: `2.11 GiB`
- max completion length in failed run: `384`
- number of generations: `4`

This is expected behavior for FSDP under online GRPO: FSDP shards parameters, gradients, and optimizer states, but generation-time KV cache and attention activations are not sharded. Variable prompt/completion lengths can therefore create a late-run peak even when earlier steps fit.

No checkpoint was available from the failed run because `save_steps=100000` was used to avoid the expensive full-state checkpoint path. The 16 optimizer steps cannot be exactly recovered.

## Recovery Changes

- FSDP checkpoint type changed from `FULL_STATE_DICT` to `SHARDED_STATE_DICT`.
- Save every 5 optimizer steps, `save_total_limit=3`, so future failure loses at most 5 steps.
- Reduce `max_completion_length` from `384` to `256` to lower generation KV/attention peak.
- Keep `num_generations=4` to preserve GRPO group signal.
- Keep `max_prompt_length=3072` for now to preserve Planner state context.
- Keep full-parameter training, no LoRA.

## Launch Settings

- `learning_rate=1e-6`
- `gradient_accumulation_steps=8`
- `per_device_train_batch_size=1`
- `num_train_epochs=1`
- expected optimizer steps: `30`
- checkpoint cadence: every `5` steps
