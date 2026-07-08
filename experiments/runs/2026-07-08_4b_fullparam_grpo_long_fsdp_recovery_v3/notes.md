# Qwen3.5-4B Full-Parameter GRPO FSDP Long Run Recovery v3

## Purpose

Run the corrected long GRPO recovery after v2 was stopped before optimizer step 1 because it would only run 7 optimizer steps.

## Configuration Fix

- Keep the OOM fix from v2: reduce per-rank generation peak with `num_generations=2` and `max_completion_length=192`.
- Preserve roughly the same completions per optimizer update by increasing `gradient_accumulation_steps=16`.
- Restore long-run scale by explicitly setting `max_steps=30`.
- Protect progress with `save_steps=2` and `SHARDED_STATE_DICT`.

## Progress Guarantee

The first failed long run completed 16 steps without a checkpoint. Recovery v1 completed 2 steps without a checkpoint. This run should emit a sharded checkpoint at step 2, so subsequent failures lose at most 2 optimizer steps.
