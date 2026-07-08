# Qwen3.5-4B Full-Parameter GRPO FSDP Long Run Recovery v2

## Purpose

Recover the full-parameter GRPO long run after two online-generation OOM failures while preserving useful GRPO signal and guaranteeing earlier checkpoint progress.

## Root cause

Both failures occurred inside TRL GRPO online generation, not during optimizer state sharding:

- FSDP shards parameters, gradients, and optimizer state, but `unwrapped_model.generate()` still creates rank-local activation, attention, and KV-cache peaks.
- The OOM stack ends in `torch.nn.functional.scaled_dot_product_attention` during generation prefill.
- The focused Planner prompts are short enough after tokenization (`max=390` tokens), so `max_prompt_length=3072` is not the real peak driver.
- The driver is the rank-local generation batch: `per_device_train_batch_size=1` multiplied by `num_generations=4`, plus long sampled completions.

## Recovery v2 choice

- `num_generations`: 4 -> 2.
- `gradient_accumulation_steps`: 8 -> 16, keeping approximately the same completions per optimizer update.
- `max_completion_length`: 256 -> 192.
- `save_steps`: 5 -> 2, so progress is protected before the previously observed failure point.
- `fsdp_state_dict_type`: `SHARDED_STATE_DICT`, avoiding full-state gather on checkpoint.
- Keep full-parameter training and no LoRA.

This is preferable to dropping to PPO or LoRA because it addresses the specific online-generation memory peak while preserving the GRPO objective.
