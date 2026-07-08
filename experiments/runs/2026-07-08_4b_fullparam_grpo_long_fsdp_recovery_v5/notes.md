# Qwen3.5-4B Full-Parameter GRPO FSDP Recovery v5

## Why v4 stopped

v4 correctly used empty physical GPUs `3,4,5,6`, but failed before training while restoring the FSDP optimizer/scaler state from `checkpoint-2`:

```text
AssertionError: Attempted step but _scale is None
```

This is an optimizer/scaler restore problem in the current Transformers/Accelerate/FSDP2 stack, not a GPU OOM and not foreign GPU usage.

## Recovery Strategy

- Use only empty physical GPUs: `3,4,5,6`.
- Load the step-2 model weights from `checkpoint-2` as the model directory.
- Do not pass `--resume-from-checkpoint`, so optimizer/scaler state is reinitialized cleanly.
- Run `max_steps=28` to finish the remaining budget after the first 2 successful optimizer steps.
- Keep `max_completion_length=160`, `num_generations=2`, `gradient_accumulation_steps=16`.
- Keep `save_steps=2` for progress protection.

This preserves model progress from step 2 while avoiding the broken optimizer-state restore path.
