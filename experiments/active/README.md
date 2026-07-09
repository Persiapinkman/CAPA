# Active Experiment Pointers

This directory is the lightweight entry point for the current CAPA training state. It contains metadata and symlinks only; no large checkpoint files are stored here.

Current truth:

- No full-parameter GRPO/FSDP training process is active.
- Best full-parameter GRPO checkpoint: `best_fullparam_checkpoint` -> `outputs/planner-grpo-qwen35-4b-focused-full-fsdp-long-20260708-recovery-v5/checkpoint-2`.
- Best LoRA GRPO checkpoint: `best_lora_checkpoint` -> `outputs/planner-grpo-qwen35-4b-focused-lora/checkpoint-50`.
- Latest full-parameter run metadata: `latest_fullparam_run`.
- Machine-readable summary: `current_grpo_status.json`.

Use `experiments/runs/2026-07-08_4b_fullparam_grpo_fsdp_RECOVERY_CHAIN.md` for the v0-v5 failure/recovery timeline.
