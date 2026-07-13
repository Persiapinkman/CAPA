# Active Experiment Pointers

This directory contains lightweight pointers only. The authoritative run history is `experiments/registry.jsonl`; the current human-readable status is generated at `reports/CURRENT.md`.

## Current Qwen2.5 Line

- `best_candidate`: Qwen2.5-7B base model used with native ChatML, the minimal configuration supported by the completed development study.
- `best_lora_checkpoint`: GRPOv4 diagnostic adapter; its incremental routing benefit is inconclusive.
- `sft_initializer`: SFTv3 ChatML initializer and direct comparison baseline.
- `current_grpo_status.json`: machine-readable current state and next decision.

## Legacy Qwen3.5 Pointers

- `legacy_qwen35_missing_artifact.json`: audit record for the historical full-parameter pointer whose checkpoint is no longer present.
- `legacy_qwen35_latest_fullparam_run`

Legacy pointers are retained for auditability but are not the active research candidate.
