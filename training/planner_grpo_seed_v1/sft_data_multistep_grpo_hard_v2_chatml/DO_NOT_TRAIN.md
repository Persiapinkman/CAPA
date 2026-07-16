# Evaluation/support rows — not a training split

This directory stores canonical step rows used to validate prompts, rewards, and GRPO
sampling support for `planner_multistep_grpo_hard_v2`.

- `calibration.jsonl`: visible family-calibration steps; not an approved train split.
- `support_audit.jsonl`: 128-row sampling audit subset; not an approved train split.
- `confirmation.jsonl`: held-out confirmation gold; **must never be used for SFT or GRPO**.

No train/validation split has been generated for this dataset version. A future training
dataset must use new entities, templates, fixture families, and case IDs, pass an explicit
leakage audit against both calibration and confirmation, and receive separate approval.
