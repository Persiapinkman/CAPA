# 9B Result Audit

Date: 2026-07-09

## Verdict

The local 9B compound eval output is invalid and must not be used as a model-quality result.

## Findings

- The run name used `compound245`, but the command read `training/planner_grpo_seed_v1/cases/planner_grpo_train_cases.jsonl`, which currently has 313 rows.
- Run1 produced 313 predictions, with 302 empty-decision rows and 304 `planner rollout step timed out` errors.
- The run1 reward was `0/313` passed with `pass_rate=0.0`; this is a serving failure, not a meaningful 9B planner result.
- A minimal JSON prompt against the local FP16 V100 service returned repeated garbage and `finish_reason=length`.
- FP32 attempts on local V100 were not viable: multi-GPU vLLM 0.10.2 failed NCCL init, cu124/vLLM 0.8.5 hit Qwen3.5 config/model-init incompatibilities, and single-GPU FP32 CPU-offload still OOMed during `lm_head` allocation.
- The historical good 9B results used `http://10.111.32.253:8000/v1`; that endpoint was not reachable from this host on 2026-07-09.

## Corrective Actions

- Stopped the bad 9B eval runner and rollout.
- Created explicit 245-case eval set: `training/planner_grpo_seed_v1/cases/planner_grpo_compound245_eval_cases.jsonl`.
- Wrote invalid-run audit artifacts:
  - `results/compound_planner_eval/qwen35_9b_compound245_stateprompt_t60_3x_20260708_invalid_diagnostic.json`
  - `results/compound_planner_eval/qwen35_9b_compound245_stateprompt_t60_3x_20260708_invalid_run1_case_audit.csv`
  - `results/compound_planner_eval/qwen35_9b_compound245_stateprompt_t60_3x_20260708_invalid_run1_failed_cases.csv`

## Required Resolution

Rerun 9B formal eval only on a BF16-capable local service or a previously validated remote 9B endpoint. Do not use the current V100 FP16 9B vLLM service for formal results.
