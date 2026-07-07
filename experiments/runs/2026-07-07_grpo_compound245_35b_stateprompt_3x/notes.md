# GRPO Compound Planner Eval Cleanup

- Date: 2026-07-07
- Runner: `scripts/run_grpo_repro_eval_3x.sh`
- Model: `Qwen3.5-35B-A3B`
- API base: `http://10.111.32.253:8000/v1`
- Cases: `training/planner_grpo_seed_v1/cases/planner_grpo_train_cases.jsonl`
- Case count: 245
- Generation: `temperature=0`, `top_p=1`, `seed=42`, `do_sample=false`
- Timeout: 60 seconds per Planner step

Purpose:

- Validate that the GRPO compound Planner eval set is not dominated by reward noise or prompt/tool-description ambiguity.
- Check whether 35B can pass the cleaned eval with stable 3-repeat reproducibility.

Cleanup applied before final 3x:

- Treated `qwen_detection` and `rexomni_detection` as equivalent open-set single-image detection tools in the reward verifier.
- Relaxed `migration_advisor.user_query` scoring so paraphrases are accepted when they retain the core business object.
- Clarified Planner prompt/tool descriptions:
  - single-image quick probe/direct detection -> `qwen_detection` or `rexomni_detection`
  - generated/expanded samples plus model comparison/evaluation report -> `pipeline_eval`
  - migration feasibility/capability boundary/low-cost validation plan -> `migration_advisor`
  - completed `query_trajectories.steps` should not be repeated unless observation asks for retry
- Added `current_step_index` and `max_steps` to Planner user payload.

Aggregate result:

- Mean score mean: 0.991211
- Pass rate mean: 0.964626
- Pass rate stdev: 0.004713
- Pass-all-runs rate: 0.959184
- Pass-any-run rate: 0.967347
- Empty decisions: 0
- Errors: none

Important category result:

- `probe_then_migration` pass-all-runs rate: 0.852459
- All other major categories except `general_answer` reached pass-all-runs rate 1.0.

Interpretation:

- Overall 35B reproducibility target is met after cleanup: pass-all-runs rate is above 90%.
- Remaining failures are concentrated in compound state transitions, especially detection probe -> migration advisor.
- This supports using compound multi-step state transition cases as the likely GRPO training direction, rather than generic single-step tool routing.

Artifacts:

- Aggregate: `training/planner_grpo_seed_v1/reports/repro_eval/qwen35_35b_a3b_grpo_compound245_stateprompt_t60_3x_aggregate.json`
- Summary: `training/planner_grpo_seed_v1/reports/repro_eval/grpo_agent_eval_cleanup_summary.md`
- Residual bad cases: `training/planner_grpo_seed_v1/reports/repro_eval/badcase_audit/35b_stateprompt_3x_residual_badcases.csv`
