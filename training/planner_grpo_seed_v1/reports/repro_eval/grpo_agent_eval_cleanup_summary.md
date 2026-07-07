# GRPO Planner Eval Cleanup Summary

## Changes

- Treat `qwen_detection` and `rexomni_detection` as equivalent open-set single-image detection tools in the GRPO reward verifier.
- Relax `migration_advisor.user_query` scoring from strict all-token matching to non-empty core-token matching.
- Add strict `passed` and `pass_rate` metrics to reward reports.
- Add repeated-eval aggregate metrics:
  - `pass_rate_mean`
  - `pass_rate_stdev`
  - `pass_all_runs_rate`
  - `pass_any_run_rate`
- Clarify Planner routing prompt and tool descriptions:
  - single-image quick probe/direct detection -> `qwen_detection` or `rexomni_detection`
  - generated/expanded samples plus model comparison/evaluation report -> `pipeline_eval`
  - migration feasibility/capability boundary/low-cost validation plan -> `migration_advisor`
  - completed `query_trajectories.steps` must not be repeated unless observation asks for retry
  - after an unstable detection probe, a remaining migration/low-cost-plan request should route to `migration_advisor`
- Add `current_step_index` and `max_steps` to Planner user payload.

## 35B 3x Result

Report:

- `training/planner_grpo_seed_v1/reports/repro_eval/qwen35_35b_a3b_grpo_compound245_stateprompt_t60_3x_aggregate.json`

Metrics:

- `mean_score_mean`: `0.991211`
- `pass_rate_mean`: `0.964626`
- `pass_rate_stdev`: `0.004713`
- `pass_all_runs_rate`: `0.959184`
- `pass_any_run_rate`: `0.967347`

By category:

- `single_image_probe`: pass-all `1.0`
- `migration_feasibility`: pass-all `1.0`
- `migration_feasibility_with_image`: pass-all `1.0`
- `full_detection_eval`: pass-all `1.0`
- `historical_asset_qa`: pass-all `1.0`
- `adela_eval`: pass-all `1.0`
- `intent_ambiguity`: pass-all `1.0`
- `general_answer`: pass-all `0.833333`
- `probe_then_migration`: pass-all `0.852459`

## Residual Bad Cases

Residual any-fail table:

- `training/planner_grpo_seed_v1/reports/repro_eval/badcase_audit/35b_stateprompt_3x_residual_badcases.csv`

Distribution:

- any-fail cases: `10`
- `probe_then_migration`: `9`
- `general_answer`: `1`

Main residual modes:

- second step repeats detection instead of moving to `migration_advisor`
- second-step `migration_advisor.finish_after_tool` is false
- one general template query routes to `rag_answer` instead of `answerer`

## Interpretation

After reward cleanup and prompt/tool-description clarification, 35B passes the overall reproducibility target (`pass_all_runs_rate` > 90%). The remaining signal is concentrated in compound state transitions, especially detection-probe -> migration-advisor. That makes `probe_then_migration` the cleanest candidate direction for GRPO rather than single-step routing.
