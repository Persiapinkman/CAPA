# Experiment Log

人工可读台账，只保留当前需要跟进的最新结果：

- 单步路由：`planner_routing_eval_90cases` 的最新 state-prompt formal 复现评测。
- 多步路由：`planner_grpo_train_cases` 的最新 compound Planner 复现评测。

机器可读 active 元数据见 `manifest.jsonl`。历史结果已归档到 `archive/PRE_REPRO_PROTOCOL_EXPERIMENT_LOG.md` 与 `archive/pre_repro_protocol_manifest.jsonl`。

正式评测协议见 `EVALUATION_POLICY.md`：vLLM serving，`temperature=0`，`top_p=1`，`do_sample=false`，`seed=42`，每次评测 3 repeats 后取 aggregate，并产出逐 case 审查 CSV。

## Active Results

| Date | Run ID | Scope | Model | Adapter | Eval | Metric | Key Notes |
|---|---|---|---|---|---|---:|---|
| 2026-07-07 | `2026-07-07_4b_stateprompt_zip90_3x` | single-step routing | `Qwen3.5-4B` | `none` | `results/planner_routing_eval/qwen35_4b_stateprompt_zip90_3x_aggregate.json` | 81.67/90 mean (0.9074) | 当前新 prompt/tool description 下的单步路由 formal eval；3 repeats，accuracy stdev 0.0064，mean case 4397.82 ms，mean API 2395.89 ms，api_error=3。 |
| 2026-07-07 | `2026-07-07_9b_stateprompt_zip90_3x` | single-step routing | `Qwen3.5-9B` | `none` | `results/planner_routing_eval/qwen35_9b_stateprompt_zip90_3x_aggregate.json` | 85.67/90 mean (0.9519) | 当前新 prompt/tool description 下的单步路由 formal eval；3 repeats，accuracy stdev 0.0065，mean case 12227.23 ms，mean API 12198.78 ms，无 API error。 |
| 2026-07-07 | `2026-07-07_35b_a3b_stateprompt_zip90_3x` | single-step routing | `Qwen3.5-35B-A3B` | `none` | `results/planner_routing_eval/qwen35_35b_a3b_stateprompt_zip90_3x_aggregate.json` | 85/90 (0.9444) | 当前新 prompt/tool description 下的单步路由 formal eval；3 repeats 完全稳定，mean case 1145.47 ms，mean API 1117.49 ms，无 API error。 |
| 2026-07-07 | `2026-07-07_grpo_compound245_35b_stateprompt_3x` | multi-step routing | `Qwen3.5-35B-A3B` | `none` | `training/planner_grpo_seed_v1/reports/repro_eval/qwen35_35b_a3b_grpo_compound245_stateprompt_t60_3x_aggregate.json` | pass-all 235/245 (0.9592) | 最新多步路由 compound eval；清理 qwen/rexomni 等价、`migration_advisor.user_query` reward、Planner 状态转移 prompt；overall pass_rate_mean=0.9646，残余失败集中在 `probe_then_migration`。 |

## Current Takeaways

- 单步路由最新 formal 基线：9B 最高，`85.67/90 = 0.9519`；35B-A3B 为 `85/90 = 0.9444`，4B 为 `81.67/90 = 0.9074`。
- 单步路由每次 repeated eval 必须同步保留逐 case CSV：`<REPORT_PREFIX>_case_audit.csv` 展示全部 90 case 的 query、groundtruth、每轮动作/参数/失败原因；`<REPORT_PREFIX>_failed_cases.csv` 只展示任一 repeat 失败的 case。
- 多步路由评测集经过 reward 与 prompt/tool-description 清洗后，35B-A3B 达到 overall pass^3 目标：`pass_all_runs_rate=0.9592`。
- 后续 GRPO 候选方向不应继续泛化单步工具路由，而应集中在 compound state transition，尤其是 `detection probe -> migration_advisor`。
