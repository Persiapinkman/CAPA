# Experiment Log

人工可读台账，只保留当前需要跟进的最新结果：

- 单步路由：`planner_routing_eval_90cases` 的最新 state-prompt formal 复现评测。
- 多步路由：`planner_grpo_train_cases` 的最新 compound Planner formal 复现评测。

机器可读 active 元数据见 `manifest.jsonl`。历史结果已归档到 `archive/PRE_REPRO_PROTOCOL_EXPERIMENT_LOG.md` 与 `archive/pre_repro_protocol_manifest.jsonl`。

正式评测协议见 `EVALUATION_POLICY.md`：vLLM serving，`temperature=0`，`top_p=1`，`do_sample=false`，`seed=42`，每次评测 3 repeats 后取 aggregate，并产出逐 case 审查 CSV。当前最终记录口径是：单步路由使用 `state-prompt formal eval`；多步路由使用清洗过 reward 与 tool-description / state transition prompt 的 `compound Planner formal eval`，`timeout=60s`，`max_steps=3`，核心指标同时记录 `pass_all_runs_rate` 与 `pass_rate_mean`。

## Active Results

| Date | Run ID | Scope | Model | Adapter | Eval | Metric | Key Notes |
|---|---|---|---|---|---|---:|---|
| 2026-07-07 | `2026-07-07_4b_stateprompt_zip90_3x` | single-step routing | `Qwen3.5-4B` | `none` | `results/planner_routing_eval/qwen35_4b_stateprompt_zip90_3x_aggregate.json` | 81.67/90 mean (0.9074) | 当前新 prompt/tool description 下的单步路由 formal eval；3 repeats，accuracy stdev 0.0064，mean case 4397.82 ms，mean API 2395.89 ms，api_error=3。 |
| 2026-07-07 | `2026-07-07_9b_stateprompt_zip90_3x` | single-step routing | `Qwen3.5-9B` | `none` | `results/planner_routing_eval/qwen35_9b_stateprompt_zip90_3x_aggregate.json` | 85.67/90 mean (0.9519) | 当前新 prompt/tool description 下的单步路由 formal eval；3 repeats，accuracy stdev 0.0065，mean case 12227.23 ms，mean API 12198.78 ms，无 API error。 |
| 2026-07-07 | `2026-07-07_35b_a3b_stateprompt_zip90_3x` | single-step routing | `Qwen3.5-35B-A3B` | `none` | `results/planner_routing_eval/qwen35_35b_a3b_stateprompt_zip90_3x_aggregate.json` | 85/90 (0.9444) | 当前新 prompt/tool description 下的单步路由 formal eval；3 repeats 完全稳定，mean case 1145.47 ms，mean API 1117.49 ms，无 API error。 |
| 2026-07-07 | `2026-07-07_grpo_compound245_4b_stateprompt_3x` | multi-step routing | `Qwen3.5-4B` | `none` | `training/planner_grpo_seed_v1/reports/repro_eval/qwen35_4b_grpo_compound245_stateprompt_t60_3x_aggregate.json` | pass-all 0.8041; pass-rate mean 0.8517 | 最新多步路由 compound eval（3 repeats）；`migration_feasibility` 文本迁移仍弱（pass^3=0.4211，pass_rate_mean=0.4912），但 `probe_then_migration` 稳定性仍高于 9B（pass^3=0.8033）；累计 2 次 `planner rollout step timed out`。 |
| 2026-07-07 | `2026-07-07_grpo_compound245_9b_stateprompt_3x` | multi-step routing | `Qwen3.5-9B` | `none` | `training/planner_grpo_seed_v1/reports/repro_eval/qwen35_9b_grpo_compound245_stateprompt_t60_3x_aggregate.json` | pass-all 0.6933; pass-rate mean 0.8914 | 最新多步路由 compound eval（3 repeats）；`migration_feasibility` / `migration_feasibility_with_image` 已接近解决（pass^3=0.9474 / 0.9608），但 `probe_then_migration` 明显拖后腿（pass^3=0.6230，pass_rate_mean=0.6885），重复稳定性弱于 4B/35B。 |
| 2026-07-07 | `2026-07-07_grpo_compound245_35b_stateprompt_3x` | multi-step routing | `Qwen3.5-35B-A3B` | `none` | `training/planner_grpo_seed_v1/reports/repro_eval/qwen35_35b_a3b_grpo_compound245_stateprompt_t60_3x_aggregate.json` | pass-all 235/245 (0.9592) | 最新多步路由 compound eval；清理 qwen/rexomni 等价、`migration_advisor.user_query` reward、Planner 状态转移 prompt；overall pass_rate_mean=0.9646，残余失败集中在 `probe_then_migration`。 |
| 2026-07-08 | `2026-07-08_9b_32b_compound_planner_eval` | multi-step routing | `Qwen3.5-9B` / `Qwen3-32B` | `none` | `experiments/runs/2026-07-08_9b_32b_compound_planner_eval/run_status.json` | 9B invalid; 32B pass-all 0.9105 | 32B 本地 gateway 3x 评测已完成，实际跑的是当前 `planner_grpo_train_cases.jsonl` 的 313 case（虽然文件名前缀仍是 compound245），audit 与 failed cases 已落盘。9B 本地 V100 服务结果判定 invalid：run1 实际跑 313 case，302 个空决策、304 次 timeout、0/313 passed；最小 JSON smoke 在 FP16 服务上输出重复乱码。已停止错误长跑，补显式 `planner_grpo_compound245_eval_cases.jsonl`，写出 invalid audit/badcases/diagnostic；正式 9B 复跑需要 BF16-capable/已验证 endpoint。 |
| 2026-07-07 | `2026-07-07_4b_lora_grpo_focused` | train long run | `Qwen3.5-4B` | `outputs/planner-grpo-qwen35-4b-focused-lora/checkpoint-50` | `experiments/runs/2026-07-07_4b_lora_grpo_focused/run_status.json` | 66/122; checkpoint-50 | 单卡物理 GPU 3、LoRA r=16、`num_generations=4`、`max_completion_length=512`；完成 66 optimizer step 并写出 `checkpoint-25/50`（各约 56M）；step 67 在 TRL GRPO `generate()` SDPA prefill OOM。2026-07-08 从 `checkpoint-50` 续跑并将 `max_completion_length` 降到 384，仍在 step 67 OOM。尚无 formal eval。 |
| 2026-07-08 | `2026-07-08_4b_fullparam_grpo_vs_ppo_fsdp` | train smoke | `Qwen3.5-4B` | `none` | `experiments/runs/2026-07-08_4b_fullparam_grpo_vs_ppo_fsdp/smoke_results.json` | GRPO 1 step; PPO 1 update | 停止 32B 服务释放 GPU 4-7；备份 base 到 `/mnt/zkq/models/Qwen3.5-4B.backup-20260708`；建立 `.venv-train-cu124` 解决 cu128/NCCL 驱动不匹配；GRPO 全参 FSDP 1 step 成功写出 checkpoint；PPO fixed-rollout clipped objective 1 update 成功写 metrics，在线/FSDP rollout 和 PPO checkpoint 保存仍需后续修复。 |
| 2026-07-08 | `2026-07-08_4b_fullparam_grpo_long_fsdp_recovery_v3` | train long run | `Qwen3.5-4B` | `none` | `experiments/runs/2026-07-08_4b_fullparam_grpo_long_fsdp_recovery_v3/run_status.json` | failed after checkpoint-2 | 初始长跑在 step 17 的 TRL GRPO `generate()` SDPA prefill OOM，recovery v1 在 step 3 同因 OOM；v3 改为 `num_generations=2`、`max_completion_length=192`、`gradient_accumulation_steps=16`、`max_steps=30`、`save_steps=2`。完成 step 2 并写出 `checkpoint-2` 约 37G，随后再次在 GRPO 在线生成峰值 OOM。 |
| 2026-07-08 | `2026-07-08_4b_fullparam_grpo_long_fsdp_recovery_v5` | train long run | `Qwen3.5-4B` | `none` | `experiments/runs/2026-07-08_4b_fullparam_grpo_long_fsdp_recovery_v5/run_status.json` | failed after v5 checkpoint-2 | v3 后续 OOM 仍来自本任务 GRPO 生成峰值，不是物理 0/1/2 被占用；CUDA 日志 GPU id 是 `CUDA_VISIBLE_DEVICES` 后的 local id。v4 使用物理空卡 `3,4,5,6` 但 FSDP optimizer/scaler restore 失败；v5 改为加载 v3 `checkpoint-2` 模型权重、重置 optimizer，继续在物理空卡 `3,4,5,6` 上训练。完成 v5 additional step 2 并写出 `checkpoint-2` 约 37G，随后 rank2 在 Qwen3.5 linear attention generation 中申请 554MiB OOM；当前无后台训练进程。 |

## Current Takeaways

- 单步路由最新 formal 基线：9B 最高，`85.67/90 = 0.9519`；35B-A3B 为 `85/90 = 0.9444`，4B 为 `81.67/90 = 0.9074`。
- 单步路由每次 repeated eval 必须同步保留逐 case CSV：`<REPORT_PREFIX>_case_audit.csv` 展示全部 90 case 的 query、groundtruth、每轮动作/参数/失败原因；`<REPORT_PREFIX>_failed_cases.csv` 只展示任一 repeat 失败的 case。
- 多步路由 3x 对比已经补齐：35B-A3B 最强，`pass_all_runs_rate=0.9592`、`pass_rate_mean=0.9646`；4B 为 `0.8041 / 0.8517`；9B 为 `0.6933 / 0.8914`。9B 的单次平均通过率高于 4B，但跨 repeats 稳定性反而更差。
- 4B 与 9B 的失败模式不同：4B 主要卡在纯文本 `migration_feasibility`（`pass^3=0.4211`），9B 则主要卡在 `probe_then_migration` 状态转移（`pass^3=0.6230`），说明 9B 不是不会做迁移建议，而是更容易在多步衔接上失稳。
- 35B-A3B 的残余失败仍集中在 `probe_then_migration`，但该类也已经达到 `pass^3=0.8525`；后续 GRPO 候选方向不应继续泛化单步工具路由，而应集中在 compound state transition，尤其是 `detection probe -> migration_advisor`。
- 4B LoRA GRPO 是目前进度最远的训练线：`66/122` optimizer step，可用 adapter 为 `outputs/planner-grpo-qwen35-4b-focused-lora/checkpoint-50`；失败点与全参 FSDP 相同，都是 TRL 在线 `generate()` 峰值，但 LoRA checkpoint 仅约 56M，续跑/评测成本更低。
- FSDP OOM 根因不是参数/优化器状态，而是 TRL GRPO 在线生成的 rank-local activation/KV/SDPA prefill 峰值；FSDP 不会 shard 这部分内存。当前保守解法用更小 generation group 降峰值，用更高 gradient accumulation 保持每 step completion 数量，并用 `save_steps=2` 防止再次丢失长时间进度。
- GPU 使用必须看物理 `nvidia-smi`，不要把 traceback 里的 local `GPU 0/1/2` 误读成物理卡；v5 明确使用物理空卡 `3,4,5,6`，物理 `0/1/2` 上的其他进程未被使用。
- 当前没有正在运行的全参 GRPO/FSDP 训练进程；最新可用全参 checkpoint 是 `outputs/planner-grpo-qwen35-4b-focused-full-fsdp-long-20260708-recovery-v5/checkpoint-2`。
