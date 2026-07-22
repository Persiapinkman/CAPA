# Generated Research Reports

`CURRENT.md` and `leaderboard.csv` are generated from `experiments/registry.jsonl`, `experiments/project_status.json`, and versioned dataset manifests. Study-specific comparisons and error analyses are preserved beside them.

Run `python pipelines/experiments/registry_cli.py render` after registering an experiment.

## Research Memos

- `QWEN3_32B_V15_POSTHOC_COMPARISON_20260722.md`：Qwen3-32B 在最终 V15 同集上的三轮严格评测；同时报告 111:14 主加权分与不加权 case 正确率，并审计 32B 的结束语义失败。
- `QWEN3_4B_VS_32B_THROUGHPUT_20260721.md`：1×V100 Qwen3.5-4B 与 4×V100 TP Qwen3-32B 的 trial-bootstrap 吞吐、每 GPU 效率、功耗和无效后端审计。
- [Qwen3.5 capability ladder tracker](../experiments/studies/planner_qwen35_4b_capability_ladder_v1/README.md)：当前 `4B-base < 4B-SFT < 4B-GRPO < 35B` 目标的 canonical 审查入口、实验日志、候选场景与最终表验收合同。
- `POST_TRAINING_SFT_GRPO_PLAYBOOK.md`：Qwen3.5-4B 从 SFT、residual mining、GRPO support gate 到 sealed larger-model comparison 的可复现实战手册。
- `PLANNER_MULTISTEP_GRPO_HARD_V2_CONFIRMATION_20260714.md`：245 bad-case 驱动的 V2 多步工具路由新集、35B/7B 3x 模型差分、GRPO support gate 与下一阶段训练门。
- `DEMO_CAPABILITY_LIVE_CHECK_2026-07-14.md`：当前 RAG、Qwen 与 Rex-Omni 的真实服务和 Demo 端到端复验记录。
- `LONG_HORIZON_AGENT_RL_LANDSCAPE.md`：CAPA long-horizon Agent RL 场景调研、环境/奖励设计、数据构造原则与实验前门槛。当前人工讨论从此文档开始。
- `DEMO_AGENT_RUNTIME_ANALYSIS.md`：Demo 运行时、能力服务状态、历史轨迹边界与既有窄 GRPO 结论。
