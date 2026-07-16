# Generated Research Reports

`CURRENT.md` and `leaderboard.csv` are generated from `experiments/registry.jsonl`, `experiments/project_status.json`, and versioned dataset manifests. Study-specific comparisons and error analyses are preserved beside them.

Run `python pipelines/experiments/registry_cli.py render` after registering an experiment.

## Research Memos

- `PLANNER_MULTISTEP_GRPO_HARD_V2_CONFIRMATION_20260714.md`：245 bad-case 驱动的 V2 多步工具路由新集、35B/7B 3x 模型差分、GRPO support gate 与下一阶段训练门。
- `DEMO_CAPABILITY_LIVE_CHECK_2026-07-14.md`：当前 RAG、Qwen 与 Rex-Omni 的真实服务和 Demo 端到端复验记录。
- `LONG_HORIZON_AGENT_RL_LANDSCAPE.md`：CAPA long-horizon Agent RL 场景调研、环境/奖励设计、数据构造原则与实验前门槛。当前人工讨论从此文档开始。
- `DEMO_AGENT_RUNTIME_ANALYSIS.md`：Demo 运行时、能力服务状态、历史轨迹边界与既有窄 GRPO 结论。
