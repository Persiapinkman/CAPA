# Human Review: planner_multistep_grpo_value_v5_train_v1

在训练或 support audit 前，按以下顺序审核：

1. 从 8 个 `scenario_id` 各抽至少 5 个 entity block，同时查看 migrate 与 retry。
2. 确认第一步 detector 与题面点名的 Qwen/Rex 一致，且 `finish_after_tool=false`。
3. 对 retry 样本确认 `retryable=true,retry_count=0`，第二步仍为同一 detector。
4. 对 nonretryable 样本确认 `retryable=false`，第二步为 `migration_advisor`。
5. 对 budget-exhausted 样本确认 `retryable=true,retry_count>0`，第二步为
   `migration_advisor`。
6. 确认迁移参数包含完整 `project_entity`，检测 label 包含完整 `target_entity`。
7. 确认 observation 没有“重试/迁移”答案提示，badge 不能替代结构化门槛。
8. 查看 `manifest.json`：分布 TV distance 必须为 0，所有 protected overlap 必须为 0，
   canonical gold 必须为 `480/480`。

出现以下任一情况应拒绝训练：V5 实体/query/error alias/fixture 交叉、动作比例偏离、
canonical gold 失败、图片路径缺失，或人工规则判断与 gold 不一致。
