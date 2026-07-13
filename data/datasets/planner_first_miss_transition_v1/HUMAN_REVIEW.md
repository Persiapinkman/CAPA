# Human Review: planner_first_miss_transition_v1

## 审阅顺序

1. 阅读 `DATASET_CARD.md`，确认本数据只改变训练采样概率。
2. 检查 `metadata.json`：`rows=1584`、`unique_source_rows=1080`、`entities=72`。
3. 确认 double-miss step 1/2 权重分别为 2/4，single-miss step 2 权重为 4，其余均为 1。
4. 抽查 `training_row_id` 和 `source_case_id`，确保副本可追溯。
5. 确认 reward_spec 与 v2 完全一致，错误动作 task cap 仍为 0.20。
6. 确认开发评估使用 v2 的原始 `dev.jsonl`，测试仍封存。

## 拒绝条件

- 删除任一原始 source step 或任一 guardrail。
- 把副本计为独立 case/entity。
- 根据开发实体单独选择训练行。
- 修改奖励或同时修改训练超参数，导致无法归因于采样课程。
