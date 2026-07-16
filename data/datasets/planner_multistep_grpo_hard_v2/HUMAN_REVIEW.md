# Human Review: planner_multistep_grpo_hard_v2

## 当前状态

- 自动 schema、transition、fixture、strict reward 与泄漏检查：384/384 通过。
- Agent 抽查：首个 12-case entity bundle 已检查。
- 业务人员 100% gold 复核：待签字。
- Calibration family gate：8/12 个完整场景族准入，未逐 case 筛选。
- Confirmation：600/600 已生成并通过自动 schema/transition/leakage 检查；
  业务人员 100% gold 复核仍待签字。
- Confirmation model gate：整版通过；35B pass-all 97.17%，raw 7B 22.67%；
  该模型结果不能替代业务人员 gold 审核。

## 审核重点

1. 三种不确定 observation 是否都足以触发迁移，而不是重试检测。
2. 高置信 observation 是否只允许 end，防止固定动作串。
3. 点名 Qwen/Rex 是否严格执行对应检测工具。
4. 迁移顾问是否明确需要当前图片和内部视觉探针。
5. 最终 migration report 是否确实可以结束请求。
6. 四个单步 family 是否在业务目标上有实际差别，而非纯同义改写。
7. query、实体和模板是否与 V1/compound245/既有训练数据独立。

模型筛选仍以完整 family 为单位；业务审核不得根据某个模型是否答对来决定单题
去留。
