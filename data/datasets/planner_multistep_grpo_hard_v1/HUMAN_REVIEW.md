# Human Review: planner_multistep_grpo_hard_v1

## 当前审核状态

| 项目 | 状态 |
|---|---|
| schema / fixture / transition 自动检查 | 288/288 通过 |
| 既有数据 exact/normalized-query 泄漏检查 | 通过 |
| gold expected-decision 自洽 smoke | 通过 |
| Agent 结构化内容抽查 | 已完成首个完整 12-case bundle |
| 业务人员 100% gold 复核 | 待签字 |
| calibration 模型 family gate | 失败；2/12 family 准入 |
| confirmation 审核 | V1 不生成 |

在业务人员复核完成前，calibration 可用于场景筛选，但不得把数据标记为
`human_approved`，也不得生成正式 train split。

## 每条 case 审核清单

1. 用户意图是否只有一个合理 gold；clarify case 是否确实存在多条高概率工具路径。
2. 决策是否属于 Planner，而不是 orchestrator 已硬编码的动作。
3. 图片是否仅作为路由前置条件使用；目标词与 fixture 是否一致。
4. 两步 case 的 step1 是否必须 `finish_after_tool=false`。
5. observation 是否足以唯一决定第二步，且未泄露预期 JSON 字段名。
6. 最终工具是否必须 `finish_after_tool=true`；类型必须是 JSON boolean。
7. migration 的 `use_image/use_visual_probe` 是否与用户约束一致。
8. named Qwen/Rex 是否使用严格动作，不允许 alias。
9. forbidden actions 是否覆盖潜在错误副作用工具。
10. 文本是否自然、无原评测 query 改写、无真实用户或资产标识。

## Family 级审核

三个 observation 反事实必须作为一个 block 审核：初始 query 相同，只允许
observation 与第二步 gold 变化。不能单独保留模型做对的分支或删除模型做错的分支。

Calibration 只允许按完整 scenario family 做准入/拒绝。Confirmation 只允许整集
通过/失败，禁止逐题筛选。

## 拒绝条件

- gold 需要 LLM judge 才能决定；
- 真实 runtime 不会把该状态交给 Planner；
- observation 同时支持两个后继动作；
- 正确动作需要真实工具结果而 mock 无法表达；
- query 与既有 train/eval 只是实体替换或近似改写；
- 需要通过宽松 alias、字符串布尔或 parser 修复才能通过；
- 错误动作会产生副作用但未列入 forbidden/reward=0。

## 签字区

- Reviewer：`pending`
- 日期：`pending`
- Calibration decision：`family gate failed; superseded by independent V2 calibration`
- Confirmation freeze approval：`not applicable for V1`
