# 评测数据集（dataset）

本目录存放 Agent / Skill 的**标准评测 case 定义**（输入与期望），不包含真实运行产物。跑批后的 trace、报告与分数应写在 [`../results/`](../results/README.md) 或 `demo/runs/<run_stamp>/`，再用 `scores.template.csv` 汇总 trace 审查结果。

## 目录一览

```
dataset/
├── planner_routing_eval.json # Planner P0 路由评测集（30 case，DPO 前置）
├── migration_advisor.csv      # 迁移顾问原始题库（50 题，序号 + 题目）
├── migration_advisor_eval.json  # 迁移顾问结构化集（30 case，对应 CSV 序号 1–30）
├── migration_advisor_eval.csv   # 迁移顾问扁平清单
```

## 文件说明

| 文件 | 规模 | 用途 |
|------|------|------|
| `planner_routing_eval.json` | 30 case | Planner P0 路由评测：历史资产问答、Adela 评测、通用问答、单图视觉工具、完整视觉探针、迁移边界 |
| `migration_advisor.csv` | 50 题 | 迁移顾问**完整题库**（制表/人工扩题来源） |
| `migration_advisor_eval.json` | 50 case | 由 CSV **50 题**结构化，用于 `POST /test/migration-advisor` 批量回归 |
| `migration_advisor_eval.csv` | 50 行 | 与 `migration_advisor_eval.json` 对应 |

> **注意**：`migration_advisor.csv` 与 `migration_advisor_eval.json` 现已对齐 50 题；`source_index` 与 CSV 序号一致。

## `planner_routing_eval.json` 期望

- 只评估 Planner 的工具路由与关键槽位抽取，不评估最终自然语言答案质量。
- 每条 case 包含 `expected.primary_action`、`required_slots`、`forbidden_actions` 与 `dpo_preference`。
- 可作为 Planner DPO 的前置 ground truth：当前 Agent 跑出的错误 action 可作为 `rejected`，`expected` 规则可构造或人工复核为 `chosen`。
- 覆盖 6 类 P0 任务：历史资产问答、Adela 平台精度/性能、通用问答、可执行视觉单工具、完整视觉探针、迁移/能力边界。



## `migration_advisor_eval.json` 期望

- 默认 `force_migration_advisor: true`，产物为 `migration_advisor_report.json` / `.md`
- 报告应包含章节（任一命中即可）：需求摘要、是否存在直接同款、相似模型、迁移方案、建议
- 无直接支持时须披露（如「不能直接」「需定制」「证据不足」），不得声称「无需改动即可直接支持」

## 批量运行示例

```bash
# 迁移顾问（单条）
curl -s -X POST http://127.0.0.1:18080/test/migration-advisor \
  -H 'Content-Type: application/json' \
  -d '{"query":"香港项目要求识别头盔颜色…","session_id_prefix":"eval","force_migration_advisor":true}'

# 迁移顾问（按 CSV 序号批量，见 demo/README.md）
curl -s -X POST http://127.0.0.1:18080/test/migration-advisor \
  -H 'Content-Type: application/json' \
  -d '{"csv_path":"/abs/path/to/dataset/migration_advisor.csv","indices":[0,7],"session_id_prefix":"mig_eval"}'

# Agent trace：从 agent_trace_eval.csv 取 case_id，在 JSON 中查 turns，多轮需同一 session_id 逐轮 POST /run
```

跑完后：

1. 将 `demo/runs/<run_stamp>/` 路径记入 `scores.template.csv` 的 `run_stamp`
2. 按 rubric 填 T1–T6（P/F 或 N/A）
3. 迁移顾问全量报告可汇总到 `results/advisor/`（见 [`results/README.md`](../results/README.md)）

## 与仓库其他目录的关系

| 目录 | 关系 |
|------|------|
| [`demo/eval/`](../demo/eval/README.md) | 历史会话挖掘 + Planner 路由开发用例，偏迭代调试 |
| **`dataset/`** | 面向**发布门禁**的精简、可版本化 case 与 rubric |
| [`demo/runs/`](../demo/) | Demo / 迁移顾问单次运行的**原始落盘**（含 NDJSON、报告 JSON） |
| [`results/`](../results/README.md) | 脚本流水线、评测复跑、报告索引等**本地产出**（默认 gitignore） |

维护新 case 时：优先改 JSON 再导出或手改 CSV，保持 `case_id` 稳定；勿把 API key、真实分数或未脱敏客户数据提交进本目录。
