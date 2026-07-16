# Dataset Card: planner_multistep_grpo_value_v5_train_v1

## 目的

这是 `planner_multistep_grpo_value_v5` 的独立训练类比集，用于学习视觉检测后的
`retry` / `migration_advisor` 状态边界。它复现 V5 confirmation 的题族和目标动作分布，
但不复制任何 calibration 或 confirmation case。

## 组成

| 项目 | V5 confirmation | 本训练池 |
|---|---:|---:|
| entity groups | 30 | 60 |
| cases | 240 | 480 |
| Planner decisions | 480 | 960 |
| scenario families | 8 | 8 |
| migrate | 60% | 60% |
| retry | 40% | 40% |
| Qwen / Rex | 50% / 50% | 50% / 50% |

每个训练实体覆盖全部 8 个题族；同一实体的 Qwen/Rex query 在各自 detector block 内保持
一致。每个错误别名同时包含 `migrate` 和 `retry`，动作只能由 `retryable/retry_count`
决定。`overall_badge` 保持 V5 的 60/40 边际分布，但与动作解耦，避免学习 badge 捷径。

## 实体与内容隔离

训练数据使用新的：

- case ID、entity ID、项目实体和图片目标实体；
- query 表述与 template ID；
- gateway error alias；
- fixture family、fixture 路径和实际图片文件。

构建器会在生成后扫描 V5 calibration、V5 confirmation 以及仓库内其它 case 文件，并对
上述字段和 fixture SHA256 做零交叉检查。8 个 `scenario_id` 有意共享，因为它们是需要匹配
的任务 taxonomy，不是业务实体。

## 数据来源边界

构建阶段只读取 V5 confirmation 的聚合题族和类别计数。V5 case 文件仅在训练行已经构造
之后用于 overlap audit；其逐题 query、gold、实体与错误别名不会成为训练行来源。

## 文件与复现

- 训练 case：
  `training/planner_grpo_seed_v1/cases/planner_multistep_grpo_value_v5_train_v1_train_cases.jsonl`
- 机器可读统计、hash 与隔离结果：`manifest.json`
- 构建器：
  `training/planner_grpo_seed_v1/scripts/build_planner_multistep_grpo_value_v5_train_v1.py`

复现命令：

```bash
python training/planner_grpo_seed_v1/scripts/build_planner_multistep_grpo_value_v5_train_v1.py
```

## 使用边界

- `grpo_target_step=2`；第一步检测是形成状态上下文的必要前缀。
- 图片是 Planner 路由 fixture，不评估真实视觉识别质量。
- 该文件只完成数据构建，不授权启动 optimizer step。
- V5 calibration/confirmation 均不得混入训练、support audit 或 checkpoint selection。
- 正式训练前仍需对 step 2 做 stochastic-support audit，并人工抽查标签。
