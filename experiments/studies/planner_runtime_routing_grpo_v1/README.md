# Planner Runtime Routing GRPO v1

## 目的

验证 GRPO 能否改善 Demo 中真正由 Planner 决定的动作，而不是改善随后会被编排器硬编码覆盖的离线标签。

## 为什么重开场景

已确认的 `planner_complex_retrieval_grpo_v2` 在受控五步 RAG 状态机上有增长，但运行时在 RAG miss 后会强制执行改写与重试，第三次 miss 也会直接输出迁移顾问选项。因此该结果是可学习性证据，不是当前 Demo 的直接策略收益。

本研究改测以下运行时决策：首步工具路由、必要参数、`finish_after_tool`，以及单图探针设置为非终止后，下一步是否调用迁移顾问。

## 固定比较

- Baseline：合并后的 Qwen2.5-7B SFTv3。
- Candidate：已经完成训练的 route-focused GRPOv4 60-step adapter。
- 数据：新建 `planner_runtime_routing_v1`，train/dev/test 实体和句法模板隔离。
- 本轮不使用 train 训练；它只为候选失败后的新预注册训练臂保留。

## 开发门

主场景 `qwen_probe_then_migration` 的两步严格动作提升至少 `+0.05`，verifier 至少 `+0.03`。对照 `qwen_probe_only_contrast` 和任一 guardrail 的动作率不得回退超过 `0.05`，错误副作用工具调用不得增加。

只有全部通过，才保持原模型与生成设置一次性打开 test；否则 test 继续封存。

## 固定候选结果

旧 route-GRPOv4 未通过：主场景严格两步 case 成功率与 SFT 均为 `0.625`，动作增量为 `0`；verifier 增量 `+0.0292` 也低于门槛。test 未打开。

随后预注册 `runtime_probe_curriculum_arm_v1.json`：使用严格模型动作奖励和不重复的 probe/contrast 训练表述。只有采样支持门通过才运行 seed 42。

v1 合并支持门因平均动作种类 `1.375 < 1.40` 失败，未训练。失败由已饱和的 probe-only guardrail 拉低；两步主场景本身为 `1.5625`、奖励方差率 `1.0`、正确动作支持率 `1.0`。因此另行预注册 v2，把动作多样性门限定于实际需要学习的主场景，同时保留全局方差门和原 guardrail；其它数据、reward 与训练参数不变。

## Seed-42 屏幕结果

新 GRPO 将主场景完整两步 case 成功率从 `0.625` 提升到 `1.000`（`+0.375`），verifier 提升 `+0.2298`；step 2、probe-only 对照和所有其它 category 均无动作回退。全开发集实体聚类 case-macro verifier 增量为 `+0.1225`，95% CI `[+0.1026,+0.1417]`。

按规则已预注册 seeds 43/44 的多种子复现门；test 仍未打开。

## 多种子确认规则

seeds 42/43/44 固定使用同一数据、reward、80 步、学习率和生成参数。seeds 43/44 采用每任务 4 GPU、梯度累积 4，保持与 seed 42 相同的全局有效 batch 16。

最终多种子比较按仓库评估规范对 SFT 与三个 seed 各运行 3 次确定性生成（`do_sample=false`、`temperature=0`）。早期 seed 42 的 1-repeat 结果只作为触发复现的 screen，不作为最终确认结果。

开发复现门要求：主场景三种子平均完整 case 动作增量至少 `+0.125`，全开发集平均 step 动作增量至少 `+0.10`，至少两个 seed 的主场景为正，实体聚类动作区间下界大于 0，主场景 verifier 至少 `+0.05`，且 step 2、单探针对照、其它类别和错误副作用守门项通过。

在未读取 test 时已固定最终确认门：若开发门通过，只运行一次 SFT 与三个 seed 的封存 test；主场景完整 case、总体动作和 verifier 的三种子均值必须为正，实体聚类动作区间下界必须大于 0，并保持同样的 step 2、对照、类别与副作用约束。

## 多种子最终结果

四个模型的 3 次确定性生成完全一致。主场景完整两步 case 的三种子平均增量为 `+0.3333`（seed42 `+0.375`、seed43 `+0.375`、seed44 `+0.250`）；总体动作率平均增量 `+0.1333`，实体聚类 95% CI `[+0.0917,+0.1778]`；主场景 verifier 增量 `+0.2147`。这些主效应均通过。

整体门仍失败：错误副作用动作从 baseline 的 11 次变为三种子均值 11.67 次。seed42/43 均为 11，seed44 为 13；新增两次均把 `pipeline_eval` 误路由成 `flux-image-generation`。按预注册规则不改阈值、不追加 seed、不打开 test。结论是“找到可复现的窄场景增长”，而不是“得到可安全上线的整体 Planner”。

## 人工阅读

先读 `data/datasets/planner_runtime_probe_curriculum_v1/HUMAN_REVIEW.md`，再读同目录的 `DATASET_CARD.md`。人工样例只展示 train；dev 用于门槛，test 在开发门通过前不展示也不读取。

## 结论边界

本研究只评估 Planner JSON 决策，不执行 Flux、Adela 或完整 pipeline，也不声称改善检索或视觉模型质量。
