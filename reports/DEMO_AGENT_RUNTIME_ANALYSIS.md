# Demo Agent 运行时分析

## 结论

CAPA Demo 是一个面向公司视觉能力服务的多工具 Agent。它负责把自然语言、图片和会话状态路由到知识库问答、通用回答、图像生成、两种开放集检测、完整检测评测、迁移顾问或 Adela benchmark，并把中间 observation 持久化后继续规划。

它不是完全由 LLM 自由控制的 Agent。Planner 只拥有运行时允许变化的决策；RAG miss 重写/重试、第三次 miss 后询问迁移报告等闭环由 Orchestrator 固定执行。

## 能力服务复现结论

| 范围 | 结论 |
|---|---|
| 9 个工具静态 schema、注册和执行分支 | 已复现 |
| Demo HTTP `/health`、能力清单与 `/run` NDJSON | 2026-07-14 再次通过真实 RAG/Qwen/Rex 端到端复验 |
| Planner 和 Answerer 模型请求 | 已通过真实网关冒烟 |
| RAG 服务接口 | 6061/6062 health、生成式问答与 Demo `rag_answer` 均通过；生产索引含 5,673 chunks |
| Rex-Omni / Qwen 检测 | 同一真实图片的 raw skill 与 Demo executor 均通过，各返回 2 个合法框和标注图 |
| Flux | 默认未执行，避免凭据化费用动作 |
| Adela | 本机无可用 CLI/登录态，未执行部署 benchmark |

因此当前可准确表述为“Demo Agent 的静态能力契约、运行时主干、RAG、Qwen 单图检测和 Rex-Omni 单图检测已在线复现”。不能表述为“所有能力均已复现”，因为 Flux、完整 pipeline 和 Adela 仍未做有副作用的端到端验收。详细证据见 `reports/DEMO_CAPABILITY_LIVE_CHECK_2026-07-14.md`。

## 历史记录意味着什么

初始 `demo/sessions` 审计解析 271 个历史文件；能力复现又产生 1 个显式 `smoke` session，因此最新报告为 272。目录混合浏览器使用、迁移数据、Codex eval 和 synthetic smoke cohort，不能把全部内容当作自然用户分布。真实会话最常见路径是三轮 RAG miss 闭环；`demo/llm_debug` 还包含大量 GRPO eval 与 capability smoke 调用，也不能直接作为生产流量统计。

未来合成数据应只使用聚合模式和去标识化意图，不复制原始 query、回答、session/client 标识、资产 ID 或内部 RAG 文本。当前 `planner_runtime_probe_curriculum_v1` 遵守这一边界。

## GRPO 场景判断

以下两步场景是已经完成的窄技能实验结论，不是下一阶段 long-horizon 研究的最终推荐场景。新的场景选择与实验前置门槛见 `reports/LONG_HORIZON_AGENT_RL_LANDSCAPE.md`。

当前最合适的增长场景是 `qwen_detection(finish=false) -> migration_advisor(finish=true)`，并以几乎同文的 `qwen_detection(finish=true)` 单探针对照。它具备四个必要条件：

1. 两个动作都由当前 Planner 控制，不会被编排器覆盖。
2. 首步工具相同，但终止语义不同，可以排除只学关键词路由的捷径。
3. 工具 observation 会改变下一步状态，属于真正的两步 Agent 决策。
4. 动作、参数、终止标志和错误副作用均可程序化验证。

三种子最终 dev 结果支持这个窄场景：完整 case 动作率由 baseline `0.625` 提升为 seed42 `1.000`、seed43 `1.000`、seed44 `0.875`，平均增量 `+0.3333`；总体 step 动作平均增量 `+0.1333`，实体聚类 95% CI `[+0.0917,+0.1778]`。但 seed44 额外把两个完整评测请求误路由到有副作用的 Flux，导致错误副作用三种子均值增加 `+0.6667`，预注册整体开发门失败。因此 test 未打开，模型不作安全推广。

## 后续实验优先级

当前实验完成后不应继续在同一 test 上扩数据或选 checkpoint。下一阶段应新建实体、模板和测试集，按以下顺序提高难度：

1. **结果条件分支**：让同一视觉探针分别返回“高置信有效框、零框、格式错误/超时”，要求 Agent 选择直接总结、进入迁移顾问或有限重试。主指标是完整轨迹成功率，必须同时报告错误副作用和工具预算。
2. **有噪历史 replay**：从真实 session 只提取去标识的状态类型和工具结果形态，注入矛盾证据、长 answer 但 `fully_answered=false`、过期资产等情况；评估证据忠实度，而非仅动作匹配。
3. **成本约束规划**：为 Flux、pipeline、Adela 和视觉探针赋予固定成本，奖励采用任务成功减超预算惩罚。必须与“始终调用最强工具”的策略比较 Pareto 前沿。
4. **外部真实评估**：建立只读、人工双标、不可训练的少量去标识真实 query 集，按意图和实体聚类置信区间报告；合成 test 通过不等价于真实增长。
5. **影子流量验证**：在不执行副作用工具的 shadow 模式比较 SFT 与 GRPO 决策，仅在人工否决率、越权动作和成本均不恶化后考虑在线小流量。

若目标是训练 RAG miss 的改写/重试策略，必须先重构运行时，把这些动作的所有权从 Orchestrator 交给可验证的 policy；否则继续训练这些标签只会优化一个部署时不可控的离线目标。

当前 Qwen2.5 合并 tokenizer 在 Transformers 4.57.6 下会提示 `fix_mistral_regex` 警告。SFT baseline 和三个 GRPO seed 使用完全相同的 tokenizer，因此本轮配对效应未混入 tokenizer 差异；但部署前应在新的开发集上单独做开关等价性审计，不能在本轮封存 test 结果之后据此重选模型。

## 人工阅读顺序

1. `data/datasets/planner_runtime_probe_curriculum_v1/HUMAN_REVIEW.md`：先看真实 train 样例、训练理由和人工拒绝条件。
2. `data/datasets/planner_runtime_probe_curriculum_v1/DATASET_CARD.md`：看来源、规模、split 隔离和使用边界。
3. `reports/planner_runtime_probe_curriculum_v2_multiseed_dev3x_gate.md`：看三种子增长、失败的副作用守门项和 test 未开原因。
4. `experiments/studies/planner_runtime_routing_grpo_v1/runtime_probe_multiseed_replication_v1.json`：看预注册门槛及机器可读结果。
5. `reports/demo_agent_capability_reproduction.md`：看哪些能力已真实冒烟、哪些只有代码契约。

日常人工审阅应以第一份文档为入口；模型结论以 study gate 和最终 dev/test 报告为准。
