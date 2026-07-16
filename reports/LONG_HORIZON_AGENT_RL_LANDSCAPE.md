# CAPA 长程 Agent RL 场景调研与路线建议

> 调研日期：2026-07-13
> 状态：研究与方案设计，不包含数据扩增、训练、GRPO rollout 或测试集解封
> 适用范围：当前 CAPA Demo Agent、RAG、Qwen/Rex-Omni 检测、pipeline、migration advisor 与 Adela 工具体系

## 结论先行

CAPA 最值得做的长程 Agent RL 场景不是“把 9 个工具串得更长”，而是：

**知识驱动的视觉模型迁移/故障处置工单**。

Agent 需要在多轮用户交互中完成需求澄清、RAG 取证、样例图探针、冲突诊断、评测或迁移路径选择、风险控制和结果核验。工具返回会改变后续最优动作；任务允许多条有效路径；最终结果可由项目状态、证据覆盖、评测结论和副作用记录程序化验证。

这是目前最接近公开研究中 `tau-Knowledge + ToolSandbox + AppWorld` 的 CAPA 原生场景：

- `tau-Knowledge` 把知识检索、工具发现、政策推理、用户交互和可验证状态变更放在同一任务里，而不是只评 RAG 命中率。
- `ToolSandbox` 强调隐式状态依赖、失败恢复、必须发生的 milestone 和绝不能发生的 minefield。
- `AppWorld` 用起止数据库 diff 同时检查目标完成和 collateral damage，正好对应当前 GRPO 已暴露的错误 Flux 副作用。

当前两步 `detection -> migration_advisor` 结果证明了“observation 后继续决策”存在可学信号，但它仍是窄的两步技能，不是完整的 long-horizon 能力。下一阶段的研究对象应从“动作序列拟合”升级为“在有状态环境中完成工单”。

## 当前能力边界

### 已知事实

- Demo 暴露 9 个工具，但 Planner 并不拥有所有状态转移。RAG miss 后的改写、重试和第三次 miss 迁移询问主要由 Orchestrator 固定执行。
- `AGENT_MAX_STEPS` 当前为 10。仅提高这个数字不会产生长程任务，只会允许更长的循环。
- 历史审计包含 295 个 query、1083 个 trajectory step；最常见长序列是三轮 RAG miss/rewrite 闭环。它主要反映固定编排，而不是自由策略。
- 历史图像 session 只有 7 个，`qwen_detection`、`pipeline_eval`、`migration_advisor` 的自然使用证据很少，不能直接当成长程 RL 的经验分布。
- 2026-07-14 已通过 SSH 端口转发完成 6061/6062 health、生成式问答和 Demo `rag_answer` 端到端复验；生产索引与 LLM 回答链路均可用。
- Rex-Omni 与 Qwen 已用同一真实图片完成 raw skill 和 Demo executor 复验，均返回 2 个合法框及标注图。正常 observation 已确认，但零框、错误、超时和 schema 异常 fixture 仍需冻结后才能构成长程 RL 环境。

### 对 RL 的含义

现有运行时更接近“workflow + 少量 Planner 决策”，而不是完全由 policy 控制的 Agent。只有满足以下条件的决策才能作为 RL action：

1. 动作由训练 policy 实际拥有，而不是随后被 Orchestrator 覆盖。
2. observation 会改变后续可选动作或动作价值。
3. 环境能重置、重放并验证状态。
4. 失败、重试、停止和副作用都有可观察结果。

## 什么才算适合 Long-Horizon Agent RL

“轮数多”不是充分条件。一个适合长程 RL 的场景至少应同时具备以下性质：

| 性质 | 必要原因 | CAPA 中的具体形式 |
|---|---|---|
| 顺序依赖 | 早期动作影响后续信息和可行路径 | 先确定标签/模型约束，探针结果再决定是否评测或迁移 |
| 部分可观测 | Agent 必须主动取证，而非一次读完答案 | 内部模型资产、文档、图片目标和平台能力分散在 RAG 与工具返回中 |
| 条件分支 | 相同请求因 observation 不同而走不同路线 | 有框、零框、冲突框、超时分别触发总结、补证、换模型或有限重试 |
| 多条有效路径 | 防止训练成固定链复读 | 可先查资产再探针，也可先澄清标签；只要最终状态正确均可通过 |
| 可恢复错误 | 长程能力的关键是纠错而非永不犯错 | 参数错误、服务超时、空结果、过期证据和用户纠正 |
| 可验证终态 | 稀疏奖励必须可信 | 决策、证据、评测、预算、审批与风险字段满足目标断言 |
| 有意义的约束 | 否则“多调用最强工具”会成为捷径 | 调用预算、延迟、Flux/Adela 权限、隐私和只读限制 |
| 可重置环境 | 在线 RL 需要隔离且可复现 | 每个 rollout 独立项目 ledger、工具 fixture 或 session-scoped sandbox |
| 难度有梯度 | 全成功或全失败都没有稳定组内信号 | 从单分支工单逐步增加冲突、故障、跨阶段记忆和审批 |

一个操作性定义是：任务至少跨越“取证、诊断、行动、验证”四个阶段，包含多个真正改变 belief/state 的决策点，并且不能被单个工具或预设固定链完成。最初可以是 8-20 个有效决策；以后再扩到跨 context window 的多 session 任务。不要用空洞的反思或重复检索人为凑步数。

## 业界与学术界可参考场景

### 环境与基准

| 工作 | 场景与关键设计 | 对 CAPA 的可复用经验 | 主要局限 |
|---|---|---|---|
| [WebArena, ICLR 2024](https://arxiv.org/abs/2307.13854) | 812 个真实网站任务；独立可复现网站；以数据库/API/页面状态检查功能正确性 | 高层任务、动态环境、结果验证优于参考轨迹匹配 | 网页 GUI 噪声不是 CAPA 的核心问题 |
| [WorkArena++](https://arxiv.org/abs/2407.05291) | 682 个企业工作流；原子任务组合；oracle、validator、seeded curriculum、任务级隔离 | 可组合 oracle/validator，显式和隐式指令形成难度梯度 | 纯原子链组合容易退化成固定 workflow，必须增加结果条件分支 |
| [AppWorld](https://arxiv.org/abs/2407.18901) | 9 个模拟应用、457 个 API；平均 9.5 个 API；起止状态 diff 和 unit tests | `expected changes + allowed changes` 可同时奖励完成、惩罚 collateral damage | 高保真环境建设成本高 |
| [ToolSandbox](https://arxiv.org/abs/2408.04682) | 有状态工具、on-policy 用户、milestone DAG、minefield；平均 13.9 轮 | 适合表示前置条件、恢复、澄清、禁止动作和部分进度 | milestone 人工标注昂贵；用户模拟仍有约 8% 错误 |
| [tau-bench](https://arxiv.org/abs/2406.12045) / [tau2-bench](https://arxiv.org/abs/2506.07982) | 用户、工具、政策、隐藏数据库；按最终状态评估；`pass^k` 衡量重复可靠性；dual-control 用户也能执行动作 | 用户不是静态 prompt；Agent 要引导用户、遵守规则并稳定完成同一语义的不同对话 | 后续 [SABER / tau2-Verified](https://arxiv.org/abs/2512.07850) 发现任务、政策、数据库和评估不一致，说明 benchmark 也必须审计 |
| [tau-Knowledge, 2026 preprint](https://arxiv.org/abs/2603.04370) | 约 700 篇内部文档、51 个可发现工具、97 个工单；平均需 18.6 篇文档和 9.52 次工具调用，最多 33 次 | 与 CAPA 最接近：RAG 只是信息动作，最终奖励来自正确且合规的状态变化 | 尚属预印本；领域为银行客服，需验证迁移到视觉研发工单的外部有效性 |
| [Tool Decathlon, 2025 preprint](https://arxiv.org/abs/2510.25726) | 32 个应用、604 个工具、108 个任务；平均约 20 轮；执行脚本严格验证 | 证明专业工具任务需要同时看成功率、turn 数和跨应用状态 | 规模仍小，且复杂运行环境使重复成本很高 |
| [CRMArena-Pro](https://arxiv.org/abs/2505.18878) | 销售、服务、CPQ 企业任务；persona、多轮交互和保密约束 | 内部研发 Agent 也必须把隐私、权限和多轮信息披露作为任务约束 | 主要是评估，不直接解决 RL 信用分配 |
| [SWE-bench](https://arxiv.org/abs/2310.06770) / [SWE-Gym](https://arxiv.org/abs/2412.21139) | 长上下文代码修改；容器化运行；测试作为终态反馈 | 可执行验证和独立环境是强奖励来源 | 测试不完备、污染和 solution leakage 会制造虚假增长，不应把“测试通过”等同真实正确 |
| [OSWorld](https://arxiv.org/abs/2404.07972) / [AndroidWorld](https://arxiv.org/abs/2405.14573) | 真实桌面/移动系统；每题独立 setup、success check、teardown | 环境初始化、状态检查和 teardown 应成为 benchmark 一等公民 | GUI grounding 与 CAPA 的主要决策瓶颈不同 |

`tau-Knowledge` 是当前最接近 CAPA 的公开参照：其在线检索配置的最好 `pass^1` 约为 25.52%，即使直接提供 gold documents 也只有约 39.69%。这组结果表明“检索到正确材料”仍不足以完成需要政策推理、用户协作和工具状态变化的工单；CAPA 不应把 RAG answer quality 当作 long-horizon 任务的最终奖励。

### 训练与系统方法

| 工作 | 可借鉴结论 | 不应直接照搬的部分 |
|---|---|---|
| [WebRL, ICLR 2025](https://arxiv.org/abs/2411.02337) | 从失败任务生成适配当前能力的 curriculum；保留成功 replay；限制 policy drift | 它主要用最终 HTML 的 outcome reward model。CAPA 能做状态断言时，不应让 LLM judge 成为主奖励 |
| [RAGEN](https://arxiv.org/abs/2504.20073) | 多样初始状态、每题多 rollout、适中 action budget、频繁刷新 on-policy 数据；低 reward variance 会导致训练塌缩 | 结论主要来自符号环境和 WebShop；不能视为企业工具场景的直接复现证据 |
| [GiGPO, NeurIPS 2025](https://arxiv.org/abs/2505.10978) | 同时使用 episode-level 与相同状态下的 step-level relative advantage，缓解长程信用分配 | CAPA 的文本 observation 很难精确匹配；必须先定义规范化项目状态或结构等价状态 |
| [Agent Lightning](https://arxiv.org/abs/2508.03680) | 将 agent trace 表示为 transition，解耦运行时与训练，复用 tracing/observability | 框架当前实验仍把最终 return 等值分配给各 action；它提供接口，不等于已经解决信用分配 |
| [ARTIST](https://arxiv.org/abs/2505.01441) | RL 可改善工具选择、参数和错误后的自修正 | 其 function-calling 训练把 user turn 固定为 1、completion 仅 2048 token，更接近 multi-step 而非 long-horizon |
| [PROVE, 2026 preprint](https://arxiv.org/abs/2606.03892) | live stateful MCP、rollout 隔离、先采真实状态再合成 query、replay validation、程序化奖励和自适应效率惩罚 | 合成工具链仅 2-5，`max_turns=3`；是优秀基础设施参考，不是长程规划证据 |
| [EnterpriseBench Corecraft, 2026 preprint](https://arxiv.org/abs/2602.16179) | 高保真企业环境、专家 rubric 和真实 workflow 的 GRPO 报告了 held-out 与外部基准增长 | 同环境随机 split、LLM rubric judge、单模型单轮研究限制因果解释；结果有启发性但需要独立复现 |

### 生产工程经验

- Anthropic 将 workflow 与 agent 区分：固定、可预测的任务优先用 workflow；只有无法预先确定步骤且环境反馈决定下一步时才值得用 agent。其生产经验还强调 sandbox、停止条件和清晰的 agent-computer interface。[Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
- 长任务的核心不是无限 context，而是高信号上下文、结构化笔记、compaction、checkpoint 和可恢复的外部状态。[Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)、[Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- 多 Agent 只适合可并行且高价值的任务。Anthropic 报告其研究系统使用的 token 约为普通 chat 的 15 倍，而且共享状态依赖强的任务并不适合当前多 Agent 架构。CAPA 初始场景应先采用单 Agent + 有状态工具，不要用多 Agent 掩盖单 policy 的决策问题。[Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)
- Agent 评估必须区分 transcript 与 outcome。Agent 说“完成”不代表环境状态完成；多 trial、代码 grader、人工校准和持续读 trace 都是必要环节。[Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- [METR 的 time-horizon 研究](https://arxiv.org/abs/2503.14499)用人类完成同一任务的时间而不是 token/step 数衡量自主性，提醒我们报告“任务对应多少人工时间”，而不只报告 Agent 走了多少轮。

这些生产文章是工程经验，不是随机对照实验；它们适合指导 harness 和评估设计，不能单独证明某种训练算法有效。

## CAPA 候选场景排序

| 排名 | 场景 | 适合度 | 判断 |
|---:|---|---|---|
| 1 | 知识驱动的视觉模型迁移/能力验证工单 | 高 | 同时覆盖 RAG、视觉探针、评测、迁移建议、预算与审批；终态可验证；有真实业务价值 |
| 2 | 检测能力故障诊断与恢复 | 高 | 工具错误、零框、冲突结果、历史版本和回退构成真实分支；比迁移工单更窄、更容易先建环境 |
| 3 | 主动学习/数据扩增 campaign | 中高，后续 | 可形成“发现失败 slice -> 选样/生成 -> 评测 -> 再决策”闭环，是真长程；但奖励受数据质量、标注和模型训练噪声影响，第一阶段不宜直接上 |
| 4 | Adela 部署与 benchmark 工单 | 中，后续 | 最终状态明确且有工程价值，但不可逆/高成本动作多，必须先有 sandbox、审批和 rollback |
| 5 | 开放式技术调研/报告生成 | 低，作为辅助 | 多路径且长，但结果主观，容易让 LLM judge 奖励文风、长度或自信而非真实解决 |
| 6 | 固定三轮 RAG miss 或固定工具链 | 不适合 | 长度来自编排器，policy 没有真实选择；训练只会拟合不可控的离线标签 |

## 推荐场景：视觉模型迁移与能力验证工单

### 任务定义

一个典型请求不是“依次调用 RAG、Qwen、Rex 和 migration advisor”，而是：

> 我们需要在某部署环境中检测一个新目标。请确认现有模型是否支持；若证据不足，用给定图片做最小成本探针；比较候选能力，在满足质量、延迟、调用预算和权限约束的前提下给出可审计的接入或迁移结论。未经批准不要生成图片、跑完整 pipeline 或执行部署。

Agent 的职责是维护一个工单，从未知状态推进到以下终态之一：

- `ready`：有足够证据证明某条能力路线满足约束，报告包含可追溯证据与验证结果。
- `migration_required`：现有能力不满足要求，迁移建议、证据缺口和下一步验证均明确。
- `needs_data`：当前不能证明支持或不支持，明确缺少的数据 slice、标注或测试，而不是伪造结论。
- `blocked`：权限、服务、预算或输入缺失导致不能继续，已说明可解除的阻塞条件。
- `escalated`：证据冲突或风险超过自动决策边界，带完整上下文交给人工。

任务成功不要求唯一轨迹。比如先澄清目标标签再查 RAG，或先查看已有资产再补问部署约束，都可以；只要动作合法、信息足够且终态断言成立，就应通过。

### 为什么它是真正的长程决策

该场景至少包含四类相互依赖的决策：

1. **信息价值决策**：下一步应该问用户、查 RAG、查看资产还是做视觉探针。
2. **证据融合决策**：文档声称支持但探针失败时，不能简单相信任一方；要检查版本、标签语义、阈值和输入质量。
3. **行动与风险决策**：只读探针、完整 pipeline、图片生成和部署的成本与权限不同。
4. **停止决策**：证据何时足够，何时继续调用只会增加成本，何时必须转人工。

真正的难度来自隐藏状态和 observation 条件分支，而不是预先规定调用次数。同一个用户表述必须有多个环境变体：在变体 A 中 RAG 已给出新鲜且一致的支持证据，最优动作可能是小样本验证；在变体 B 中证据过期，最优动作是补查版本；在变体 C 中两个探针冲突，最优动作是诊断而非直接宣告成功。

### 项目状态而非无限聊天历史

建议为每个 rollout 建一个可重置的结构化 `project_ledger`。完整 transcript 用于审计，但 policy 的长期工作记忆应由以下规范化状态承载。

```yaml
request:
  intent: validate_or_migrate
  target_label: null
  deployment_target: null
constraints:
  min_quality: null
  max_latency_ms: null
  max_tool_calls: 12
  allowed_side_effects: []
assets:
  images: []
  datasets: []
  permissions: []
evidence:
  rag_items: []
  conflicts: []
  freshness: unknown
  provenance_complete: false
probes:
  qwen: not_run
  rexomni: not_run
evaluations: []
decision:
  status: open
  candidate_route: null
  blockers: []
approvals:
  flux: false
  pipeline: false
  deployment: false
budget:
  calls_used: 0
  cost_used: 0
tool_health:
  rag: unknown
  qwen: unknown
  rexomni: unknown
artifacts:
  migration_report: null
  rollback_plan: null
```

每个工具调用应产生 `before_state`、`action`、`observation`、`after_state`、成本和时间戳。这样才能做 transition-level 分析、状态等价分组和可重放评估；只保留自然语言 history 无法稳定判断 Agent 到底学到了什么。

### 必须覆盖的观测分支

| 环境返回 | 正确行为原则 | 常见错误捷径 |
|---|---|---|
| RAG 证据充分、新鲜且一致 | 引用证据，按风险决定是否做最小验证，不重复检索 | 不管证据质量固定重试三次 |
| RAG 无结果 | 改写一次或改查资产；记录知识缺口 | 把“没检索到”说成“不支持” |
| RAG 文档冲突或版本过期 | 比较来源、版本和适用范围，必要时升级人工 | 任取第一条或最长回答 |
| 检测返回高置信有效框 | 检查标签、坐标和输入对应关系，再更新候选结论 | 看到非空 JSON 就宣告成功 |
| 检测返回零框 | 区分目标确实不存在、模型能力不足、阈值/图像/标签问题 | 把零框直接等价为“图片中没有目标” |
| Qwen 与 Rex 结果冲突 | 检查模型版本、阈值和图像 slice；按预算选择补证或升级 | 多数投票或永远信某个模型 |
| 工具超时/可恢复错误 | 有限重试、替代工具或记录阻塞；避免相同无效循环 | 无上限重复相同调用 |
| 工具返回 schema 错误 | 不消费不可信字段，记录契约故障并安全降级 | 猜测 bbox、ID 或状态字段 |
| 用户补充或修改约束 | 更新 ledger，使旧证据按新约束重新判定 | 忽略更正，继续执行旧计划 |
| 请求完整 pipeline/生成/部署 | 先确认显式权限和成本，使用 dry-run/fixture 优先 | 未经批准触发有费用或不可逆动作 |

### 难度阶梯

| 等级 | 有效决策规模 | 任务构成 | 用途 |
|---|---:|---|---|
| L0 单工具回归 | 1-3 | 单次检索或探针，验证 schema、参数和基本终态 | 只做基础设施回归，不纳入 long-horizon 主结果 |
| L1 短工单 | 5-8 | 澄清、检索、一次探针、形成报告 | 验证环境和奖励是否可靠 |
| L2 条件工单 | 8-15 | 多证据、两模型探针、冲突/预算分支 | 第一阶段 long-horizon capability 集 |
| L3 恢复工单 | 12-20 | 服务故障、用户纠正、过期证据、有限重试和升级 | 训练恢复、停止与可靠性 |
| L4 campaign | 20-40 | 多图片、多 slice、数据缺口分析和迭代验证 | 环境成熟后研究更长信用分配 |
| L5 跨会话执行 | 40+ 或数小时人工等价时间 | 审批等待、部署、监控、rollback 与多 session memory | 最后阶段，必须先有强 sandbox |

第一版 benchmark 应以 L1-L3 为主。L4-L5 在工具状态、奖励和隔离还不可靠时只会把环境噪声误当成 policy 学习问题。

## 奖励与评估设计

### 先定义状态断言，再定义参考轨迹

参考轨迹只能用于调试和验证任务可解，不能作为唯一答案。每个任务实例应包含：

- `initial_state`：初始工单、资产、文档版本、权限、预算和工具健康状态。
- `target_assertions`：终态必须成立的业务条件。
- `expected_changes`：必须发生的状态变化。
- `allowed_changes`：可以发生但不影响正确性的变化。
- `forbidden_changes`：任何情况下都不能发生的副作用。
- `milestone_dag`：可部分计分且满足前置依赖的中间状态。
- `minefields`：危险动作、虚假结论或不可恢复的错误。
- `validated_solution`：至少一条经真实环境 replay 通过的解，不用于逐步模仿评分。

这直接借鉴 AppWorld 的状态 diff 和 ToolSandbox 的 milestone/minefield 思路。任务评测器本身也要有单元测试：应接受多条等价成功路径，并拒绝看似合理但破坏状态的轨迹。

### 分层而非可互相抵消的奖励

评估时使用字典序结果，而不是先把所有指标相加：

```text
(safety_gate, terminal_success, milestone_progress, recovery_quality, efficiency)
```

优先级如下：

1. **安全硬门**：越权 Flux、pipeline、Adela/部署动作，伪造资产 ID 或写入禁止状态，直接判失败；任务完成不能抵消。
2. **终态成功**：目标断言和状态 diff 全部成立，且 Agent 的完成声明与环境一致。
3. **里程碑进度**：只在未完成任务中区分是否取得真实、依赖关系正确的部分进展。
4. **恢复质量**：错误后是否诊断、有限重试、安全降级或正确升级，而不是碰巧在末尾成功。
5. **效率**：只有成功且安全的 rollout 才比较调用量、费用、延迟和上下文消耗。

GRPO 实现需要标量时，也应保留硬门：`unsafe -> fixed negative`，`safe but failed -> milestone-shaped range`，`safe and successful -> success range + bounded efficiency bonus`。效率奖励必须有上限，不能让少调用但错误的策略超过正确策略；也不能用固定“越短越好”迫使 Agent 跳过必要验证。可以按任务 oracle 或经审计的合理调用区间做自适应预算。

### 建议的里程碑和 minefield

| 类型 | CAPA 示例 |
|---|---|
| Milestone | 目标标签、部署环境、质量门槛和权限已澄清 |
| Milestone | RAG 证据带来源、版本、新鲜度和适用范围写入 ledger |
| Milestone | 根据不确定性选择了必要而非固定的 Qwen/Rex 探针 |
| Milestone | 对 bbox、零框、冲突或错误做了语义正确的解释 |
| Milestone | 结论、阻塞项、证据缺口和下一步写入可验证报告 |
| Recovery | schema 错误后未消费伪字段，并改为安全路径 |
| Recovery | 用户纠正标签后使旧证据失效并重新评估 |
| Minefield | 未获明确授权触发 Flux、完整 pipeline、Adela 或部署 |
| Minefield | 把零框直接宣称为目标不存在或模型一定不支持 |
| Minefield | 把 RAG 文本本身当成已经通过实测的事实 |
| Minefield | 对不可恢复错误重复完全相同的调用直到预算耗尽 |
| Minefield | Agent 声称完成，但 terminal state 或必要 artifact 不存在 |

### 核心指标

| 指标 | 解释 |
|---|---|
| Terminal success | 程序化终态通过率，是首要能力指标 |
| `pass^k`，建议 `k=3` 或 `4` | 同一任务独立运行 k 次全部成功的比例，暴露不稳定性 |
| Scenario success | 同一语义任务的多个隐藏状态变体全部正确，防止关键词捷径 |
| Unsafe mutation rate | 越权或 forbidden state change 的 episode 比例，独立硬门 |
| False-success rate | Agent 宣称成功但环境断言失败的比例 |
| Recovery success | 注入错误、用户纠正或服务恢复后最终正确处置的比例 |
| Evidence traceability | 结论中的事实能否映射到有效文档或工具 observation |
| Success@budget | 在调用、费用、延迟上限内的成功率 |
| Oracle regret | 成功策略相对可行低成本方案的额外成本，只在成功集上计算 |
| Horizon curve | 按必要决策数和对应人工完成时间分桶的成功率 |

同时报告均值、按任务实体/模板聚类的置信区间和失败类型。不能只报 step accuracy，因为长任务中多数简单步骤会掩盖关键分支失败；也不能只报 `pass@k`，因为大量采样可能掩盖单次运行不可靠。

### Grader 层级

1. 程序化 grader 检查 ledger、数据库 diff、artifact、预算、权限和工具调用事实。
2. 确定性规则检查引用是否存在、版本是否匹配、bbox/schema 是否有效。
3. 模型 grader 只评沟通质量、报告可读性和难以形式化的解释，不决定核心成功。
4. 人工审阅校准模型 grader，并专项审计高风险、争议和 reward-hacking 样例。

不要奖励可见 chain-of-thought 的长度或某种固定措辞。应奖励可审计的外部状态、简洁理由字段和证据引用；否则模型可能学会“写得像在深思”而不是正确行动。

## 数据集构造原则

### 数据单元是任务世界，不是对话文本

每条训练/评测样本应先生成隐藏环境状态，再生成用户表达：

1. 从经审计的模型、文档版本、图片 slice、工具健康、权限和预算中采样 `initial_state`。
2. 采样需要学习的 branch topology，例如“文档支持但探针冲突”或“用户中途提高质量门槛”。
3. 构造 target、expected/allowed/forbidden changes、milestone DAG 和 minefield。
4. 用规则或 oracle 检查任务可解，并 replay 至少一条有效路径。
5. 最后才将状态渲染为自然用户对话，可生成多种措辞但不能泄漏隐藏答案。

如果先写一段用户问题，再让大模型自由编造工具结果，数据容易出现不可解任务、矛盾 schema、答案泄漏和虚假“长链”。PROVE 的“先读取 live state，再生成任务并 replay 验证”值得借鉴，但 CAPA 需要比其 2-5 工具链更丰富的条件分支。

### 如何使用真实 Demo 记录

`demo/sessions` 与 `demo/llm_debug` 应只用于提取去标识化模式：

- 用户如何描述目标、补充图片、纠正标签和追问结果。
- RAG 回答长但未完全回答、连续 miss、工具错误和 finish 选择等形态。
- 常见意图、语言风格、歧义类型和实际工具分布。
- 不复制原始 query、回答、session/client 标识、资产 ID、图片地址或内部 RAG 文本。

真实记录提供语言和失败形态，不提供 ground truth。任务真值必须由独立的环境状态与 verifier 产生。

Rex-Omni 和 Qwen 服务上线后，应先采集并冻结少量经脱敏的实际 observation 契约，包括正常框、零框、错误、超时和字段边界。本轮“按正常返回假设”只用于方案推演，不应被写成训练真值；在未看到真实 schema 前自由编造语义结果会把模型训练到一个不存在的环境上。

### 对比组与课程

一个高价值 task family 应同时包含：

- 同一用户表述、不同隐藏 observation，要求后续动作不同。
- 同一隐藏状态、多种表述，要求终态一致。
- 近似请求但权限或预算不同，要求副作用决策不同。
- 成功、零框、冲突、可恢复错误和不可恢复阻塞等结果变体。
- 一条捷径似乎可行但会触发 minefield 的 hard negative。

课程不应只按预设步数排序，而应按当前 policy 的结果分层：稳定全对的任务保留作 replay/回归，稳定全错的任务先拆解或降低难度，重点训练同组 rollout 有成功也有失败的任务。该原则与 WebRL 的失败驱动 curriculum、RAGEN 对 reward variance 的观察一致。

### 用户模拟与关键状态

关键用户行为应由规则或有限状态机控制，例如只有 Agent 正确询问权限时用户才提供授权，Agent 误解标签时用户会纠正，信息已提供时不重复回答。LLM 只负责措辞和自然度，不能决定隐藏真值、权限或任务是否通过。ToolSandbox 已显示即使有知识边界和示例，纯 LLM 用户模拟仍会产生可观测错误，因此模拟器也必须有一致性测试。

### Split 隔离

禁止按对话行随机切分。至少同时隔离：

- 业务实体与目标标签族。
- prompt 模板和语言变体生成器。
- 图片来源、场景与 failure slice。
- RAG 文档、版本和证据组合。
- 工具 fixture、错误码和 branch topology。
- 任务生成 seed 与用户 persona。

开发集可用于选任务和超参；能力 test 在环境与指标冻结后密封；另保留一个不会被训练循环频繁触碰的 regression set。真实只读样例应作为最后的外部有效性检查，不能混入合成训练或被反复人工调参。

## GRPO 在这个场景中的角色

GRPO 可以作为基线，但“用了 GRPO”不是研究问题。真正的问题是：在同一初始工单的多个 rollout 中，是否存在足够的成功/失败方差，并且奖励能把终态差异归因到关键状态转移。

标准 episode-level GRPO 有三项风险：

1. 长轨迹最终奖励稀疏，早期关键取证动作和后期报告 token 获得近似相同信用。
2. 同组 rollout 全成功或全失败时相对优势接近零，复杂任务无法学习。
3. policy 更新后轨迹分布变化快，旧离线轨迹的 observation 与当前 policy 不匹配。

因此环境成熟后的算法比较应至少包含：

| 方法 | 目的 |
|---|---|
| SFT / prompting baseline | 判断是否根本不需要 RL，并给出能力下限 |
| Episode GRPO + terminal reward | 最简单、可解释的 RL 基线 |
| GRPO + 受约束 milestone shaping | 判断可信中间状态是否缓解稀疏奖励 |
| GiGPO 类 state-anchored advantage | 在规范化等价状态上比较局部动作价值 |
| Transition/critic 类 credit assignment | 判断显式状态转移是否优于整段均匀信用 |

只有在 `project_ledger` 能可靠地把不同自然语言 history 映射到规范化状态后，state-anchored 方法才有意义。Agent Lightning 可作为 tracing 和训练解耦接口参考，但不能把它的接口能力误写成信用分配已经解决。

训练时应使用新鲜 on-policy rollout、同一初始状态多次采样、适中 action budget 和 KL/策略漂移监控。首先证明 L2 条件分支在安全硬门下优于 SFT，再增加 L3 故障恢复；不要一开始把 40 步任务、用户模拟、真实服务噪声和新算法同时引入，否则无法定位增益来源。

## 不建议的捷径

- 把 `AGENT_MAX_STEPS` 从 10 改大，然后把更长 transcript 称为 long-horizon。
- 把现有固定三轮 RAG retry 当成 Agent RL；这些转移当前由 Orchestrator 拥有。
- 为所有任务规定相同工具序列，再用 exact action match 奖励模仿。
- 用一个 LLM judge 同时充当任务生成器、用户、reward 和最终裁判。
- 把“RAG 找不到”“检测零框”或“Agent 自称完成”直接当成业务真值。
- 在 Rex/Qwen 尚未验证真实 observation 契约时大规模合成虚构返回。
- 只做随机 row split、单 seed 和单次成功率，或在开发过程中持续查看 test。
- 让训练 rollout 调用真实 Flux、完整 pipeline、Adela 或部署环境，而没有 fixture、审批、隔离和 rollback。
- 把更长的 chain-of-thought、反思次数或工具调用次数作为能力代理指标。

## 开始实验前的门槛

在以下条件全部满足前，不建议扩增 long-horizon 训练集或启动 GRPO：

1. RAG、Qwen、Rex 的 Demo 实际 observation schema 均已在线复验，正常、零结果和错误分支有冻结 fixture。
2. 每个 rollout 使用独立、可重置的 `project_ledger` 或 session sandbox，并能导出完整 state diff。
3. 所有副作用工具有 dry-run/fixture、明确授权位、预算和 rollback；默认禁止真实执行。
4. 每个任务有 `initial_state`、target、expected/allowed/forbidden changes、milestone/minefield 和 replay 通过的解。
5. 任务与 grader 经至少两名独立审阅者检查可解性、歧义、政策一致性和奖励漏洞。
6. 基础模型在目标难度上产生混合结果，而不是接近全对或全错；同组 reward variance 足够。
7. 实体、模板、图片、文档、fixture 和 branch topology 的 split 隔离已自动检查。
8. capability、regression、dev 和 sealed test 分开，指标与安全硬门在看 test 前冻结。

建议先建 20-30 个高质量、人工审计的 L1-L2 任务来验证环境，而不是先合成上千条对话。这里的目标是发现 verifier 和状态设计错误，不是得到可发表的训练曲线。环境通过对抗审计后，再按 task family 扩展并开展预注册实验。

## 教授意见与研究主张

CAPA 有机会形成有研究价值的 Agent RL 工作，但论文主张不应是“GRPO 让工具调用准确率提高”，而应是：

> 在知识不完整、视觉 observation 不一致、成本与副作用受约束的模型迁移工单中，结构化状态、程序化终态奖励和状态条件信用分配是否能稳定提高长程任务成功率与恢复能力，并保持跨实体、跨工具结果和跨 horizon 的泛化。

最重要的科学变量是环境状态与可验证因果链，而不是模型生成了多少思考 token。第一项严肃实验应隔离一个变量：相同任务、相同基础模型、相同 rollout 预算下，比较 terminal-only GRPO 与 milestone/state-aware credit assignment；同时把安全变异率、`pass^k` 和跨隐藏状态的 scenario success 设为共同主指标。

如果 L2 已经由强提示或 SFT 稳定解决，就不应为了使用 RL 人为增加长度，应转向 L3 的错误恢复或 L4 的主动数据闭环。如果 verifier 无法可靠地区分“能力不足”和“样例中目标不存在”，则暂时没有可信 RL 问题，应该先完善环境和标注。一个较小但因果清楚、可复现且没有奖励漏洞的研究，价值高于一个工具很多、轨迹很长但终态不可证的系统。
