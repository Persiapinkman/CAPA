# Dataset Card: planner_stateful_retrieval_v2

## 研究目的

验证 GRPO 能否在比 v1 更复杂、但仍可解释的 Planner 状态机上产生严格动作增长。核心端点不是 JSON 格式，而是模型能否根据历史工具结果，在 `rag_answer`、`re_question`、`answerer` 和 `end` 之间切换。

## 数据规模

| Split | 实体组 | Case | Step | 用途 |
|---|---:|---:|---:|---|
| train | 72 | 504 | 1080 | GRPO 训练 |
| dev | 16 | 112 | 240 | 支持审计、超参数筛选 |
| test | 32 | 224 | 480 | 开发复现门通过后一次性确认 |

全部 1800 个 step 都有不同的 prompt/completion 对；没有用复制行伪装数据扩增。train/dev/test 的实体、模板、case ID 和精确 query 均不重叠。

使用当前固定 tokenizer 审计时，train/dev/test 的最长 prompt 分别为 4913/4893/4898 tokens，均低于训练上限 6144；不存在 prompt 截断样本。

## 场景组成

- `rag_double_miss_recovery`：五步主场景，动作序列为 `rag_answer -> re_question -> rag_answer -> re_question -> rag_answer`。
- `rag_single_miss_recovery`：三步单次空检索恢复，用于检查复杂场景训练是否保留简单状态机。
- `rag_hit_then_synthesize`：检索成功后不能提前结束，需转到 `answerer(mode=rag_evidence)` 综合。
- `coref_rewrite_then_rag`：先从历史补全实体，再检索。
- `direct_rag_guardrail`：无歧义时直接检索，防止无条件改写。
- `memory_hit_end_guardrail`：已有充分证据时结束，防止重复工具调用。
- `general_answer_guardrail`：通用问题交给直接回答，防止无条件 RAG。

## 为什么这样训练

v1 的 40 步动作奖励实验虽然有 32/40 个有效奖励批次，但 GRPO LoRA 的有效 `BA` 范数只有 `0.01196`，开发集采样动作与 SFT 完全相同（均为 `147/192`）。这排除了“继续复制同一批 240 行”的合理性。

v2 同时解决两个问题：

1. 用 1080 个独立状态 prompt 增加实体、表述和轨迹位置的覆盖，而不是增加重复权重。
2. 用五步轨迹制造明确的中间状态区分，让奖励直接作用于“当前应该检索还是改写”。

每个 step 采用动作主导奖励：动作匹配权重 `0.75`，错误动作的 task reward 上限为 `0.20`；外层训练权重为 task `0.95`、format `0.05`。因此错误动作的总奖励最高为 `0.24`，不能靠参数或格式部分分混过 GRPO 排序。

## 实际五步例子

用户问题：`对露天矿运输道路液体泄漏检测的模型版本做最多三轮内部查询，两次空结果分别触发一次范围收窄。`

| Step | 已知 observation | 期望动作 | 关键参数 |
|---:|---|---|---|
| 1 | 尚未检索 | `rag_answer` | `finish_after_tool=false` |
| 2 | 第一轮未命中 | `re_question` | `rag_miss`, round 2 |
| 3 | 第一轮改写完成 | `rag_answer` | `finish_after_tool=false` |
| 4 | 第二轮仍未命中 | `re_question` | `narrow_scope`, round 3 |
| 5 | 最终改写完成 | `rag_answer` | `finish_after_tool=true` |

完整 case 在 `training/planner_grpo_seed_v1/cases/planner_stateful_retrieval_v2_dev_cases.jsonl` 的 `SRV2-DEV-MISS2-001`。

## 完整性与边界

- 测试集状态：训练、种子和开发门锁定期间保持封存；三种子开发复现门通过后于 2026-07-12 一次性打开并完成确认。
- bootstrap 单位：`entity_id`，同一实体下七类场景不能当作独立样本。
- observation 是确定性 mock；正结果只证明 Planner 状态决策可学，不证明真实 RAG 工具质量。
- 文本是受控合成模板；通过测试后仍需在脱敏真实流量上复现。

哈希、分布和相似度诊断见 `manifest.json`。人工审阅顺序见 `HUMAN_REVIEW.md`。
