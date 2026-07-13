# ACE RAG

`ace-rag` 是 v3 Playbook sidecar。它只实现 ACE Playbook、反馈记录和答案编排，不复制 `gbrain-rag` v2 的 gbrain SQLite、实体图谱、三源检索、结构化统计和数据更新流程。


## 与 gbrain-rag 的关系

`gbrain-rag` 是事实检索层，负责 document/table/adela 三源召回、实体和图信号、结构化字段与 evidence 排序。`ace-rag` 位于它之前和之后：先从 Playbook 检索策略记忆，把命中的扩展词、数据源提示和回答约束注入到 v2 retrieve 请求；拿到 gbrain-rag 的 evidence 后，再用 evidence + Playbook 生成最终回答。

简化链路如下：

```text
用户问题
  -> ACE Playbook lexical search
  -> 生成 query_expansion_terms / source_hints / answer rules
  -> 调用 gbrain-rag /retrieve 获取事实证据
  -> online_feedback 精确纠错覆盖或 LLM 答案生成
  -> 记录 qa_run，等待反馈闭环
```

因此 Playbook 不替代知识库事实：除 `online_feedback` 的用户纠错外，Playbook 主要表达检索策略、字段绑定、路由偏好、回答风格和业务约定；模型名、版本、OID、did/rid、指标等事实仍以 gbrain-rag evidence 为准。

## 功能边界

v3 负责：

- Playbook item 存储、检索和 seed 导入。
- 用 Playbook 生成 `query_expansion_terms` 和 `source_hints`。
- 用 keyword/BM25 风格的 lexical 检索命中 Playbook 规则，不对 Playbook 使用 embedding RAG。
- 调用 v2 `/api/v1/rag/retrieve` 获取事实证据。
- 用事实证据 + Playbook 生成答案。
- 记录 `qa_runs`、用户反馈和 pending Playbook 操作，并支持在线反馈即时转成 Playbook 记忆。
- 在在线反馈写入记忆后自动触发 Playbook 整理，辅助 memory 管理、去重和规则沉淀。

v3 不负责：

- document/table/adela 数据加载。
- gbrain facts brain 和索引构建。
- 实体抽取、FTS、BM25、向量召回、graph score、structured score。

## Playbook 反馈闭环

当反馈能关联到已有 `qa_run`，且包含纠错答案、缺失证据或备注时，ACE 会即时写入一条 `online_feedback` Playbook 记忆。随后服务比较“上次自动整理后的 active playbook 条数”和“本次写入后的 active 条数”，当差值达到阈值时自动调用 LLM 整理器。

自动整理不需要 preview，会直接应用到 SQLite：

- LLM 会把重复规则合并，把单次在线纠错沉淀为更稳定的 `query_expansion`、`source_routing`、`field_binding`、`answer_strategy` 或 `aggregate_semantics` 规则。
- 新规则写入 `playbook_items`，被覆盖的旧条目会被置为 `inactive`。为了保留精确纠错能力，本轮刚写入的 `online_feedback` 条目不会被同一轮整理立即退役。
- 整理操作写入 `playbook_operations`，状态通常为 `applied`、`noop` 或 `skipped`；整理基线写入 `playbook_state`。
- 如果 LLM 未配置或调用失败，本轮会记录一次 `skipped`，并更新整理基线，避免每条反馈都重复触发失败调用。

## 当前实现状态

当前版本是可运行的最小 sidecar：

- Playbook 检索使用 keyword + BM25 lexical scoring，并叠加 intent、section、confidence 和在线反馈权重。
- 答案生成支持 OpenAI-compatible LLM；未配置 LLM 时降级为证据片段。
- 反馈会写入 `qa_feedback`，生成 `REVIEW_FEEDBACK` pending operation，并对纠错/缺失证据反馈即时生成 `online_feedback` Playbook item 强化下次 query。
- 在线反馈写入后会按阈值触发 LLM 自动整理，整理结果直接写回 Playbook，并通过 `playbook_operations` 和 `playbook_state` 留痕。
- `rebuild_playbook_embeddings.py` 仍为兼容占位脚本；Playbook 自身不依赖 embedding 检索。
