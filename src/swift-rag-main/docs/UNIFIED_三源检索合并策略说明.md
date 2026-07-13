# unified 方法三源检索合并策略说明

最后更新：2026-04-21

本文说明 unified 方法里，部署文档（document）、Excel 表格（table）、Adela 数据（adela）当前的检索与合并策略。

## 1. 总体流程

实现入口：`src/api/routes.py`

- `unified_query`：`query_unified_with_gateway()`
- `unified_retrieve`：`retrieve_unified_with_gateway()`

流程分两段：

1. 各数据源独立检索（并行执行）。
2. 三路结果做 RRF 融合，得到统一证据列表。

区别：

- `unified_query` 默认 `route_with_llm=true`，会先做一次 LLM 路由决定检索哪些源。
- `unified_query` 默认 `stream=false`，返回标准 JSON；当 `stream=true` 时改为 SSE 流式分片输出（最后一条 `data: [DONE]`）。
- `unified_retrieve` 不做 LLM 路由；默认检索所有 enabled 源，或按 `source_types` 指定范围。

## 2. 三个数据源各自怎么检索

### 2.1 部署文档（document）

调用 `rag_service.retrieving()`（`src/rag/service.py`）。

默认配置（见 `src/api/schemas.py`）：

- `retrieval_method=hybrid`
- `top_k=5`
- `similarity_threshold=0.5`

`hybrid` 细节：

- 同时做向量检索 + BM25。
- 用 `rrf_fuse_results` 融合两路文档候选。
- 融合后再按 `similarity_threshold` 过滤。

### 2.2 Excel 表格（table）

调用 `rag_service.table_chat_retrieving()`。

默认配置：

- `retrieval_method=hybrid`
- `top_k=20`
- `similarity_threshold=0.15`

打分逻辑：

- `keyword`：按字段匹配（包含、反向包含、token overlap）得到关键词分。
- `vector`：对行文本做向量余弦相似度。
- `hybrid`：`final_score = keyword_score * 0.45 + vector_score * 0.55`。

按 `final_score` 降序，截断 `top_k`。

### 2.3 Adela 数据（adela）

调用 `rag_service.adela_chat_retrieving()`。

默认配置：

- `retrieval_method=hybrid`
- `top_k=20`
- `similarity_threshold=0.15`

策略：

- 先确保 Adela JSONL 存在（必要时由 `source_dir` 自动导出）。
- 核心检索复用 `table_chat_retrieving()`，即和 table 使用同一套 keyword/vector/hybrid 打分公式。
- 检索后补充 `model_info`、`benchmark_info` 等字段，形成最终 adela 行结果。

## 3. unified 如何做跨源合并

三路检索结果先统一成 `UnifiedEvidenceItem`，然后调用 `_rrf_fuse_evidences()` 融合（`src/api/routes.py`）。

RRF 公式：

- 对每条证据，按其“来源内名次”累计分数：
- `score += 1 / (rrf_k + rank)`

默认参数：

- `rrf_k=60`
- `fused_top_k=12`

排序与输出：

- 按融合分 `score` 降序排序。
- 取前 `fused_top_k` 条作为 `fused_evidences`。

## 4. 关键点（当前实现）

- 三源检索是并行的（`asyncio.gather`）。
- 每个源内部先有自己的打分体系；跨源时不直接比较原始分，统一靠 RRF 名次融合。
- 证据 `evidence_id` 带源前缀（`document::` / `table::` / `adela::`），因此跨源不会相互覆盖。
- `source_status.used_count` 表示该源最终进入融合结果的条数。
