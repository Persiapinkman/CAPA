# table 表格问答 v1 简介与耗时

## v1 简介

当前的 table 表格问答 v1 是“行级检索 + 大模型生成”链路，接口流程如下：

1. 接收用户问题。
2. 按 `TABLE_SEARCHABLE_FIELDS` 构造查询文本并执行行级检索。
3. 根据 `retrieval_method`（`keyword` / `vector` / `hybrid`）计算相关性分数。
4. 过滤、排序后得到 `matched_rows`。
5. 将命中行格式化为上下文，调用 LLM 生成最终答案。
6. 返回回答、命中行和分步耗时 `timings`。

当前接口已经支持：

- 在响应中返回 `retrieve_ms`、`answer_ms`、`total_ms`
- 在调用脚本中记录端到端耗时，便于后续聚合分析

## 当前默认使用方法

在不额外传参（或仅传 `query`）时，table 问答默认按下面方式运行：

- 默认接口：`POST /api/v1/rag/chat_engine/table_query`
- 默认检索方法：`hybrid`
- 默认 `top_k`：`20`
- 默认 `similarity_threshold`：`0.15`
- 默认向量模型：`["bge_m3"]`
- 默认数据文件：`data_source/tables/model_release_records.jsonl`
- 默认检索字段：`TABLE_SEARCHABLE_FIELDS`
- 默认返回字段：`TABLE_RETURN_FIELDS`

也就是“关键词规则打分 + 向量相似度”融合检索，再交给 LLM 生成最终答案。

## 检索方法与过程（按代码实现）

以下流程对应 `src/rag/service.py` 的 `table_chat_retrieving()`。

### 1. 请求接入与耗时统计边界

- 接口入口：`POST /api/v1/rag/chat_engine/table_query`。
- `retrieve_ms` 统计范围：`rag_service.table_chat_retrieving(request)` 整段执行时间。
- `answer_ms` 统计范围：`answer_table_question()` 调用 LLM 生成答案时间。
- `total_ms`：从请求进入路由到响应返回。

### 2. 读取数据与字段集合

1. 读取 `data_path`（默认 `data_source/tables/model_release_records.jsonl`）全部 JSONL 行到内存。
2. 选取可检索字段：
   - 请求传 `searchable_fields` 用请求值；
   - 否则用 `TABLE_SEARCHABLE_FIELDS`。
3. 选取返回字段：
   - 请求传 `return_fields` 用请求值；
   - 否则用 `TABLE_RETURN_FIELDS`。
4. 对 query 做分词，得到 `query_tokens`（tokenizer 来自当前 embedding 模型）。

### 3. 关键词打分（`keyword` 分支核心）

对每一行、每个检索字段计算字段分数 `field_score`，字段分数取以下三项最大值：

1. 完整包含：若 `query_lower in field_lower`，得分 `1.0`。
2. 反向包含：若 `field_lower in query_lower` 且字段长度 `>=2`，得分 `0.85`。
3. token 重叠：`overlap = |query_tokens ∩ field_tokens| / |query_tokens|`。

行级关键词分数不是简单 max，而是：

- `avg_score = 所有命中字段分数均值`
- `max_score = 所有命中字段分数最大值`
- `keyword_score = min(1.0, max_score * 0.7 + avg_score * 0.3)`

同时记录 `matched_fields`（该行中分数 `>0` 的字段名列表）。

### 4. 向量打分（`vector` / `hybrid` 分支）

1. 仅支持一个 embedding 模型（`embedding_models` 长度必须为 1）。
2. 每行先拼接 `search_text`（格式：`field: value` 按行拼接）。
3. 先尝试从 `EmbeddingArtifactStore` 加载 `.npy + .meta.json` 缓存（目录：`data_source/embedding_artifacts/tables/`）。
4. 若缓存不可用，则批量生成行向量并落盘缓存。
5. 对 query 生成向量，与每行向量做余弦相似度：
   - `vector_score = dot(row_emb, query_emb) / (||row_emb|| * ||query_emb|| + 1e-8)`

### 5. 融合、过滤、排序、截断

每行最终分数 `final_score` 规则：

- `keyword`：`final_score = keyword_score`
- `vector`：`final_score = vector_score`
- `hybrid`：`final_score = keyword_score * 0.45 + vector_score * 0.55`

后处理顺序：

1. 阈值过滤：`final_score < similarity_threshold` 的行丢弃。
2. 构造返回对象：`row_id`、`score`、`matched_fields`、`entity`（只保留 return_fields）。
3. 按 `score` 降序排序。
4. 截取 `top_k`（未传则返回全部命中行）。

补充：代码默认 `similarity_threshold` 为空时按 `0.0` 处理。

### 6. 回答阶段

- 将 `matched_rows` 统一格式化后填入 `TABLE_QA_PROMPT`。
- 调用 OpenAI 兼容接口生成答案（`enable_thinking=False`）。
- table 链路没有 documents 的 `reference` 映射步骤。

## 当前耗时情况

以下数据来自 2026-04-21 的实测：

```bash
python sample_code/rag_chat_benchmark.py \
  --chat-api-url http://127.0.0.1:6060/api/v1/rag/chat_engine/table_query \
  --query "安全绳有哪些模型？" \
  --repeat 10 \
  --similarity-threshold 0.15 \
  --embedding-models bge_m3 \
  --output-jsonl results/table_chat_benchmark_20260421_repeat10.jsonl
```

- 样本量：10（1 个问题 x 重复 10 次）
- 成功率：100%（10/10）

### 端到端耗时（客户端视角，ms）

| 指标 | avg | min | p50 | p95 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 客户端总耗时 | 2109.408 | 1759.811 | 2057.640 | 2526.314 | 2636.156 |
| 服务端总耗时 | 1840.100 | 1502.727 | 1742.093 | 2280.872 | 2369.232 |
| 客户端额外开销 | 269.308 | 219.187 | 269.390 | 320.639 | 335.939 |

### 服务端分步耗时（ms）

| 阶段 | avg | min | p50 | p95 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 检索耗时 | 179.234 | 169.536 | 177.440 | 191.384 | 192.209 |
| 回答耗时 | 1660.836 | 1329.933 | 1560.693 | 2104.134 | 2186.622 |

- `retrieve_ms` 和 `answer_ms` 可直接从响应 `timings` 获取。
- 该压测脚本原本为 document 问答设计，`retrieved_count/reference_count` 字段对 table 语义不完全对应，建议重点看 `timings` 与 `answer_length`。
