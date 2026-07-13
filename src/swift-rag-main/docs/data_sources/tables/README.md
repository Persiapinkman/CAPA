# 模型发版表数据源

最后更新：2026-04-21

本文档说明结构化表格数据源的导出、检索、缓存和问答方式。当前表格问答对应接口为 `POST /api/v1/rag/chat_engine/table_query`。

## 当前状态

- 原始 Excel：`data_source/模型发版记录汇总.xlsx`
- 标准化 JSONL：`data_source/tables/model_release_records.jsonl`
- 当前记录数：`293`
- 默认检索方法：`hybrid`
- 默认向量模型：`bge_m3`
- 行级 embedding 缓存目录：`data_source/embedding_artifacts/tables/`

表格问答不走文档 chunk 和 Milvus，而是直接对 JSONL 的“行”做检索。

## 标准化导出

导出脚本：`scripts/export_model_release_table_to_jsonl.py`

```bash
python scripts/export_model_release_table_to_jsonl.py
```

自定义输入输出：

```bash
python scripts/export_model_release_table_to_jsonl.py \
  --input data_source/模型发版记录汇总.xlsx \
  --output data_source/tables/model_release_records.jsonl
```

导出时会做这些处理：

- 将中文列名映射为稳定英文 key
- 标准化日期字段 `last_updated` 和 `last_updated_month`
- 生成行级唯一标识 `row_id`
- 保留 `source_file`、`sheet_name`、`source_row_number`
- 基于可检索字段生成 `search_text`

## 主要字段

默认检索字段来自 `src/core/config.py` 的 `TABLE_SEARCHABLE_FIELDS`：

```text
target_name, algorithm_type, algorithm_name, application_scene,
owner, model_name, supported_device, recommended_config,
last_updated, last_updated_month
```

默认返回字段来自 `TABLE_RETURN_FIELDS`：

```text
target_name, algorithm_type, algorithm_name, application_scene,
owner, model_name, supported_device, recommended_config,
last_updated, ones_release_link, oid
```

## 检索方式

表格检索入口：`RAGService.table_chat_retrieving()`。

| 方法 | 说明 |
| --- | --- |
| `keyword` | 基于字段包含关系、查询分词与字段分词重叠做轻量规则打分 |
| `vector` | 对行级拼接文本做 embedding，并用余弦相似度排序 |
| `hybrid` | `keyword_score * 0.45 + vector_score * 0.55` |

过滤与排序流程：

1. 对每一行计算相关性分数。
2. 按 `similarity_threshold` 过滤。
3. 按分数降序排序。
4. 截取 `top_k`。
5. 将命中行交给 LLM 生成最终答案。

## embedding 缓存

表格行向量不会写入 Milvus，而是由 `EmbeddingArtifactStore` 写入：

```text
data_source/embedding_artifacts/tables/*.npy
data_source/embedding_artifacts/tables/*.meta.json
```

缓存文件名会包含：

- 源数据文件 stem
- embedding 模型名
- searchable_fields hash
- 源文件路径 hash

缓存有效性校验会检查：

- `data_path`
- `model_name`
- `searchable_fields`
- `row_ids`
- 源文件大小与 mtime

任一项变化都会重新生成 embedding。

## 调用示例

```bash
python sample_code/table_chat_api_client.py
```

或直接 curl：

```bash
curl -X POST http://127.0.0.1:6060/api/v1/rag/chat_engine/table_query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "安全绳有哪些模型？",
    "retrieval_method": "hybrid",
    "top_k": 20,
    "similarity_threshold": 0.15,
    "embedding_models": ["bge_m3"]
  }'
```

响应重点字段：

- `matched_rows`：命中的结构化行
- `matched_rows[].entity`：返回给前端和 LLM 的字段
- `matched_rows[].matched_fields`：关键词命中的字段
- `answer`：LLM 总结后的答案
- `timings`：检索、回答和总耗时

## 适用与限制

适合：模型列表、负责人、OID、支持设备、推荐配置、最近更新时间等结构化问题。

限制：

- 当前在线检索只读取 JSONL，不直接查询 Excel。
- `vector` / `hybrid` 模式下只支持一个 embedding 模型。
- 关键词检索是轻量规则，不是 BM25。
- 当前没有单元格级精排或复杂表结构理解。

## 相关文档

- `TABLE_RETRIEVAL_METHOD.md`
- `RAG_CHAT_V1_TIMING.md`
- `../../API_USAGE.md`
