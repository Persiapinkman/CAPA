# API 调用说明

最后更新：2026-04-27

本文档提供核心 API 的最小调用示例与参数说明。每个接口的参数表由脚本从 `openapi.json` 自动同步。

## 基础信息

- 默认服务地址：`http://127.0.0.1:6060`
- API 前缀：`/api/v1`
- 完整基础路径：`http://127.0.0.1:6060/api/v1`

服务启动：

```bash
python -m src.main
```

探活：

```bash
curl http://127.0.0.1:6060/openapi.json
curl http://127.0.0.1:6060/docs
```

## 接口总览

| 接口 | 方法 | 路径 | 数据源/用途 |
| --- | --- | --- | --- |
| 文档分块与向量化 | `POST` | `/rag/doc_engine/chunking_embedding` | 原始文本/PDF blocks 转 chunk + embedding |
| document 文档问答 | `POST` | `/rag/chat_engine/query` | 模型发版文档正文（ONES 工作文档 / 发版 PDF）一次检索 + 回答 |
| 表格问答 | `POST` | `/rag/chat_engine/table_query` | tables 行级检索 + 回答 |
| adela 问答 | `POST` | `/rag/chat_engine/adela_query` | adela 部署记录检索 + 回答 |
| unified 检索直出 | `POST` | `/rag/chat_engine/unified_retrieve` | 跨 documents / tables / adela 检索融合，直接返回证据 |
| unified 问答 | `POST` | `/rag/chat_engine/unified_query` | 跨 documents / tables / adela 路由、检索、融合、回答 |

说明：接口枚举里的 `document` 是历史兼容保留名，当前固定表示“模型发版文档正文内容”，来源于 ONES 工作文档及对应发版 PDF，并不是泛指任意 document。
| 文本向量化 | `POST` | `/rag/embedding` | 直接生成 embedding |

## 1) 文档分块与向量化

```bash
curl -X POST http://127.0.0.1:6060/api/v1/rag/doc_engine/chunking_embedding \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "这是一个用于验证接口可用性的测试文档。",
    "input_type": "raw",
    "doc_id": "_test_doc_raw_001",
    "doc_name": "data_source_smoke_test.txt"
  }'
```

<!-- AUTO_TABLES: /api/v1/rag/doc_engine/chunking_embedding START -->
**输入参数**

| 参数 | 类型 | 必填 | 默认值 | 说明/限制 |
| --- | --- | --- | --- | --- |
| `text` | `string` | 是 | - | 待进行处理的原始文本内容 |
| `input_type` | `string` | 否 | `autopdf` | 输入文本的格式类型，当前支持 autopdf、raw、json_list、markdown、mineru、pdf_blocks |
| `doc_id` | `string` | 是 | - | 唯一标识文档 |
| `doc_name` | `string` | 是 | - | 原始文档的文件名 |
| `embedding_models` | `string[]` | 否 | `["bge_m3", "EvoQwen2.5-VL-Retriever-3B-v1"]` | 用于生成文本向量的模型列表,可同时使用多个模型 |

**输出参数**

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `doc_id` | `string` | 唯一标识文档 |
| `doc_name` | `string` | 原始文档的文件名 |
| `index_nodes` | `object[]` | 文档分块和向量化后的结果列表 |
| `index_nodes[].id` | `string` | 节点（小文本块）唯一标识符 |
| `index_nodes[].text` | `string` | 小文本块内容 |
| `index_nodes[].index_id` | `string` | 对应的大文本块的唯一标识符 |
| `index_nodes[].index_text` | `string` | 对应的大文本块的完整内容 |
| `index_nodes[].metadata` | `object` | 包含文档标题、页码等元数据信息 |
| `index_nodes[].doc_id` | `string` | 文档的唯一标识符 |
| `index_nodes[].doc_name` | `string` | 文档的原始文件名 |
| `index_nodes[].embeddings` | `object[]` | 文本块在不同模型下的向量表示列表 |
| `index_nodes[].embeddings[].model` | `string` | 用于生成向量的模型名称,如bge-m3 |
| `index_nodes[].embeddings[].embedding` | `number[]` | 文本块的向量表示 |
| `success` | `boolean` | 文档处理是否成功完成 |
| `message` | `string \| null` | 处理失败时的错误信息说明 |
<!-- AUTO_TABLES: /api/v1/rag/doc_engine/chunking_embedding END -->

## 2) document 文档问答

这里的 `document` 指模型发版文档正文内容，通常用于回答当前版本输入输出、默认阈值、优化点、追加数据、标签、背景说明等问题。

```bash
curl -X POST http://127.0.0.1:6060/api/v1/rag/chat_engine/query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "safety_rope v0.2.1 追加了什么数据，标签有哪些？"
  }'
```

<!-- AUTO_TABLES: /api/v1/rag/chat_engine/query START -->
**输入参数**

| 参数 | 类型 | 必填 | 默认值 | 说明/限制 |
| --- | --- | --- | --- | --- |
| `query` | `string` | 是 | - | 用户问题 |
| `retrieval_method` | `string` | 否 | `hybrid` | 检索方法，`vector` 为向量检索，`bm25` 为关键词检索，`hybrid` 为向量+BM25 混合检索（默认，RRF 融合）；枚举: `vector` / `bm25` / `hybrid` |
| `top_k` | `integer \| null` | 否 | `5` | 返回的最相关文档块数量 |
| `similarity_threshold` | `number \| null` | 否 | `0.5` | 相似度过滤阈值 |
| `filter` | `string \| null` | 否 | - | 检索过滤条件 |
| `uri` | `string` | 否 | `.../documents/milvus_data_source_evoqwen_3b.db` | 向量数据库路径，默认使用 data_source 离线入库生成的本地库 |
| `collection_name` | `string` | 否 | `llamacollection` | 向量数据库 collection 名称 |
| `embedding_models` | `string[]` | 否 | `["bge_m3", "EvoQwen2.5-VL-Retriever-3B-v1"]` | 用于检索的向量模型列表 |
| `vector_store_configs` | `object \| null` | 否 | - | 默认按 embedding 模型路由到各自的 data_source 向量库 |
| `llm_config` | `object \| null` | 否 | - | 可选的大模型配置，不传则使用服务默认配置 |

**输出参数**

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `query` | `string` | 用户问题 |
| `retrieved_chunks` | `object[]` | 检索返回的相关文档块 |
| `retrieved_chunks[].id` | `string` | 节点（小文本块）唯一标识符 |
| `retrieved_chunks[].text` | `string` | 小文本块内容 |
| `retrieved_chunks[].index_id` | `string` | 对应的大文本块的唯一标识符 |
| `retrieved_chunks[].index_text` | `string` | 对应的大文本块的完整内容 |
| `retrieved_chunks[].metadata` | `object` | 包含文档标题、页码等元数据信息 |
| `retrieved_chunks[].doc_id` | `string` | 文档的唯一标识符 |
| `retrieved_chunks[].doc_name` | `string` | 文档的原始文件名 |
| `retrieved_chunks[].score` | `number` | 文本块与查询的相关性得分 |
| `reference` | `object[]` | 检索命中的 PDF 来源链接列表，按文档去重 |
| `reference[].doc_name` | `string` | 去重后的来源名称（document 通常为模型发版文档文件名；adela 通常为模型名 + did） |
| `reference[].url` | `string \| null` | 来源对应的跳转链接，未配置时为 null |
| `answer` | `string` | 基于检索结果生成的回答 |
| `timings` | `object` | - |
| `timings.retrieve_ms` | `number` | 检索阶段耗时，单位毫秒 |
| `timings.answer_ms` | `number` | 大模型回答生成阶段耗时，单位毫秒 |
| `timings.reference_ms` | `number` | reference 去重与映射阶段耗时，单位毫秒 |
| `timings.total_ms` | `number` | 模型发版文档正文问答总耗时，单位毫秒 |
| `success` | `boolean` | 问答流程是否成功 |
| `message` | `string \| null` | 失败时的错误信息 |
<!-- AUTO_TABLES: /api/v1/rag/chat_engine/query END -->

## 3) 表格问答

```bash
curl -X POST http://127.0.0.1:6060/api/v1/rag/chat_engine/table_query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "安全绳有哪些模型？"
  }'
```

<!-- AUTO_TABLES: /api/v1/rag/chat_engine/table_query START -->
**输入参数**

| 参数 | 类型 | 必填 | 默认值 | 说明/限制 |
| --- | --- | --- | --- | --- |
| `query` | `string` | 是 | - | 用户问题 |
| `retrieval_method` | `string` | 否 | `hybrid` | 表格检索方式，支持关键词、向量和混合检索；枚举: `keyword` / `vector` / `hybrid` |
| `top_k` | `integer \| null` | 否 | `20` | 返回的相关行数量上限 |
| `similarity_threshold` | `number \| null` | 否 | `0.15` | 行级相似度过滤阈值 |
| `data_path` | `string` | 否 | `.../tables/model_release_records.jsonl` | 表格 JSONL 数据路径 |
| `searchable_fields` | `string[] \| null` | 否 | - | 参与检索的字段列表；不传则使用服务默认配置 |
| `return_fields` | `string[] \| null` | 否 | - | 返回给前端和 LLM 的字段列表；不传则使用服务默认配置 |
| `embedding_models` | `string[]` | 否 | `["bge_m3"]` | 表格向量检索使用的 embedding 模型列表；vector / hybrid 下仅支持一个模型 |
| `llm_config` | `object \| null` | 否 | - | 可选的大模型配置，不传则使用服务默认配置 |

**输出参数**

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `query` | `string` | 用户问题 |
| `matched_rows` | `object[]` | 命中的结构化表格行 |
| `matched_rows[].row_id` | `string` | 唯一的行 ID |
| `matched_rows[].score` | `number` | 行级相关性分数 |
| `matched_rows[].matched_fields` | `string[]` | 命中的字段列表 |
| `matched_rows[].entity` | `object` | 命中的结构化行内容 |
| `answer` | `string` | 基于命中行生成的回答 |
| `timings` | `object` | - |
| `timings.retrieve_ms` | `number` | 检索阶段耗时，单位毫秒 |
| `timings.answer_ms` | `number` | 回答生成阶段耗时，单位毫秒 |
| `timings.total_ms` | `number` | 总耗时，单位毫秒 |
| `success` | `boolean` | 问答流程是否成功 |
| `message` | `string \| null` | 失败时的错误信息 |
<!-- AUTO_TABLES: /api/v1/rag/chat_engine/table_query END -->

## 4) adela 部署记录问答

```bash
curl -X POST http://127.0.0.1:6060/api/v1/rag/chat_engine/adela_query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "有哪些 cuda11.0-trt7.1-fp16-T4 的部署模型？"
  }'
```

说明：`reference[].url` 由 `ADELA_DEPLOYMENT_URL_TEMPLATE` 基于命中 `did` 进行拼接（将 `{did}` 替换为实际值）。

<!-- AUTO_TABLES: /api/v1/rag/chat_engine/adela_query START -->
**输入参数**

| 参数 | 类型 | 必填 | 默认值 | 说明/限制 |
| --- | --- | --- | --- | --- |
| `query` | `string` | 是 | - | 用户问题 |
| `retrieval_method` | `string` | 否 | `hybrid` | adela 检索方式，支持关键词、向量和混合检索；枚举: `keyword` / `vector` / `hybrid` |
| `top_k` | `integer \| null` | 否 | `20` | 返回的相关记录数量上限 |
| `similarity_threshold` | `number \| null` | 否 | `0.15` | 记录级相似度过滤阈值 |
| `data_path` | `string` | 否 | `.../adela/adela_release_records.jsonl` | adela 规范化 JSONL 数据路径 |
| `source_dir` | `string \| null` | 否 | `.../adela/data` | 可选的 adela 原始 JSON 目录；当 data_path 不存在时用于自动导出 |
| `searchable_fields` | `string[] \| null` | 否 | - | 参与检索的字段列表；不传则使用 adela 默认配置 |
| `return_fields` | `string[] \| null` | 否 | - | 返回给前端和 LLM 的字段列表；不传则使用 adela 默认配置 |
| `embedding_models` | `string[]` | 否 | `["bge_m3"]` | adela 向量检索使用的 embedding 模型列表；vector / hybrid 下仅支持一个模型 |
| `llm_config` | `object \| null` | 否 | - | 可选的大模型配置，不传则使用服务默认配置 |

**输出参数**

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `query` | `string` | 用户问题 |
| `matched_records` | `object[]` | 命中的 adela 结构化记录 |
| `matched_records[].row_id` | `string` | 唯一的行 ID |
| `matched_records[].score` | `number` | 行级相关性分数 |
| `matched_records[].matched_fields` | `string[]` | 命中的字段列表 |
| `matched_records[].entity` | `object` | 命中的结构化行内容 |
| `reference` | `object[]` | 命中的 adela 部署链接列表（基于 did 去重） |
| `reference[].doc_name` | `string` | 去重后的来源名称（document 通常为模型发版文档文件名；adela 通常为模型名 + did） |
| `reference[].url` | `string \| null` | 来源对应的跳转链接，未配置时为 null |
| `answer` | `string` | 基于命中记录生成的回答 |
| `timings` | `object` | - |
| `timings.retrieve_ms` | `number` | 检索阶段耗时，单位毫秒 |
| `timings.answer_ms` | `number` | 回答生成阶段耗时，单位毫秒 |
| `timings.total_ms` | `number` | 总耗时，单位毫秒 |
| `success` | `boolean` | 问答流程是否成功 |
| `message` | `string \| null` | 失败时的错误信息 |
<!-- AUTO_TABLES: /api/v1/rag/chat_engine/adela_query END -->

## 5) unified 统一检索直出（不走 LLM answer）

```bash
curl -X POST http://127.0.0.1:6060/api/v1/rag/chat_engine/unified_retrieve \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "安全绳使用到的相关模型有哪些记录？"
  }'
```

<!-- AUTO_TABLES: /api/v1/rag/chat_engine/unified_retrieve START -->
**输入参数**

| 参数 | 类型 | 必填 | 默认值 | 说明/限制 |
| --- | --- | --- | --- | --- |
| `query` | `string` | 是 | - | 用户问题 |
| `source_types` | `string[] \| null` | 否 | - | 限制检索的数据源类型范围；不传时使用全部启用来源；枚举: `document` / `table` / `adela`。其中 `document` 表示模型发版文档正文 |
| `fused_top_k` | `integer \| null` | 否 | `12` | 融合后返回证据条数上限 |
| `rrf_k` | `integer \| null` | 否 | `60` | RRF 融合参数 k |
| `document_config` | `object` | 否 | - | - |
| `document_config.enabled` | `boolean` | 否 | `true` | 是否启用模型发版文档正文检索 |
| `document_config.retrieval_method` | `string` | 否 | `hybrid` | 模型发版文档正文检索方法；枚举: `vector` / `bm25` / `hybrid` |
| `document_config.top_k` | `integer \| null` | 否 | `5` | 模型发版文档正文返回块数上限 |
| `document_config.similarity_threshold` | `number \| null` | 否 | `0.5` | 模型发版文档正文相似度阈值 |
| `document_config.filter` | `string \| null` | 否 | - | 模型发版文档正文检索过滤条件 |
| `document_config.uri` | `string` | 否 | `.../documents/milvus_data_source_evoqwen_3b.db` | 模型发版文档正文向量库路径 |
| `document_config.collection_name` | `string` | 否 | `llamacollection` | 模型发版文档正文向量库 collection |
| `document_config.embedding_models` | `string[]` | 否 | `["bge_m3", "EvoQwen2.5-VL-Retriever-3B-v1"]` | 模型发版文档正文检索使用的 embedding 模型列表 |
| `document_config.vector_store_configs` | `object \| null` | 否 | - | 模型发版文档正文可选模型到向量库路由配置 |
| `table_config` | `object` | 否 | - | - |
| `table_config.enabled` | `boolean` | 否 | `true` | 是否启用 tables 检索 |
| `table_config.retrieval_method` | `string` | 否 | `hybrid` | tables 检索方式；枚举: `keyword` / `vector` / `hybrid` |
| `table_config.top_k` | `integer \| null` | 否 | `20` | tables 返回行数上限 |
| `table_config.similarity_threshold` | `number \| null` | 否 | `0.15` | tables 相似度阈值 |
| `table_config.data_path` | `string` | 否 | `.../tables/model_release_records.jsonl` | tables JSONL 路径 |
| `table_config.searchable_fields` | `string[] \| null` | 否 | - | tables 参与检索字段列表 |
| `table_config.return_fields` | `string[] \| null` | 否 | - | tables 返回字段列表 |
| `table_config.embedding_models` | `string[]` | 否 | `["bge_m3"]` | tables 向量检索模型列表 |
| `adela_config` | `object` | 否 | - | - |
| `adela_config.enabled` | `boolean` | 否 | `true` | 是否启用 adela 检索 |
| `adela_config.retrieval_method` | `string` | 否 | `hybrid` | adela 检索方式；枚举: `keyword` / `vector` / `hybrid` |
| `adela_config.top_k` | `integer \| null` | 否 | `20` | adela 返回记录数上限 |
| `adela_config.similarity_threshold` | `number \| null` | 否 | `0.15` | adela 相似度阈值 |
| `adela_config.data_path` | `string` | 否 | `.../adela/adela_release_records.jsonl` | adela JSONL 路径 |
| `adela_config.source_dir` | `string \| null` | 否 | `.../adela/data` | adela 原始 JSON 目录（当 data_path 不存在时用于自动导出） |
| `adela_config.searchable_fields` | `string[] \| null` | 否 | - | adela 参与检索字段列表 |
| `adela_config.return_fields` | `string[] \| null` | 否 | - | adela 返回字段列表 |
| `adela_config.embedding_models` | `string[]` | 否 | `["bge_m3"]` | adela 向量检索模型列表 |

**输出参数**

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `query` | `string` | 用户问题 |
| `selected_sources` | `string[]` | 本次请求实际执行检索的数据源；枚举: `document` / `table` / `adela` |
| `fused_evidences` | `object[]` | 融合后的统一证据列表 |
| `fused_evidences[].evidence_id` | `string` | 统一证据ID |
| `fused_evidences[].source_type` | `string` | 证据来源类型；枚举: `document` / `table` / `adela`；其中 `document` 表示模型发版文档正文 |
| `fused_evidences[].score` | `number` | 融合后的相关性分数 |
| `fused_evidences[].source_rank` | `integer` | 该来源内的排序名次（从1开始） |
| `fused_evidences[].source_score` | `number \| null` | 该来源原始分数 |
| `fused_evidences[].title` | `string` | 证据标题 |
| `fused_evidences[].snippet` | `string` | 证据摘要内容 |
| `fused_evidences[].payload` | `object` | 证据原始结构化数据 |
| `reference` | `object[]` | 融合证据对应的来源链接列表（包含模型发版文档来源链接与 adela 的 did 链接） |
| `reference[].doc_name` | `string` | 去重后的来源名称（document 通常为模型发版文档文件名；adela 通常为模型名 + did） |
| `reference[].url` | `string \| null` | 来源对应的跳转链接，未配置时为 null |
| `source_status` | `object[]` | 各来源检索状态与统计 |
| `source_status[].source_type` | `string` | 来源类型；枚举: `document` / `table` / `adela`；其中 `document` 表示模型发版文档正文 |
| `source_status[].enabled` | `boolean` | 是否启用该来源 |
| `source_status[].success` | `boolean` | 该来源检索是否成功 |
| `source_status[].retrieve_ms` | `number` | 该来源检索耗时，单位毫秒 |
| `source_status[].retrieved_count` | `integer` | 该来源原始命中数量 |
| `source_status[].used_count` | `integer` | 该来源参与融合的数量 |
| `source_status[].message` | `string \| null` | 该来源失败原因 |
| `timings` | `object` | - |
| `timings.retrieve_ms` | `number` | 三路检索总耗时（并行墙钟），单位毫秒 |
| `timings.fuse_ms` | `number` | 融合耗时，单位毫秒 |
| `timings.total_ms` | `number` | 总耗时，单位毫秒 |
| `success` | `boolean` | 检索流程是否成功 |
| `message` | `string \| null` | 失败时的错误信息 |
<!-- AUTO_TABLES: /api/v1/rag/chat_engine/unified_retrieve END -->

## 6) unified 统一检索问答

```bash
curl -X POST http://127.0.0.1:6060/api/v1/rag/chat_engine/unified_query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "安全绳使用到的相关模型有哪些记录？",
    "stream": false
  }'
```

如需流式（SSE）返回，可将 `stream` 设为 `true` 并使用 `-N`：

```bash
curl -N -X POST http://127.0.0.1:6060/api/v1/rag/chat_engine/unified_query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "安全绳使用到的相关模型有哪些记录？",
    "stream": true
  }'
```

<!-- AUTO_TABLES: /api/v1/rag/chat_engine/unified_query START -->
**输入参数**

| 参数 | 类型 | 必填 | 默认值 | 说明/限制 |
| --- | --- | --- | --- | --- |
| `query` | `string` | 是 | - | 用户问题 |
| `fused_top_k` | `integer \| null` | 否 | `12` | 融合后返回证据条数上限 |
| `rrf_k` | `integer \| null` | 否 | `60` | RRF 融合参数 k |
| `stream` | `boolean` | 否 | `false` | 是否开启流式回答（SSE） |
| `route_with_llm` | `boolean` | 否 | `true` | 是否在检索前先让 LLM 判断需要调用哪些数据源 |
| `document_config` | `object` | 否 | - | - |
| `document_config.enabled` | `boolean` | 否 | `true` | 是否启用模型发版文档正文检索 |
| `document_config.retrieval_method` | `string` | 否 | `hybrid` | 模型发版文档正文检索方法；枚举: `vector` / `bm25` / `hybrid` |
| `document_config.top_k` | `integer \| null` | 否 | `5` | 模型发版文档正文返回块数上限 |
| `document_config.similarity_threshold` | `number \| null` | 否 | `0.5` | 模型发版文档正文相似度阈值 |
| `document_config.filter` | `string \| null` | 否 | - | 模型发版文档正文检索过滤条件 |
| `document_config.uri` | `string` | 否 | `.../documents/milvus_data_source_evoqwen_3b.db` | 模型发版文档正文向量库路径 |
| `document_config.collection_name` | `string` | 否 | `llamacollection` | 模型发版文档正文向量库 collection |
| `document_config.embedding_models` | `string[]` | 否 | `["bge_m3", "EvoQwen2.5-VL-Retriever-3B-v1"]` | 模型发版文档正文检索使用的 embedding 模型列表 |
| `document_config.vector_store_configs` | `object \| null` | 否 | - | 模型发版文档正文可选模型到向量库路由配置 |
| `table_config` | `object` | 否 | - | - |
| `table_config.enabled` | `boolean` | 否 | `true` | 是否启用 tables 检索 |
| `table_config.retrieval_method` | `string` | 否 | `hybrid` | tables 检索方式；枚举: `keyword` / `vector` / `hybrid` |
| `table_config.top_k` | `integer \| null` | 否 | `20` | tables 返回行数上限 |
| `table_config.similarity_threshold` | `number \| null` | 否 | `0.15` | tables 相似度阈值 |
| `table_config.data_path` | `string` | 否 | `.../tables/model_release_records.jsonl` | tables JSONL 路径 |
| `table_config.searchable_fields` | `string[] \| null` | 否 | - | tables 参与检索字段列表 |
| `table_config.return_fields` | `string[] \| null` | 否 | - | tables 返回字段列表 |
| `table_config.embedding_models` | `string[]` | 否 | `["bge_m3"]` | tables 向量检索模型列表 |
| `adela_config` | `object` | 否 | - | - |
| `adela_config.enabled` | `boolean` | 否 | `true` | 是否启用 adela 检索 |
| `adela_config.retrieval_method` | `string` | 否 | `hybrid` | adela 检索方式；枚举: `keyword` / `vector` / `hybrid` |
| `adela_config.top_k` | `integer \| null` | 否 | `20` | adela 返回记录数上限 |
| `adela_config.similarity_threshold` | `number \| null` | 否 | `0.15` | adela 相似度阈值 |
| `adela_config.data_path` | `string` | 否 | `.../adela/adela_release_records.jsonl` | adela JSONL 路径 |
| `adela_config.source_dir` | `string \| null` | 否 | `.../adela/data` | adela 原始 JSON 目录（当 data_path 不存在时用于自动导出） |
| `adela_config.searchable_fields` | `string[] \| null` | 否 | - | adela 参与检索字段列表 |
| `adela_config.return_fields` | `string[] \| null` | 否 | - | adela 返回字段列表 |
| `adela_config.embedding_models` | `string[]` | 否 | `["bge_m3"]` | adela 向量检索模型列表 |
| `llm_config` | `object \| null` | 否 | - | 可选的大模型配置，不传则使用服务默认配置 |

**输出参数**

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `query` | `string` | 用户问题 |
| `fused_evidences` | `object[]` | 融合后的统一证据列表 |
| `fused_evidences[].evidence_id` | `string` | 统一证据ID |
| `fused_evidences[].source_type` | `string` | 证据来源类型；枚举: `document` / `table` / `adela`；其中 `document` 表示模型发版文档正文 |
| `fused_evidences[].score` | `number` | 融合后的相关性分数 |
| `fused_evidences[].source_rank` | `integer` | 该来源内的排序名次（从1开始） |
| `fused_evidences[].source_score` | `number \| null` | 该来源原始分数 |
| `fused_evidences[].title` | `string` | 证据标题 |
| `fused_evidences[].snippet` | `string` | 证据摘要内容 |
| `fused_evidences[].payload` | `object` | 证据原始结构化数据 |
| `reference` | `object[]` | 融合证据对应的来源链接列表（包含模型发版文档来源链接与 adela 的 did 链接） |
| `reference[].doc_name` | `string` | 去重后的来源名称（document 通常为模型发版文档文件名；adela 通常为模型名 + did） |
| `reference[].url` | `string \| null` | 来源对应的跳转链接，未配置时为 null |
| `source_status` | `object[]` | 各来源检索状态与统计 |
| `source_status[].source_type` | `string` | 来源类型；枚举: `document` / `table` / `adela`；其中 `document` 表示模型发版文档正文 |
| `source_status[].enabled` | `boolean` | 是否启用该来源 |
| `source_status[].success` | `boolean` | 该来源检索是否成功 |
| `source_status[].retrieve_ms` | `number` | 该来源检索耗时，单位毫秒 |
| `source_status[].retrieved_count` | `integer` | 该来源原始命中数量 |
| `source_status[].used_count` | `integer` | 该来源参与融合的数量 |
| `source_status[].message` | `string \| null` | 该来源失败原因 |
| `route_plan` | `object` | - |
| `route_plan.route_with_llm` | `boolean` | 是否启用 LLM 路由 |
| `route_plan.selected_sources` | `string[]` | LLM 决策后实际执行检索的数据源；枚举: `document` / `table` / `adela`；其中 `document` 表示模型发版文档正文 |
| `route_plan.skipped_sources` | `string[]` | 被 LLM 跳过或配置禁用的数据源；枚举: `document` / `table` / `adela`；其中 `document` 表示模型发版文档正文 |
| `route_plan.fallback_used` | `boolean` | LLM 路由失败后是否回退到默认策略 |
| `route_plan.reason` | `string \| null` | LLM 路由理由或回退说明 |
| `answer` | `string` | 基于融合证据生成的最终回答 |
| `timings` | `object` | - |
| `timings.route_ms` | `number` | LLM 路由耗时，单位毫秒 |
| `timings.retrieve_ms` | `number` | 三路检索总耗时（并行墙钟），单位毫秒 |
| `timings.fuse_ms` | `number` | 融合耗时，单位毫秒 |
| `timings.answer_ms` | `number` | 回答生成耗时，单位毫秒 |
| `timings.total_ms` | `number` | 总耗时，单位毫秒 |
| `success` | `boolean` | 问答流程是否成功 |
| `message` | `string \| null` | 失败时的错误信息 |
<!-- AUTO_TABLES: /api/v1/rag/chat_engine/unified_query END -->

## 7) 文本向量化

```bash
curl -X POST http://127.0.0.1:6060/api/v1/rag/embedding \
  -H 'Content-Type: application/json' \
  -d '{
    "input": ["安全绳检测模型", "Shikra Embedding"]
  }'
```

<!-- AUTO_TABLES: /api/v1/rag/embedding START -->
**输入参数**

| 参数 | 类型 | 必填 | 默认值 | 说明/限制 |
| --- | --- | --- | --- | --- |
| `input` | `string \| string[]` | 是 | - | 需要向量化的文本内容，可以是单个字符串或字符串列表 |
| `model` | `string` | 否 | `EvoQwen2.5-VL-Retriever-3B-v1` | 用于生成向量的模型名称 |

**输出参数**

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `object` | `string` | 对象类型 |
| `data` | `object[]` | 向量列表 |
| `data[].index` | `integer` | 向量索引 |
| `data[].embedding` | `number[]` | 文本的向量表示 |
| `data[].object` | `string` | 对象类型 |
| `model` | `string` | 使用的模型名称 |
| `usage` | `object` | - |
| `usage.prompt_tokens` | `integer` | 输入文本的token数量 |
| `usage.completion_tokens` | `integer` | 完成的token数量 |
| `usage.total_tokens` | `integer` | 总token数量 |
<!-- AUTO_TABLES: /api/v1/rag/embedding END -->

## `llm_config` 覆盖说明

问答类接口可选传入 `llm_config`：

```json
{
  "llm_config": {
    "model": "Qwen3.5-4B",
    "base_url": "http://10.111.32.253:8000/v1",
    "api_key": "your-api-key",
    "max_tokens": 2048,
    "temperature": 0.7,
    "top_p": 0.9
  }
}
```

## 自动同步参数表

执行以下命令，可基于 `openapi.json` 自动刷新本文档中所有参数表：

```bash
python scripts/sync_api_usage_tables.py --openapi-url http://127.0.0.1:6060/openapi.json --doc docs/API_USAGE.md
```

也支持本地文件：

```bash
python scripts/sync_api_usage_tables.py --openapi-file /path/to/openapi.json --doc docs/API_USAGE.md
```
