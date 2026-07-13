# GBrain RAG API 调用说明

最后更新：2026-05-19

本文档提供 `gbrain-rag` 核心 API 的最小调用示例与参数说明。`gbrain-rag` 是 v2 RAG 服务，负责 document / table / adela 三类数据源的路由、检索、融合、结构化统计和回答生成。

## 基础信息

- 默认服务地址：`http://127.0.0.1:6061`
- API 前缀：`/api/v1`
- RAG 路由前缀：`/rag`
- 完整基础路径：`http://127.0.0.1:6061/api/v1/rag`

服务启动：

```bash
cd gbrain-rag
PYTHONPATH=src python -m gbrain_rag.main
```

也可以用 uvicorn 启动：

```bash
cd gbrain-rag
PYTHONPATH=src python -m uvicorn gbrain_rag.main:app --host 0.0.0.0 --port 6061
```

使用 tmux 后台启动：

```bash
mkdir -p logs
tmux -L gbrain-rag new -d -s gbrain-rag-api 'PYTHONPATH=src python -m uvicorn gbrain_rag.main:app --host 0.0.0.0 --port 6061 2>&1 | tee -a logs/gbrain-rag-api.log'
```

常用 tmux 操作：

```bash
# 查看后台会话
tmux -L gbrain-rag ls

# 进入服务会话，查看实时输出
tmux -L gbrain-rag attach -t gbrain-rag-api

# 在 tmux 会话内退出但不停止服务：按 Ctrl-b，然后按 d

# 查看日志
tail -f logs/gbrain-rag-api.log

# 停止后台服务
tmux -L gbrain-rag kill-session -t gbrain-rag-api
```

探活：

```bash
curl http://127.0.0.1:6061/api/v1/rag/health
curl http://127.0.0.1:6061/openapi.json
curl http://127.0.0.1:6061/docs
```

## 接口总览

| 接口 | 方法 | 路径 | 数据源/用途 |
| --- | --- | --- | --- |
| 健康检查 | `GET` | `/rag/health` | 查看索引 chunk 数、source 计数、embedding 计数和健康状态 |
| unified 检索直出 | `POST` | `/rag/chat_engine/unified_retrieve` | 跨 document / table / adela 路由、检索、融合，直接返回证据 |
| unified 问答 | `POST` | `/rag/chat_engine/unified_query` | 跨 document / table / adela 路由、检索、融合、回答 |
| document 文档问答 | `POST` | `/rag/chat_engine/query` | 固定只查模型发版文档正文 |
| table 表格问答 | `POST` | `/rag/chat_engine/table_query` | 固定只查模型发版信息汇总表 |
| adela 部署记录问答 | `POST` | `/rag/chat_engine/adela_query` | 固定只查 Adela 部署记录 |
| 原生检索直出 | `POST` | `/rag/retrieve` | 与 unified 检索使用同一套请求/响应模型 |
| 原生问答 | `POST` | `/rag/query` | 与 unified 问答使用同一套请求/响应模型 |
| 文本向量化 | `POST` | `/rag/embedding` | 直接生成 embedding |

说明：

- `document` 表示模型发版文档正文，通常来自 ONES 工作文档 / 发版 PDF。
- `table` 表示模型发版信息汇总表，适合负责人、OID、更新时间、推荐配置、支持设备、数量统计等结构化字段。
- `adela` 表示 Adela 部署信息，适合 did/rid、部署平台、部署状态、部署版本等问题。
- `/rag/chat_engine/unified_retrieve` 是 `/rag/retrieve` 的兼容别名；`/rag/chat_engine/unified_query` 是 `/rag/query` 的兼容别名。

## 1) unified 检索直出

```bash
curl --noproxy '*' -X POST http://127.0.0.1:6061/api/v1/rag/chat_engine/unified_retrieve \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "安全绳检测 v0.2.1 的输出是什么？",
    "top_k": 8,
    "retrieval_method": "hybrid",
    "include_full_documents": false
  }'
```

手动指定数据源：

```bash
curl --noproxy '*' -X POST http://127.0.0.1:6061/api/v1/rag/chat_engine/unified_retrieve \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "安全绳检测在 T4 上有哪些部署，did 是多少？",
    "sources": ["adela"],
    "top_k": 8
  }'
```

**输入参数**

| 参数 | 类型 | 必填 | 默认值 | 说明/限制 |
| --- | --- | --- | --- | --- |
| `query` | `string` | 是 | - | 用户问题 |
| `retrieval_method` | `string` | 否 | `hybrid` | 检索方法；枚举：`vector` / `keyword` / `bm25` / `hybrid` |
| `top_k` | `integer` | 否 | `8` | 融合后返回证据数量上限，范围 1-100 |
| `candidate_limit` | `integer` | 否 | `80` | 每路候选召回上限，范围 1-1000 |
| `similarity_threshold` | `number \| null` | 否 | - | 相似度过滤阈值；不传则使用服务默认策略 |
| `sources` | `string[] \| null` | 否 | - | 手动指定数据源；可选 `document` / `table` / `adela`；不传时自动路由 |
| `route_with_llm` | `boolean` | 否 | `false` | 预留字段；当前主要使用确定性路由 |
| `expand_query_with_llm` | `boolean` | 否 | `true` | 是否使用 LLM 生成查询扩展词 |
| `query_expansion_terms` | `string[]` | 否 | `[]` | 手动提供查询扩展词；提供后优先使用请求中的扩展词 |
| `include_full_documents` | `boolean` | 否 | `false` | 是否返回命中 chunk 所属文档的完整索引内容；如果多个命中 chunk 对应同一个 `doc_id`，先按 `doc_id` 去重后只返回一份 |
| `embedding_model` | `string \| null` | 否 | `bge_m3` | 单 embedding 模型名 |
| `embedding_models` | `string[] \| null` | 否 | - | 多 embedding 模型列表；document 默认使用配置中的多模型，table/adela 默认使用 `bge_m3` |
| `embedding_backend` | `string \| null` | 否 | - | embedding 后端；常见值为 `sentence-transformers` / `hashing` |
| `document` | `object` | 否 | 见下表 | document 数据源独立配置 |
| `table` | `object` | 否 | 见下表 | table 数据源独立配置 |
| `adela` | `object` | 否 | 见下表 | adela 数据源独立配置 |

**数据源独立配置**

| 参数 | 类型 | 必填 | 默认值 | 说明/限制 |
| --- | --- | --- | --- | --- |
| `enabled` | `boolean` | 否 | `true` | 是否启用该数据源 |
| `top_k` | `integer \| null` | 否 | - | 该数据源独立返回上限；不传则使用请求级 `top_k` |
| `retrieval_method` | `string \| null` | 否 | - | 该数据源独立检索方法；不传则使用请求级 `retrieval_method` |
| `similarity_threshold` | `number \| null` | 否 | - | 该数据源独立相似度阈值；不传则使用请求级 `similarity_threshold` |

**输出参数**

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `query` | `string` | 用户问题 |
| `route_plan` | `object` | 路由计划 |
| `route_plan.document` | `boolean` | 是否检索 document |
| `route_plan.table` | `boolean` | 是否检索 table |
| `route_plan.adela` | `boolean` | 是否检索 adela |
| `route_plan.reason` | `string` | 路由原因 |
| `route_plan.sources` | `string[]` | 实际检索的数据源列表 |
| `evidences` | `object[]` | 融合后的证据列表 |
| `evidences[].evidence_id` | `string` | 统一证据 ID |
| `evidences[].legacy_evidence_id` | `string \| null` | 兼容旧格式的证据 ID |
| `evidences[].source_type` | `string` | 证据来源；`document` / `table` / `adela` |
| `evidences[].score` | `number` | 融合后的相关性分数 |
| `evidences[].source_rank` | `integer` | 该来源内排序名次 |
| `evidences[].source_score` | `number` | 该来源原始分数 |
| `evidences[].title` | `string` | 证据标题 |
| `evidences[].snippet` | `string` | 证据摘要内容 |
| `evidences[].doc_id` | `string` | 文档或结构化来源 ID |
| `evidences[].doc_name` | `string` | 文档名或结构化来源名 |
| `evidences[].page_label` | `integer \| string \| null` | 页码或页标签 |
| `evidences[].block_type` | `string` | 证据块类型，例如 `text` / `table` / `structured_row` / `aggregate` |
| `evidences[].source_path` | `string \| null` | 原始文件路径 |
| `evidences[].metadata` | `object` | 元数据 |
| `evidences[].matched_entities` | `string[]` | 命中的实体 |
| `evidences[].retrieval_signals` | `object` | 检索阶段信号 |
| `evidences[].payload` | `object` | 原始结构化数据或 chunk 载荷 |
| `full_documents` | `object[]` | 当 `include_full_documents=true` 时返回的去重完整文档列表；默认关闭时为空数组 |
| `full_documents[].doc_id` | `string` | 文档 ID，与命中 evidence 的 `doc_id` 对应 |
| `full_documents[].doc_name` | `string` | 文档名 |
| `full_documents[].source_type` | `string` | 文档来源；`document` / `table` / `adela` |
| `full_documents[].source_path` | `string \| null` | 原始文件路径 |
| `full_documents[].content` | `string` | 该文档在索引中的完整内容，按页码/片段顺序拼接 |
| `full_documents[].chunk_count` | `integer` | 该文档包含的索引 chunk 数 |
| `full_documents[].metadata` | `object` | 附加信息；当前包含 `matched_evidence_ids`，表示哪些 evidence 命中了该文档 |
| `timings` | `object` | 检索耗时信息，包含 `retrieve_ms`、各 source 耗时和查询扩展耗时等 |
| `retrieved_count` | `integer` | 返回证据数量 |

返回完整文档示例：

```bash
curl --noproxy '*' -X POST http://127.0.0.1:6061/api/v1/rag/chat_engine/unified_retrieve \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "安全绳检测 v0.2.1 的输出是什么？",
    "sources": ["document"],
    "top_k": 5,
    "include_full_documents": true
  }'
```

响应中会继续返回 `evidences`，并额外返回去重后的 `full_documents`：

```json
{
  "evidences": [
    {
      "evidence_id": "chunk_1",
      "doc_id": "doc_a",
      "doc_name": "PDFs/safety_rope v0.2.1.pdf",
      "snippet": "..."
    },
    {
      "evidence_id": "chunk_2",
      "doc_id": "doc_a",
      "doc_name": "PDFs/safety_rope v0.2.1.pdf",
      "snippet": "..."
    }
  ],
  "full_documents": [
    {
      "doc_id": "doc_a",
      "doc_name": "PDFs/safety_rope v0.2.1.pdf",
      "source_type": "document",
      "source_path": "/path/to/safety_rope v0.2.1.pdf",
      "content": "[page=1 block_type=text title=...]\n完整文档内容...",
      "chunk_count": 12,
      "metadata": {
        "matched_evidence_ids": ["chunk_1", "chunk_2"]
      }
    }
  ]
}
```

## 2) unified 问答

```bash
curl --noproxy '*' -X POST http://127.0.0.1:6061/api/v1/rag/chat_engine/unified_query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "safety_rope v0.2.1 追加了什么数据，标签有哪些？",
    "top_k": 8,
    "retrieval_method": "hybrid"
  }'
```

流式问答使用 SSE，设置 `stream=true`：

```bash
curl --noproxy '*' -N http://127.0.0.1:6061/api/v1/rag/chat_engine/unified_query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "安全绳检测 v0.2.1 的输出是什么？",
    "top_k": 8,
    "retrieval_method": "hybrid",
    "stream": true
  }'
```

**输入参数**

除 unified 检索直出的全部参数外，还支持：

| 参数 | 类型 | 必填 | 默认值 | 说明/限制 |
| --- | --- | --- | --- | --- |
| `llm_config` | `object \| null` | 否 | - | 可选的大模型配置；不传则使用服务默认配置 |
| `stream` | `boolean` | 否 | `false` | 是否开启 SSE 流式回答 |

**`llm_config` 参数**

| 参数 | 类型 | 必填 | 默认值 | 说明/限制 |
| --- | --- | --- | --- | --- |
| `model` | `string \| null` | 否 | `Qwen3.5-4B` | 模型名称 |
| `base_url` | `string \| null` | 否 | 配置值 | OpenAI-compatible API 地址 |
| `api_key` | `string \| null` | 否 | 配置值 | API key |
| `max_tokens` | `integer \| null` | 否 | `2048` | 最大生成 token 数 |
| `temperature` | `number \| null` | 否 | `0.1` | 采样温度 |
| `top_p` | `number \| null` | 否 | `0.5` | nucleus sampling 参数 |
| `seed` | `integer \| null` | 否 | - | 采样随机种子，后端支持时生效 |

**输出参数**

包含 unified 检索直出的全部输出字段，并额外返回：

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `answer` | `string` | 基于证据生成的回答 |
| `knowledge_base_fully_answered` | `number` | LLM 给出的知识库证据充分回答置信度，范围 `0.0`-`1.0`；低分表示未命中、证据不足或仅返回降级片段 |
| `llm_config` | `object` | 本次回答使用的 LLM 配置 |

流式响应为 `text/event-stream`：

- 中间事件通常形如 `data: {"content":"..."}`。
- 最后一个 JSON 事件包含完整 `QueryResponse` 字段，包括 `knowledge_base_fully_answered`。
- 结束事件为 `data: [DONE]`。

## 3) 单数据源问答

这些接口和 unified 问答使用同一套请求/响应模型，但服务端会强制指定 `sources`。

### document 文档问答

```bash
curl --noproxy '*' -X POST http://127.0.0.1:6061/api/v1/rag/chat_engine/query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "安全绳检测 v0.2.1 的输入输出是什么？",
    "top_k": 5
  }'
```

适合回答模型发版正文里的输入输出、阈值、算法边界、功能介绍、优化点、追加数据、标签、精度指标等问题。

### table 表格问答

```bash
curl --noproxy '*' -X POST http://127.0.0.1:6061/api/v1/rag/chat_engine/table_query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "安全绳有哪些模型，OID 是多少？",
    "top_k": 8
  }'
```

适合回答模型清单、负责人、OID、更新时间、推荐配置、支持设备、数量统计等结构化字段。

### adela 部署记录问答

```bash
curl --noproxy '*' -X POST http://127.0.0.1:6061/api/v1/rag/chat_engine/adela_query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "安全绳检测在 T4 上有哪些部署，did/rid 是多少？",
    "top_k": 8
  }'
```

适合回答 did/rid、部署平台、部署状态、部署版本、部署记录细节等问题。

## 4) 文本向量化

```bash
curl --noproxy '*' -X POST http://127.0.0.1:6061/api/v1/rag/embedding \
  -H 'Content-Type: application/json' \
  -d '{
    "input": ["安全绳检测模型", "T4 部署 did"],
    "model": "bge_m3"
  }'
```

**输入参数**

| 参数 | 类型 | 必填 | 默认值 | 说明/限制 |
| --- | --- | --- | --- | --- |
| `input` | `string \| string[]` | 是 | - | 需要向量化的文本，可以是单个字符串或字符串列表 |
| `model` | `string \| null` | 否 | `bge_m3` | embedding 模型名 |
| `embedding_backend` | `string \| null` | 否 | - | embedding 后端；不传则使用服务默认后端 |

**输出参数**

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `object` | `string` | 固定为 `list` |
| `data` | `object[]` | 向量列表 |
| `data[].object` | `string` | 固定为 `embedding` |
| `data[].index` | `integer` | 输入文本索引 |
| `data[].embedding` | `number[]` | 文本向量 |
| `model` | `string` | 实际使用的模型名 |
| `usage` | `object` | token 使用统计；当前返回 `prompt_tokens` 和 `total_tokens` |

## 5) 健康检查

```bash
curl --noproxy '*' http://127.0.0.1:6061/api/v1/rag/health
```

**输出参数**

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `status` | `string` | `ok` 或 `degraded` |
| `chunks` | `integer` | 索引中的 chunk 总数 |
| `sources` | `object` | 各数据源 chunk 数 |
| `embeddings` | `object` | 各 embedding 模型向量数量 |
| `index_warnings` | `object` | 低于健康阈值的数据源及期望最小值 |
