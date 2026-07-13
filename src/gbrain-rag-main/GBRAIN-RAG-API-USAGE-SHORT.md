# GBrain RAG API 简版接入说明

面向下游服务，只保留最常用的请求参数和最重要的返回字段。详细参数仍看 `GBRAIN-RAG-API-USAGE.md`。

## 基础信息

- 默认地址：`http://127.0.0.1:6061`
- API 前缀：`/api/v1/rag`
- 推荐检索接口：`POST /chat_engine/unified_retrieve`
- 推荐问答接口：`POST /chat_engine/unified_query`

## 最小请求

检索直出，不生成答案：

```bash
curl --noproxy '*' -X POST http://127.0.0.1:6061/api/v1/rag/chat_engine/unified_retrieve \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "安全绳检测 v0.2.1 的输出是什么？",
    "top_k": 8
  }'
```

问答接口，返回答案和证据：

```bash
curl --noproxy '*' -X POST http://127.0.0.1:6061/api/v1/rag/chat_engine/unified_query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "安全绳检测 v0.2.1 的输出是什么？",
    "top_k": 8
  }'
```

常用输入参数：

| 参数 | 是否必填 | 说明 |
| --- | --- | --- |
| `query` | 是 | 用户问题 |
| `top_k` | 否 | 返回证据条数，默认 `8` |
| `sources` | 否 | 限定来源：`["document"]` / `["table"]` / `["adela"]`；不传则默认检索全部启用来源 |
| `include_full_documents` | 否 | 是否返回命中文档的完整拼接内容，默认 `false` |

## 下游最关心的返回字段

检索接口和问答接口都会返回证据列表：

```text
evidences[]
```

问答接口额外返回：

```text
answer
knowledge_base_fully_answered
```

核心字段：

| 字段 | 说明 |
| --- | --- |
| `answer` | 生成后的最终答案，仅 `unified_query` 有 |
| `evidences[].snippet` | 命中 chunk 的正文摘要，适合列表展示 |
| `evidences[].payload.index_text` | 命中 chunk 的检索正文，通常比 `snippet` 更完整，适合给下游 LLM |
| `evidences[].payload.field_summary` | 表格/结构化字段摘要；如果存在，字段类问题优先用它 |
| `evidences[].evidence_id` | 证据 ID，基本等同 chunk ID |
| `evidences[].payload.chunk_id` | 原始 chunk ID |
| `evidences[].source_type` | 来源类型：`document` / `table` / `adela` |
| `evidences[].block_type` | chunk 类型：`text` / `table` / `row` / `aggregate` |
| `evidences[].doc_name` | 文档名或结构化来源名 |
| `evidences[].page_label` | 页码或页标签 |
| `evidences[].source_path` | 原始文件路径 |
| `full_documents[].content` | `include_full_documents=true` 时返回，命中文档的完整索引内容拼接 |

## 检索到的正文在哪里

按优先级取：

```python
for ev in resp["evidences"]:
    chunk_id = ev["payload"].get("chunk_id") or ev["evidence_id"]
    text_for_llm = ev["payload"].get("field_summary") or ev["payload"].get("index_text") or ev["snippet"]
    preview_text = ev["snippet"]
```

注意：

- `snippet` 是摘要，可能截断。
- `payload.index_text` 是检索正文，也可能截断，但通常比 `snippet` 更完整。
- 如果需要完整文档上下文，请在请求里设置 `include_full_documents=true`，然后读取 `full_documents[].content`。

## 不同来源怎么定位原文

| 来源 | 判断方式 | 定位字段 |
| --- | --- | --- |
| ONES/PDF 正文 | `source_type=document` | `source_path + page_label + block_type + title` |
| ONES/PDF 表格 | `source_type=document` 且 `block_type=table` | `source_path + page_label + title` |
| 模型发版汇总表 | `source_type=table` | `payload.row_id`，以及 `metadata.sheet_name/source_row_number` |
| Adela 部署记录 | `source_type=adela` | `payload.did` / `payload.rid` / `reference_url` |
| 聚合统计 | `block_type=aggregate` | 不是原始 chunk，需要按聚合条件回查结构化行 |

一句话：普通下游问答用 `payload.index_text`；表格字段优先用 `payload.field_summary`；原文溯源用 `source_path/page_label/title`；完整文档展开用 `include_full_documents=true`。
