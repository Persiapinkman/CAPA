# Swift RAG API 简版接入说明

面向下游服务，只保留最常用的请求参数和最重要的返回字段。详细参数仍看 `docs/API_USAGE.md`。

## 基础信息

- 默认地址：`http://127.0.0.1:6060`
- API 前缀：`/api/v1`
- 推荐检索接口：`POST /rag/chat_engine/unified_retrieve`
- 推荐问答接口：`POST /rag/chat_engine/unified_query`

## 最小请求

统一检索直出，不生成答案：

```bash
curl -X POST http://127.0.0.1:6060/api/v1/rag/chat_engine/unified_retrieve \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "安全绳使用到的相关模型有哪些记录？",
    "fused_top_k": 12
  }'
```

统一问答，返回答案和证据：

```bash
curl -X POST http://127.0.0.1:6060/api/v1/rag/chat_engine/unified_query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "安全绳使用到的相关模型有哪些记录？",
    "fused_top_k": 12,
    "stream": false
  }'
```

常用输入参数：

| 参数 | 是否必填 | 说明 |
| --- | --- | --- |
| `query` | 是 | 用户问题 |
| `fused_top_k` | 否 | 融合后返回证据条数，默认 `12` |
| `source_types` | 否 | 仅 `unified_retrieve` 使用，限定来源：`document` / `table` / `adela` |
| `stream` | 否 | 仅 `unified_query` 使用，是否 SSE 流式返回 |
| `route_with_llm` | 否 | 仅 `unified_query` 使用，是否让 LLM 判断数据源 |

## 下游最关心的返回字段

统一接口的证据列表在：

```text
fused_evidences[]
```

问答接口额外返回：

```text
answer
```

核心字段：

| 字段 | 说明 |
| --- | --- |
| `answer` | 生成后的最终答案，仅 `unified_query` 有 |
| `fused_evidences[].snippet` | 命中证据摘要，适合列表展示 |
| `fused_evidences[].payload.index_text` | document 来源的检索正文，适合给下游 LLM |
| `fused_evidences[].payload.entity` | table/adela 来源的结构化行内容 |
| `fused_evidences[].evidence_id` | 证据 ID |
| `fused_evidences[].source_type` | 来源类型：`document` / `table` / `adela` |
| `fused_evidences[].title` | 证据标题 |
| `fused_evidences[].payload.doc_name` | 文档名，document 来源常用 |
| `fused_evidences[].payload.page_label` | 页码或页标签，document 来源常用 |
| `fused_evidences[].payload.source_path` | 原始文件路径或结构化记录路径 |
| `reference[]` | 去重后的来源链接列表 |

## 检索到的内容在哪里

统一接口按来源取内容：

```python
for ev in resp["fused_evidences"]:
    payload = ev.get("payload") or {}

    if ev["source_type"] == "document":
        text_for_llm = payload.get("index_text") or ev.get("snippet")
    else:
        text_for_llm = payload.get("entity") or ev.get("snippet")
```

单数据源接口字段不同：

| 接口 | 命中内容字段 |
| --- | --- |
| `/rag/chat_engine/query` | `retrieved_chunks[].index_text`，摘要可用 `retrieved_chunks[].text` |
| `/rag/chat_engine/table_query` | `matched_rows[].entity` |
| `/rag/chat_engine/adela_query` | `matched_records[].entity` |
| `/rag/chat_engine/unified_retrieve` | `fused_evidences[].payload.index_text` 或 `payload.entity` |
| `/rag/chat_engine/unified_query` | `answer` + `fused_evidences[]` |

## 不同来源怎么定位原文

| 来源 | 判断方式 | 定位字段 |
| --- | --- | --- |
| ONES/PDF 正文 | `source_type=document` | `payload.source_path + payload.page_label + title` |
| 模型发版汇总表 | `source_type=table` | `payload.entity.row_id` 或结构化字段，如 `model_name/oid` |
| Adela 部署记录 | `source_type=adela` | `payload.entity.did` / `payload.entity.rid` / `reference[].url` |

一句话：`swift-rag` 统一接口看 `fused_evidences`；正文内容在 `payload.index_text`，结构化行内容在 `payload.entity`，最终答案在 `answer`。
