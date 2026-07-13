# ACE RAG API 简版接入说明

面向下游服务，只保留最常用的请求参数和最重要的返回字段。详细参数仍看 `ACE-RAG-API-USAGE.md`。

## 基础信息

- 默认地址：`http://127.0.0.1:6062`
- API 前缀：`/api/v1/playbook`
- 推荐检索接口：`POST /retrieve`
- 推荐问答接口：`POST /query`
- 依赖事实检索服务：`gbrain-rag`，默认 `http://127.0.0.1:6061/api/v1/rag`

ACE RAG 是 Playbook 增强层：它负责查询扩展、来源提示、回答编排和反馈记录；事实证据仍来自 `gbrain-rag`。

## 最小请求

Playbook 增强检索，不生成答案：

```bash
curl --noproxy '*' -X POST http://127.0.0.1:6062/api/v1/playbook/retrieve \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "安全绳检测在 T4 上有部署吗？did 是多少？",
    "top_k": 12,
    "use_playbook": true
  }'
```

Playbook 增强问答，返回答案和证据：

```bash
curl --noproxy '*' -X POST http://127.0.0.1:6062/api/v1/playbook/query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "安全绳检测在 T4 上有部署吗？did 是多少？",
    "top_k": 12,
    "use_playbook": true
  }'
```

常用输入参数：

| 参数 | 是否必填 | 说明 |
| --- | --- | --- |
| `query` | 是 | 用户问题 |
| `top_k` | 否 | 返回事实证据条数，默认 `12` |
| `sources` | 否 | 限定事实来源：`["document"]` / `["table"]` / `["adela"]` |
| `use_playbook` | 否 | 是否启用 Playbook，默认 `true` |
| `playbook_top_k` | 否 | 命中 Playbook 条数，默认 `8` |
| `playbook_only` | 否 | 仅 `/retrieve` 使用；为 `true` 时只返回 Playbook 命中，不调用 `gbrain-rag`，`evidences=[]` |
| `stream` | 否 | `/query` 当前必须为 `false` 或不传 |

## 下游最关心的返回字段

检索接口和问答接口都会返回事实证据：

```text
evidences[]
```

问答接口额外返回：

```text
answer
run_id
```

核心字段：

| 字段 | 说明 |
| --- | --- |
| `answer` | 生成后的最终答案，仅 `/query` 有 |
| `run_id` | 本次问答运行 ID，可用于提交反馈 |
| `evidences[].snippet` | 命中 chunk 的正文摘要，适合列表展示 |
| `evidences[].payload.index_text` | 命中 chunk 的检索正文，通常比 `snippet` 更完整 |
| `evidences[].payload.field_summary` | 表格/结构化字段摘要；如果存在，字段类问题优先用它 |
| `evidences[].evidence_id` | 证据 ID |
| `evidences[].payload.chunk_id` | 原始 chunk ID |
| `evidences[].source_type` | 事实来源：`document` / `table` / `adela` |
| `evidences[].block_type` | chunk 类型：`text` / `table` / `row` / `aggregate` |
| `evidences[].doc_name` | 文档名或结构化来源名 |
| `evidences[].page_label` | 页码或页标签 |
| `evidences[].source_path` | 原始文件路径 |
| `playbook.items[]` | 命中的 Playbook 规则，用于解释为什么扩展查询或调整来源 |
| `playbook.query_expansion_terms` | Playbook 给出的查询扩展词 |
| `playbook.source_hints` | Playbook 给出的建议事实来源 |
| `v2_request` | ACE 实际发给 `gbrain-rag` 的请求体，便于排查 |

只需要检索 Playbook 规则时，调用 `/retrieve` 并传 `"playbook_only": true`。此时 `v2_request` 只展示本来会发给 v2 的请求，并包含 `_skipped=true`。

## 检索到的正文在哪里

ACE 透传 `gbrain-rag` 的 evidence 结构，取法相同：

```python
for ev in resp["evidences"]:
    chunk_id = ev["payload"].get("chunk_id") or ev["evidence_id"]
    text_for_llm = ev["payload"].get("field_summary") or ev["payload"].get("index_text") or ev["snippet"]
    preview_text = ev["snippet"]
```

注意：

- `playbook.items[]` 不是事实原文，只是检索/回答策略。
- 真正的事实内容在 `evidences[]`。
- 字段类问题优先看 `payload.field_summary`；普通上下文优先看 `payload.index_text`；展示摘要用 `snippet`。

## 不同来源怎么定位原文

| 来源 | 判断方式 | 定位字段 |
| --- | --- | --- |
| ONES/PDF 正文 | `source_type=document` | `source_path + page_label + block_type + title` |
| ONES/PDF 表格 | `source_type=document` 且 `block_type=table` | `source_path + page_label + title` |
| 模型发版汇总表 | `source_type=table` | `payload.row_id`，以及 `metadata.sheet_name/source_row_number` |
| Adela 部署记录 | `source_type=adela` | `payload.did` / `payload.rid` / `reference_url` |

一句话：ACE 下游看 `answer` 和 `evidences`；事实正文仍在 `evidences[].payload.index_text/field_summary`，Playbook 字段只用于解释和调试。
