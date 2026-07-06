---
name: rag-retrieve-answer
description: Sends a single query to the configured Playbook-enhanced RAG query API; prints the answer on stdout. Optional full JSON via --out.
---

# RAG Retrieve Answer

## Overview

- **Input**: one natural language `--query`.
- **Behavior**: `POST` to the Playbook-enhanced RAG `/api/v1/playbook/query` endpoint. Sends `query`, `stream=false`, and Playbook defaults.
- **Output**: the **`answer`** string printed to **stdout**; if the service returns `success: false` or no answer, exits with code 1 and writes details to stderr.

## Run

```bash
python3 skills/rag-retrieve-answer/scripts/run_rag.py \
  --query "safety_rope v0.2.1 模型检测的是什么目标？"
```

Equivalent `curl` (same JSON body as `run_rag.py`):

```bash
curl -sS -X POST http://127.0.0.1:6062/api/v1/playbook/query \
  -H "Content-Type: application/json" \
  -d '{"query":"safety_rope v0.2.1 追加了什么数据，标签有哪些？","stream":false,"use_playbook":true,"playbook_top_k":8,"top_k":12}'
```

Optional full response JSON:

```bash
python3 skills/rag-retrieve-answer/scripts/run_rag.py \
  --query "你的问题" \
  --out /tmp/rag_response.json
```

Override API URL for one run:

```bash
python3 skills/rag-retrieve-answer/scripts/run_rag.py \
  --query "问题" \
  --base-url "http://host:6062/api/v1/playbook/query"
```

(`--url` is still accepted as a deprecated override of `--base-url`.)

## Environment (defaults)

| Variable | Default |
|----------|---------|
| `RAG_QUERY_URL` | Used when `--base-url` is omitted; falls back to `http://127.0.0.1:6062/api/v1/playbook/query` |

## Request shape

The script sends JSON shaped like:

```json
{
  "query": "<your query>",
  "stream": false,
  "use_playbook": true,
  "playbook_top_k": 8,
  "top_k": 12
}
```

Expected response fields include:
- `evidences`（融合后的统一证据）
- `answer`（LLM 生成答案）
- `run_id`（可用于 `/playbook/feedback` 提交反馈）
- `timings`

## Troubleshooting

- **`Connection refused` / `[Errno 111]`** — No server is accepting TCP on that host and port (RAG process not running, wrong port, or bound only on another interface). Start the ace-rag API service, or set `RAG_QUERY_URL` / pass `--base-url` with the correct `http://host:6062/api/v1/playbook/query`.

## References

- [references/request_example.json](references/request_example.json) — example HTTP POST body（仅包含 `query`；其余参数由接口默认值决定）.
- [references/response_example.json](references/response_example.json) — example success response（与 `--out` 写入文件的典型结构一致；字段以实际服务为准）.
