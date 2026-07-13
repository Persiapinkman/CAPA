---
name: rag-retrieve-answer
description: Sends a single query to the configured RAG unified_query API using service-side default parameters; prints the answer on stdout. Optional full JSON via --out.
---

# RAG Retrieve Answer

## Overview

- **Input**: one natural language `--query`.
- **Behavior**: `POST` to the RAG `chat_engine/unified_query` endpoint. Only sends required `query`; all other parameters use the API's default settings.
- **Output**: the **`answer`** string printed to **stdout**; if the service returns `success: false` or no answer, exits with code 1 and writes details to stderr.

## Run

```bash
python3 skills/rag-retrieve-answer/scripts/run_rag.py \
  --query "safety_rope v0.2.1 模型检测的是什么目标？"
```

Equivalent `curl` (same JSON body as `run_rag.py`):

```bash
curl -sS -X POST http://10.111.32.254:6060/api/v1/rag/chat_engine/unified_query \
  -H "Content-Type: application/json" \
  -d '{"query":"safety_rope v0.2.1 追加了什么数据，标签有哪些？"}'
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
  --base-url "http://host:6060/api/v1/rag/chat_engine/unified_query"
```

(`--url` is still accepted as a deprecated override of `--base-url`.)

## Environment (defaults)

| Variable | Default |
|----------|---------|
| `RAG_QUERY_URL` | Used when `--base-url` is omitted; falls back to `http://10.111.32.254:6060/api/v1/rag/chat_engine/unified_query` |

## Request shape

The script sends JSON shaped like:

```json
{
  "query": "<your query>"
}
```

Expected response fields include:
- `fused_evidences`（融合后的统一证据）
- `answer`（LLM 生成答案）
- `timings`（`route_ms` / `retrieve_ms` / `fuse_ms` / `answer_ms` / `total_ms`）
- `success`

## Troubleshooting

- **`Connection refused` / `[Errno 111]`** — No server is accepting TCP on that host and port (RAG process not running, wrong port, or bound only on another interface). Start the RAG API service, or set `RAG_QUERY_URL` / pass `--base-url` with the correct `http://host:port/.../query`.

## References

- [references/request_example.json](references/request_example.json) — example HTTP POST body（仅包含 `query`；其余参数由接口默认值决定）.
- [references/response_example.json](references/response_example.json) — example success response（与 `--out` 写入文件的典型结构一致；字段以实际服务为准）.
