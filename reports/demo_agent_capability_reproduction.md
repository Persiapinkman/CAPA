# Demo Agent Capability Reproduction

> 这是能力复现脚本生成时的历史快照，不应作为当前服务在线状态的唯一依据。2026-07-14 的实时复验已确认 RAG、Rex-Omni 和 Qwen 均通过 Demo 端到端调用；当前权威结果见 `reports/DEMO_CAPABILITY_LIVE_CHECK_2026-07-14.md`。

## 结论

静态能力契约已完整复现。
外部服务状态与代码契约分开记录；有脚本不代表对应服务当前在线。

## 能力矩阵

| Tool | Owner | Components | Services | Image | Side effect | Historical observations | Live status |
|---|---|---|---|---:|---:|---:|---|
| `rag_answer` | executor | rag-retrieve-answer | rag | false | false | 436 | partial_or_offline |
| `re_question` | executor | query-rewrite | model_gateway | false | false | 237 | online |
| `answerer` | orchestrator | answerer | model_gateway | false | false | 0 | online |
| `flux-image-generation` | executor | user-intent-understanding, llm-prompts-generation, flux-image-generation | model_gateway, flux_api | false | true | 5 | unverified |
| `qwen_detection` | executor | qwen-vlm-open-set-delection | qwen_detection | true | false | 4 | partial_or_offline |
| `rexomni_detection` | executor | rexomni-open-set-detection | model_gateway | true | false | 0 | online |
| `pipeline_eval` | executor | target-detection-evaluation, user-intent-understanding, llm-prompts-generation, flux-image-generation, qwen-vlm-open-set-delection, rexomni-open-set-detection, eval-reports-generation | model_gateway, flux_api, qwen_detection | true | true | 2 | unverified |
| `migration_advisor` | executor | migration-advisor, rag-retrieve-answer, rexomni-open-set-detection | rag, model_gateway | false | false | 1 | partial_or_offline |
| `adela_cli_eval` | executor | adela-cli, rag-retrieve-answer | rag, adela_cli | false | true | 23 | partial_or_offline |

## 历史证据

- Session JSON：发现 272，成功解析 272，错误 0。
- Threads：446；queries：295；轨迹 steps：1083。
- 带图片 session：7。
- Session cohort：`{"browser_session": 61, "codex_eval": 51, "migration_dataset": 51, "smoke": 1, "uuid_session": 108}`。
- 未知历史动作：`{}`。
- 报告仅包含聚合计数，不包含用户 query、回答或客户端地址。
- LLM debug：2178 条；Planner response 1068 条；原始 JSON 解析失败 31 条。
- LLM debug cohort：`{"grpo_eval": 1455, "runtime": 707, "synthetic_smoke": 16}`；不输出 prompt、response 或 session ID。

## 服务探活

| Service | Status | Detail |
|---|---|---|
| `model_gateway` | online | http://10.111.32.253:8000/v1/models |
| `qwen_detection` | offline | ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=9012): Max retries exceeded with url: /v1/models (Caused by NewConnectionError("HTTPConnection(host='127.0.0.1', port=901 |
| `rag_playbook` | offline | ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=6062): Max retries exceeded with url: /api/v1/playbook/health (Caused by NewConnectionError("HTTPConnection(host='127.0.0 |
| `rag_unified` | offline | ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=6061): Max retries exceeded with url: /api/v1/rag/health (Caused by NewConnectionError("HTTPConnection(host='127.0.0.1',  |
| `adela_cli` | offline |  |
| `flux_api` | not_probed |  |

## Model Smoke

- Planner：`passed`。
- Answerer：`passed`。

## Demo HTTP Smoke

- 结果：`passed`；静态健康 `True`；工具数 9。
- NDJSON done `True`；最终回答字符数 81；合成会话已清理 `True`。

## Recorded Smokes

- RAG：`contract_passed_content_missing`；HTTP contract `True`；content ready `False`。
- Rex-Omni：`passed`；boxes 1；bbox valid `True`。

## 使用边界

默认复现不会调用 Flux、Adela 部署或其它有成本/副作用的动作。
这些能力必须在隔离测试资产上显式执行端到端验收后，才能标记为 live reproduced。
