# ACE RAG API 调用说明

最后更新：2026-05-13

本文档提供 `ace-rag` 核心 API 的最小调用示例与参数说明。`ace-rag` 是 v3 Playbook sidecar，负责 Playbook 检索、查询改写提示、来源提示、回答编排和用户反馈记录；事实检索仍由 `gbrain-rag` v2 服务完成。

推荐调用链：

```text
Client -> ace-rag /api/v1/playbook/query -> gbrain-rag /api/v1/rag/retrieve
```

## 基础信息

- 默认服务地址：`http://127.0.0.1:6062`
- API 前缀：`/api/v1`
- Playbook 路由前缀：`/playbook`
- 完整基础路径：`http://127.0.0.1:6062/api/v1/playbook`
- 默认依赖的 v2 服务：`http://127.0.0.1:6061/api/v1/rag`

先启动 v2 `gbrain-rag`：

```bash
cd /media/nvme1n1p1/linshihao/projects/gbrain-rag
PYTHONPATH=src python -m gbrain_rag.main
```

再启动 v3 `ace-rag`：

```bash
cd /media/nvme1n1p1/linshihao/projects/ace-rag
PYTHONPATH=src python -m ace_rag.main
```

探活：

```bash
curl http://127.0.0.1:6062/api/v1/playbook/health
curl http://127.0.0.1:6062/openapi.json
curl http://127.0.0.1:6062/docs
```

## 接口总览

| 接口 | 方法 | 路径 | 数据源/用途 |
| --- | --- | --- | --- |
| 健康检查 | `GET` | `/playbook/health` | 查看 ace-rag、v2 gbrain-rag 和 Playbook DB 状态 |
| Playbook 增强检索 | `POST` | `/playbook/retrieve` | 检索 Playbook，生成扩展词和来源提示，再调用 v2 `/retrieve` 返回证据 |
| Playbook 增强问答 | `POST` | `/playbook/query` | Playbook 增强检索 + ACE 回答编排 + run 记录 |
| 用户反馈 | `POST` | `/playbook/feedback` | 对一次 `run_id` 提交反馈，生成待处理 Playbook 操作 |

说明：

- `ace-rag` 不维护 document / table / adela 索引；这些事实数据来自 `gbrain-rag`。
- Playbook 只提供策略、业务约定、风险提示和检索/回答方法，不作为具体事实来源。
- `/playbook/query` 当前不支持流式输出；请求里设置 `stream=true` 会返回 `400`。

## 1) Playbook 增强检索

```bash
curl --noproxy '*' -X POST http://127.0.0.1:6062/api/v1/playbook/retrieve \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "安全绳检测在 T4 上有部署吗？did 是多少？",
    "top_k": 12,
    "use_playbook": true,
    "playbook_top_k": 8
  }'
```

手动指定数据源：

```bash
curl --noproxy '*' -X POST http://127.0.0.1:6062/api/v1/playbook/retrieve \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "安全绳检测 v0.2.1 的输入输出是什么？",
    "sources": ["document"],
    "top_k": 8
  }'
```

**输入参数**

| 参数 | 类型 | 必填 | 默认值 | 说明/限制 |
| --- | --- | --- | --- | --- |
| `query` | `string` | 是 | - | 用户问题 |
| `retrieval_method` | `string` | 否 | `hybrid` | 检索方法；枚举：`vector` / `keyword` / `bm25` / `hybrid` |
| `top_k` | `integer` | 否 | `12` | 融合后返回证据数量上限，范围 1-100 |
| `candidate_limit` | `integer` | 否 | `80` | 每路候选召回上限，范围 1-1000 |
| `similarity_threshold` | `number \| null` | 否 | - | 相似度过滤阈值；不传则使用 v2 默认策略 |
| `sources` | `string[] \| null` | 否 | - | 手动指定数据源；可选 `document` / `table` / `adela`；不传时可由 Playbook `source_hints` 和 v2 路由共同决定 |
| `route_with_llm` | `boolean` | 否 | `false` | 透传给 v2 的路由控制字段 |
| `expand_query_with_llm` | `boolean` | 否 | `true` | 未命中 Playbook 扩展词时，是否允许 v2 使用 LLM 扩展查询 |
| `query_expansion_terms` | `string[]` | 否 | `[]` | 手动提供查询扩展词；会与 Playbook 命中的扩展词合并 |
| `embedding_model` | `string \| null` | 否 | - | 透传给 v2 的单 embedding 模型名 |
| `embedding_models` | `string[] \| null` | 否 | - | 透传给 v2 的多 embedding 模型列表 |
| `embedding_backend` | `string \| null` | 否 | - | 透传给 v2 的 embedding 后端 |
| `document` | `object` | 否 | 见下表 | document 数据源独立配置 |
| `table` | `object` | 否 | 见下表 | table 数据源独立配置 |
| `adela` | `object` | 否 | 见下表 | adela 数据源独立配置 |
| `use_playbook` | `boolean` | 否 | `true` | 是否启用 Playbook 检索 |
| `playbook_top_k` | `integer` | 否 | `8` | Playbook 返回条数上限，范围 0-50 |

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
| `route_plan` | `object` | v2 路由计划 |
| `route_plan.document` | `boolean` | 是否检索 document |
| `route_plan.table` | `boolean` | 是否检索 table |
| `route_plan.adela` | `boolean` | 是否检索 adela |
| `route_plan.reason` | `string` | 路由原因 |
| `route_plan.sources` | `string[]` | 实际检索的数据源列表 |
| `evidences` | `object[]` | v2 返回的融合证据列表 |
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
| `evidences[].block_type` | `string` | 证据块类型 |
| `evidences[].source_path` | `string \| null` | 原始文件路径 |
| `evidences[].metadata` | `object` | 元数据 |
| `evidences[].matched_entities` | `string[]` | 命中的实体 |
| `evidences[].retrieval_signals` | `object` | 检索阶段信号 |
| `evidences[].payload` | `object` | 原始结构化数据或 chunk 载荷 |
| `timings` | `object` | 检索耗时信息，包含 v2 检索耗时、Playbook 检索耗时和 ACE 总耗时 |
| `retrieved_count` | `integer` | 返回证据数量 |
| `playbook` | `object` | Playbook 调试信息 |
| `playbook.used` | `boolean` | 是否启用 Playbook |
| `playbook.items` | `object[]` | 命中的 Playbook 规则 |
| `playbook.query_expansion_terms` | `string[]` | Playbook 生成或合并后的扩展词 |
| `playbook.source_hints` | `string[]` | Playbook 给出的来源提示 |
| `v2_request` | `object` | 实际发送给 v2 `/retrieve` 的请求体 |

## 2) Playbook 增强问答

```bash
curl --noproxy '*' -X POST http://127.0.0.1:6062/api/v1/playbook/query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "安全绳检测在 T4 上有部署吗？did 是多少？",
    "top_k": 12,
    "use_playbook": true,
    "playbook_top_k": 8
  }'
```

**输入参数**

除 Playbook 增强检索的全部参数外，还支持：

| 参数 | 类型 | 必填 | 默认值 | 说明/限制 |
| --- | --- | --- | --- | --- |
| `llm_config` | `object \| null` | 否 | - | 可选的大模型配置；不传则使用服务默认配置 |
| `stream` | `boolean` | 否 | `false` | 当前必须为 `false`；`true` 会返回 `400` |

**`llm_config` 参数**

| 参数 | 类型 | 必填 | 默认值 | 说明/限制 |
| --- | --- | --- | --- | --- |
| `model` | `string \| null` | 否 | `Qwen3.5-4B` | 模型名称 |
| `base_url` | `string \| null` | 否 | 配置值 | OpenAI-compatible API 地址 |
| `api_key` | `string \| null` | 否 | - | API key；未配置时会降级返回证据片段 |
| `max_tokens` | `integer \| null` | 否 | `2048` | 最大生成 token 数 |
| `temperature` | `number \| null` | 否 | `0.1` | 采样温度 |
| `top_p` | `number \| null` | 否 | `0.5` | nucleus sampling 参数 |
| `seed` | `integer \| null` | 否 | - | 采样随机种子，后端支持时生效 |

**输出参数**

包含 Playbook 增强检索的全部输出字段，并额外返回：

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `answer` | `string` | 基于 v2 证据和 Playbook 策略生成的回答 |
| `llm_config` | `object` | 本次回答使用的 LLM 配置，敏感字段会脱敏 |
| `run_id` | `string` | 本次问答运行 ID，可用于提交反馈 |

## 3) 用户反馈

```bash
curl --noproxy '*' -X POST http://127.0.0.1:6062/api/v1/playbook/feedback \
  -H 'Content-Type: application/json' \
  -d '{
    "run_id": "run-xxx",
    "feedback_type": "correction",
    "rating": 2,
    "corrected_answer": "应优先包含 adela 证据中的 did/rid。",
    "expected_evidence_ids": ["structured::adela::safety_rope-cuda11.0-trt7.1-fp16-T4_36274_100635"],
    "comment": "这类部署问题需要强制查 adela。"
  }'
```

**输入参数**

| 参数 | 类型 | 必填 | 默认值 | 说明/限制 |
| --- | --- | --- | --- | --- |
| `run_id` | `string` | 是 | - | `/playbook/query` 返回的运行 ID |
| `feedback_type` | `string` | 否 | `other` | 反馈类型；枚举：`helpful` / `harmful` / `correction` / `missing_evidence` / `other` |
| `rating` | `integer \| null` | 否 | - | 评分，范围 1-5 |
| `corrected_answer` | `string \| null` | 否 | - | 用户修正后的答案 |
| `expected_evidence_ids` | `string[]` | 否 | `[]` | 用户认为应当使用的证据 ID |
| `comment` | `string \| null` | 否 | - | 备注 |

**输出参数**

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `feedback_id` | `string` | 反馈记录 ID |
| `operation_id` | `string \| null` | 生成的待处理 Playbook 操作 ID |
| `status` | `string` | 当前固定为 `pending` |

## 4) 健康检查

```bash
curl --noproxy '*' http://127.0.0.1:6062/api/v1/playbook/health
```

**输出参数**

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `status` | `string` | `ok` 或 `degraded`；v2 不可达时为 `degraded` |
| `service` | `string` | 固定为 `ace-rag` |
| `v2` | `object` | v2 `/health` 返回内容，附加 `reachable` 和 `health_ms` |
| `playbook` | `object` | Playbook DB 状态 |
| `playbook.db_path` | `string` | Playbook SQLite 路径 |
| `playbook.active_items` | `integer` | 当前 active Playbook item 数量 |
