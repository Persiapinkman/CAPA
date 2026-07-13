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
cd ../gbrain-rag
PYTHONPATH=src python -m gbrain_rag.main
```

再启动 v3 `ace-rag`：

```bash
cd ../ace-rag
PYTHONPATH=src python -m ace_rag.main
```

使用 tmux 后台启动 v3 `ace-rag`：

```bash
mkdir -p logs
tmux -L ace-rag new -d -s ace-rag-api 'PYTHONPATH=src python -m ace_rag.main 2>&1 | tee -a logs/ace-rag-api.log'
```

常用 tmux 操作：

```bash
# 查看后台会话
tmux -L ace-rag ls

# 进入服务会话，查看实时输出
tmux -L ace-rag attach -t ace-rag-api

# 在 tmux 会话内退出但不停止服务：按 Ctrl-b，然后按 d

# 查看日志
tail -f logs/ace-rag-api.log

# 停止后台服务
tmux -L ace-rag kill-session -t ace-rag-api
```

注意：`ace-rag` 默认依赖 `http://127.0.0.1:6061/api/v1/rag`，所以需要先确认 v2 `gbrain-rag` 已经启动。

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
| Playbook 增强检索 | `POST` | `/playbook/retrieve` | 检索 Playbook，生成扩展词和来源提示，默认再调用 v2 `/retrieve` 返回证据；`playbook_only=true` 时只返回 Playbook 命中 |
| Playbook 增强问答 | `POST` | `/playbook/query` | Playbook 增强检索 + ACE 回答编排 + run 记录 |
| 用户反馈 | `POST` | `/playbook/feedback` | 对一次 `run_id` 提交反馈，生成待处理 Playbook 操作 |
| Playbook 整理归纳 | `POST` | `/playbook/organize` | 生成 Playbook memory 管理、去重、规则沉淀候选，不自动改库 |

说明：

- `ace-rag` 不维护 document / table / adela 索引；这些事实数据来自 `gbrain-rag`。
- Playbook 只提供策略、业务约定、风险提示和检索/回答方法，不作为具体事实来源。
- Playbook 自身检索使用 keyword/BM25 lexical scoring，不使用 embedding RAG；请求中的 embedding 参数仅透传给 v2 事实检索。
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

只检索 Playbook，不调用 v2 事实检索：

```bash
curl --noproxy '*' -X POST http://127.0.0.1:6062/api/v1/playbook/retrieve \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "安全绳检测在 T4 上有部署吗？did 是多少？",
    "use_playbook": true,
    "playbook_only": true,
    "playbook_top_k": 8
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
| `playbook_only` | `boolean` | 否 | `false` | 仅 `/retrieve` 使用；为 `true` 时只检索 Playbook，跳过 v2 事实检索，`evidences=[]`、`retrieved_count=0` |

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

当 `playbook_only=true` 时，`v2_request` 仅用于展示原本将发送给 v2 的请求，并带有 `_skipped=true` 和 `_skip_reason=playbook_only`；不会实际调用 v2 服务。

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

## 4) Playbook 整理归纳

```bash
curl --noproxy '*' -X POST http://127.0.0.1:6062/api/v1/playbook/organize \
  -H 'Content-Type: application/json' \
  -d '{
    "include_sections": ["source_routing", "online_feedback"],
    "min_confidence": 0.8,
    "max_items": 200
  }'
```

**技术方案**

| 方案 | 说明 |
| --- | --- |
| `semantic_merge_and_deduplicate` | 按 section、query_intents、source_hints、tags 聚类，输出合并/去重候选，适合 seed 与人工规则整理。 |
| `episodic_feedback_to_semantic_rule` | 把 `online_feedback` 的单次纠错记忆抽象为可复用规则，保留 run/feedback provenance 供审核。 |
| `single_item_summary` | 对独立规则生成摘要、关键扩展词和适用范围，适合作为只读 memory index 或人工编辑入口。 |

**输入参数**

| 参数 | 类型 | 必填 | 默认值 | 说明/限制 |
| --- | --- | --- | --- | --- |
| `include_sections` | `string[] \| null` | 否 | - | 只整理指定 section；不传则包含全部 active Playbook |
| `include_inactive` | `boolean` | 否 | `false` | 是否包含 inactive item |
| `min_confidence` | `number` | 否 | `0.0` | 最低置信度，范围 0-1 |
| `max_items` | `integer` | 否 | `200` | 最大整理条数，范围 1-1000 |

**输出参数**

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `item_count` | `integer` | 本次纳入整理的 Playbook item 数量 |
| `strategies` | `object[]` | 当前服务支持的整理归纳技术方案 |
| `candidates` | `object[]` | 整理归纳候选；当前 `apply_mode` 固定为 `preview_only`，不会自动写库 |
| `candidates[].strategy` | `string` | 候选对应的技术方案 |
| `candidates[].item_ids` | `string[]` | 候选覆盖的原始 Playbook item |
| `candidates[].summary` | `string` | 归纳摘要 |
| `candidates[].rationale` | `string` | 分组依据 |

## 5) 健康检查

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
