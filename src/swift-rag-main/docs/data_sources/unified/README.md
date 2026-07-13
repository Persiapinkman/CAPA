# 统一检索网关

最后更新：2026-04-27

统一检索网关不是新的原始数据源，而是把 `documents`、`tables`、`adela` 三类数据源接到同一个入口。

- 问答接口：`POST /api/v1/rag/chat_engine/unified_query`
- 直出检索接口：`POST /api/v1/rag/chat_engine/unified_retrieve`（不走最终 LLM answer）

## 适用场景

适合这类问题：

- 需要同时查模型发版文档正文细节和结构化模型清单。
- 需要核对某个模型在文档、表格和 adela 中是否都有记录。
- 用户问题里同时包含模型名、平台、OID、did/rid、版本等跨源线索。
- 不确定应该问哪个接口，希望系统先做数据源路由。

如果问题很明确只问单一来源，优先使用对应接口，延迟和可解释性更好。

## 工作流程

实现入口：`src/api/routes.py` 的 `query_unified_with_gateway()` 和 `retrieve_unified_with_gateway()`。

一次请求的流程：

1. 根据 `document_config.enabled`、`table_config.enabled`、`adela_config.enabled` 得到可用来源。
2. 若 `route_with_llm=true`，调用 `route_unified_sources()` 让 LLM 选择需要检索的数据源。
3. 并行执行 selected sources 的检索。
4. 将文档 chunk、表格行、adela 行统一转换为 `UnifiedEvidenceItem`。
5. 用 RRF 融合三路证据。
6. 将融合证据交给 LLM 生成最终答案。
7. 返回 `route_plan`、`source_status`、`fused_evidences`、`answer` 和 `timings`。

`unified_retrieve` 与上面流程相比，跳过第 2 步（LLM 路由）和第 6 步（LLM 生成回答），直接返回融合证据。

## LLM 路由规则

路由 prompt 位于 `src/rag/qa_service.py` 的 `UNIFIED_ROUTE_PROMPT`。

当前数据源语义：

| 来源 | 适合问题 |
| --- | --- |
| `document` | 模型发版文档正文内容，来源于 ONES 工作文档 / 发版 PDF；适合查输入输出、阈值、优化点、追加数据、标签、背景说明、正文/table/image 细节 |
| `table` | 模型发版信息汇总表；适合查模型列表、负责人、设备、OID、更新时间、推荐配置等结构化字段 |
| `adela` | 部署平台、did/rid、部署状态、部署版本 |

当路由失败、返回空选择或 JSON 解析失败时，会 fallback 到所有已启用数据源。

补充说明：

- `document` 不是一个宽泛的“文档类来源”标签，而是本项目第一个核心数据源的固定代号。
- 如果用户在问“当前版本是什么输入输出”“这版默认阈值是多少”“这版追加了什么数据/标签”“发版文档里怎么写的”，这类问题通常必须命中 `document`。
- `table` 更像结构化台账，适合做筛选、枚举、负责人/OID/更新时间类查询，但不替代发版文档正文。

## 融合方式

统一网关使用 RRF：

```text
score += 1 / (rrf_k + rank)
```

默认参数：

- `fused_top_k = 12`
- `rrf_k = 60`

融合前，每个来源会先形成统一证据：

| 字段 | 说明 |
| --- | --- |
| `evidence_id` | 形如 `document::{chunk_id}`、`table::{row_id}`、`adela::{row_id}` |
| `source_type` | `document` / `table` / `adela` |
| `score` | RRF 融合后的分数 |
| `source_rank` | 来源内排名 |
| `source_score` | 来源内原始分数 |
| `title` | 文档名、模型名或记录名 |
| `snippet` | 给用户和 LLM 的摘要 |
| `payload` | 原始结构化信息 |

## 调用示例

```bash
python sample_code/unified_chat_api_client.py
```

```bash
python sample_code/unified_retrieve_api_client.py
```

最小 curl：

```bash
curl -X POST http://127.0.0.1:6060/api/v1/rag/chat_engine/unified_query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "安全绳使用到的相关模型有哪些记录？",
    "fused_top_k": 12,
    "rrf_k": 60,
    "stream": false,
    "route_with_llm": true
  }'
```

流式模式（SSE）：

```bash
curl -N -X POST http://127.0.0.1:6060/api/v1/rag/chat_engine/unified_query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "安全绳使用到的相关模型有哪些记录？",
    "stream": true
  }'
```

关闭 LLM 路由、固定检索所有启用来源：

```json
{
  "query": "安全绳使用到的相关模型有哪些记录？",
  "route_with_llm": false
}
```

一个更能体现 `document` 价值的例子：

```json
{
  "query": "车辆属性识别当前版本的模型输入输出、推荐阈值和版本优化点分别是什么？",
  "route_with_llm": true
}
```

这类问题通常应至少命中 `document`，必要时再补 `table` 或 `adela`。

只启用表格和 adela：

```json
{
  "query": "哪些安全绳模型在 T4 平台有部署记录？",
  "document_config": {"enabled": false},
  "table_config": {"enabled": true, "top_k": 20},
  "adela_config": {"enabled": true, "top_k": 20}
}
```

在 `unified_retrieve` 里可直接限制检索来源范围（无需走 LLM 路由）：

```json
{
  "query": "哪些安全绳模型在 T4 平台有部署记录？",
  "source_types": ["table", "adela"]
}
```

## 响应重点字段

- `route_plan`：是否启用 LLM 路由、选择了哪些来源、是否 fallback、原因。
- `source_status`：每个来源是否启用、是否成功、检索耗时、命中数、参与融合数。
- `fused_evidences`：融合后的统一证据列表。
- `answer`：基于融合证据生成的最终答案。
- `timings`：`route_ms`、`retrieve_ms`、`fuse_ms`、`answer_ms`、`total_ms`。

`unified_retrieve` 返回重点：

- `selected_sources`：本次实际检索的数据源（受 `source_types` 和 `*.enabled` 双重约束）。
- `source_status`：每个来源是否启用、是否成功、检索耗时、命中数、参与融合数。
- `fused_evidences`：融合后的统一证据列表。
- `timings`：`retrieve_ms`、`fuse_ms`、`total_ms`。

## 注意事项

- unified 的检索阶段是并行的，但如果启用 LLM 路由和最终 LLM 回答，总耗时仍会受模型服务影响。
- 来源内分数口径不同，最终排序不要直接解释为同一种相似度；`score` 是 RRF 融合分。
- 对强结构化问题，单独调用 `table_query` 或 `adela_query` 通常更直接。
- 对发版细节和长文本解释，单独调用 `query`（即 `document` 模型发版文档正文问答）通常上下文更完整。
