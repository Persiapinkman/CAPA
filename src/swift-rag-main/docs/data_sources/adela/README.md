# adela 部署记录数据源

最后更新：2026-04-24

本文档说明 adela 部署记录如何结合 `adela_release_records.csv + 原始 JSON` 标准化为 JSONL，并被 `POST /api/v1/rag/chat_engine/adela_query` 检索问答。

## 当前状态

- 原始目录：`data_source/adela/data/`
- 发布记录 CSV：`data_source/adela/adela_release_records.csv`
- 原始 JSON 数量：`487`
- 标准化 JSONL：`data_source/adela/adela_release_records.jsonl`
- 当前标准化记录数：`487`
- 默认检索方法：`hybrid`
- 默认向量模型：`bge_m3`
- 默认部署跳转链接模板：`ADELA_DEPLOYMENT_URL_TEMPLATE`（`...deployment_id={did}`）
- 行级 embedding 缓存目录：`data_source/embedding_artifacts/adela/`

adela 数据源复用表格数据源的结构化行检索代码，但默认字段、数据导出方式、回答 prompt 和 embedding 缓存目录都不同。

## 标准化导出

导出脚本：`scripts/export_adela_records_to_jsonl.py`

```bash
python scripts/export_adela_records_to_jsonl.py
```

自定义路径：

```bash
python scripts/export_adela_records_to_jsonl.py \
  --input-dir data_source/adela/data \
  --records-csv data_source/adela/adela_release_records.csv \
  --output data_source/adela/adela_release_records.jsonl
```

如果调用 `adela_query` 时 `data_path` 不存在，服务也会尝试用请求中的 `source_dir` 自动导出。

## 原始 JSON 到标准化行

解析逻辑位于 `src/rag/adela_dataset.py`。

处理流程：

1. 先读取 CSV（`did,rid,model_name,platform,label_list`）。
2. 通过 `rid+did` 定位元数据 JSON（文件名约定：`modelname_platform_rid_did.json` 或 `modelname-platform_rid_did.json`）。
3. 从对应 JSON 读取 `model_info` / `benchmark_info` 以及部署状态、版本等字段。

每条标准化记录主要包含：

| 字段 | 说明 |
| --- | --- |
| `row_id` | 原始 JSON 文件名 stem |
| `did` / `rid` | adela 部署/记录 ID |
| `model_name` | CSV 中的模型名（可用于模型名检索） |
| `name` | 模型或部署名称 |
| `label_list` / `labels` | CSV 标签原串 / 拆分后的标签列表（`|` 分隔） |
| `type` | 记录类型 |
| `platform` | 部署平台，如 `cuda11.0-trt7.1-fp16-T4` |
| `status` | 部署状态，如 `SUCCESS` |
| `version` | 由 `major.minor.patch` 拼出的版本号 |
| `version_train_date` | 训练日期 |
| `returncode` | 命令返回码 |
| `source_file` | 相对源文件路径 |
| `command` | 原始命令 |
| `stderr` | 标准错误摘要 |
| `model_info` | 来自元数据 JSON 的模型信息 |
| `benchmark_info` | 来自元数据 JSON 的 benchmark 信息 |
| `source_path` | 原始文件绝对路径 |
| `search_text` | 默认可检索字段拼接文本 |

默认检索字段来自 `ADELA_SEARCHABLE_FIELDS`：

```text
model_name, name, label_list, labels, type, platform, status, did, rid, version, version_train_date, source_file
```

默认返回字段来自 `ADELA_RETURN_FIELDS`，包含模型名、标签、版本拆分字段，以及 `model_info/benchmark_info`。

## 检索方式

adela 检索入口：`RAGService.adela_chat_retrieving()`。

| 方法 | 说明 |
| --- | --- |
| `keyword` | 对 `name`、`platform`、`did/rid`、`version` 等字段做规则打分 |
| `vector` | 对记录级 `search_text` 生成 embedding 后计算余弦相似度 |
| `hybrid` | 关键词分数与向量分数加权融合 |

命中记录后，服务会按 `source_path` 回读对应 JSON，补齐并返回 `model_info`（以及 `benchmark_info`）。
同时会基于 `did` 生成 `matched_records[].entity.reference`，并在响应顶层输出去重后的 `reference` 列表。

命中记录会被格式化为 `<adela_records>`，再由 `answer_adela_question()` 调用 LLM 生成答案。

## embedding 缓存

adela 和 tables 一样使用行级 embedding，但结果单独放在：

```text
data_source/embedding_artifacts/adela/
```

当前可见缓存包括：

```text
adela_release_records__bge_m3__*.npy
adela_release_records__bge_m3__*.meta.json
adela_release_records__evoqwen2_5_vl_retriever_3b_v1__*.npy
adela_release_records__evoqwen2_5_vl_retriever_3b_v1__*.meta.json
```

缓存会随源文件大小、mtime、字段集合、row_ids 或模型变化而失效重建。

## 调用示例

```bash
python sample_code/adela_chat_api_client.py
```

或直接 curl：

```bash
curl -X POST http://127.0.0.1:6060/api/v1/rag/chat_engine/adela_query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "有哪些 cuda11.0-trt7.1-fp16-T4 的部署模型？",
    "retrieval_method": "hybrid",
    "top_k": 20,
    "similarity_threshold": 0.15,
    "embedding_models": ["bge_m3"]
  }'
```

响应重点字段：

- `matched_records`：命中的 adela 结构化记录
- `matched_records[].entity`：返回给前端和 LLM 的字段
- `matched_records[].entity.reference`：每条命中记录的部署跳转链接（由 `did` 拼接）
- `reference`：去重后的 adela 部署链接列表（顶层字段）
- `answer`：LLM 总结后的答案
- `timings`：检索、回答和总耗时

## 适用与限制

适合：查询部署平台、部署状态、did/rid、版本号、训练日期、某平台有哪些模型。

限制：

- 当前语义与关键词检索粒度均为“部署记录行”。
- `vector` / `hybrid` 模式下只支持一个 embedding 模型。
- `stderr` 和 `command` 可返回给 LLM，但通常不应作为稳定业务字段过度依赖。

## 相关文档

- `RAG_CHAT_V1_TIMING.md`
- `../../API_USAGE.md`
