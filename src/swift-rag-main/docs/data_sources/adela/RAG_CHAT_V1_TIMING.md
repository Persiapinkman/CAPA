# adela 部署记录问答 v1 简介与耗时

## v1 简介

当前的 adela 问答 v1 是“结构化部署记录检索 + 大模型生成”链路，接口流程如下：

1. 接收用户问题。
2. 在 `adela_release_records.jsonl` 上执行记录级检索。
3. 根据 `retrieval_method`（`keyword` / `vector` / `hybrid`）计算相关性分数。
4. 得到 `matched_records` 后，按 `source_path` 回读原始 JSON，补齐 `model_info/benchmark_info`。
5. 基于命中 `did` 生成部署跳转链接（单条记录写入 `entity.reference`，并汇总为顶层去重 `reference`）。
6. 将命中记录格式化为上下文，调用 LLM 生成答案。
7. 返回回答、命中记录、reference 和分步耗时 `timings`。

当前接口已经支持：

- 在响应中返回 `retrieve_ms`、`answer_ms`、`total_ms`
- 在响应中返回基于 `did` 去重后的 `reference`（部署跳转链接列表）
- 在调用脚本中记录端到端耗时，便于后续聚合分析

## 当前默认使用方法

在不额外传参（或仅传 `query`）时，adela 问答默认按下面方式运行：

- 默认接口：`POST /api/v1/rag/chat_engine/adela_query`
- 默认检索方法：`hybrid`
- 默认 `top_k`：`20`
- 默认 `similarity_threshold`：`0.15`
- 默认向量模型：`["bge_m3"]`
- 默认数据文件：`data_source/adela/adela_release_records.jsonl`
- 默认原始目录（自动导出兜底）：`data_source/adela/data/`
- 默认检索字段：`ADELA_SEARCHABLE_FIELDS`
- 默认返回字段：`ADELA_RETURN_FIELDS`

也就是“关键词规则打分 + 向量相似度”融合检索，命中后补齐元数据字段，再交给 LLM 生成答案。

## 检索方法与过程（按代码实现）

以下流程对应 `src/rag/service.py` 的 `adela_chat_retrieving()`。

### 1. 请求接入与耗时统计边界

- 接口入口：`POST /api/v1/rag/chat_engine/adela_query`。
- `retrieve_ms` 统计范围：`rag_service.adela_chat_retrieving(request)` 整段执行时间。
- `answer_ms` 统计范围：`answer_adela_question()` 调用 LLM 生成答案时间。
- `total_ms`：从请求进入路由到响应返回。

### 2. 数据文件准备（adela 特有）

1. 先确定 `searchable_fields`（请求值优先，否则 `ADELA_SEARCHABLE_FIELDS`）。
2. 检查 `data_path`（默认 `data_source/adela/adela_release_records.jsonl`）是否可用。
3. 若出现以下任一情况，触发重建 JSONL：
   - `data_path` 不存在；
   - `adela_release_records.csv` 比 `data_path` 更新；
   - 现有 JSONL 缺少兼容字段（至少 `model_name`、`label_list`、`source_path`）。
4. 重建依赖 `source_dir`（adela 原始 JSON 目录）；未提供或目录不存在会报错。
5. 重建完成后，清理该数据源对应的内存缓存（行缓存与 embedding 缓存键）。

### 3. 检索主流程（复用 table 行检索）

adela 的检索分数计算本质复用 `table_chat_retrieving()`，仅命名空间和字段集合不同：

- 命名空间：`artifact_namespace="adela"`（缓存目录会落到 `data_source/embedding_artifacts/adela/`）。
- 关键词、向量、hybrid 的打分规则与 table 完全一致：
  - `keyword_score`：字段匹配规则 + `max*0.7 + avg*0.3`；
  - `vector_score`：query 与行向量余弦相似度；
  - `hybrid_score = keyword*0.45 + vector*0.55`。
- 过滤与排序规则也一致：
  - `final_score < similarity_threshold` 过滤；
  - 按分数降序；
  - 截取 `top_k`。

补充：代码默认 `similarity_threshold` 为空时按 `0.0` 处理。

### 4. 命中记录增强（adela 特有）

对每条命中行，按 `source_path` 回读原始 JSON，并做字段回填：

- 若 `model_info`/`benchmark_info`/`name`/`model_name`/`platform`/`rid`/`did` 缺失，则用原始 JSON 补齐。
- 额外做 `name` 与 `model_name` 的双向兜底（其中一个为空时用另一个补）。
- 基于 `did` 生成 `entity.reference`（按 `ADELA_DEPLOYMENT_URL_TEMPLATE` 拼接）。
- 最终输出字段不是整行原样返回，而是按 `final_return_fields` 裁剪后返回。

其中 `final_return_fields` = 请求 `return_fields`（或默认 `ADELA_RETURN_FIELDS`）再补上关键字段：
`model_name,name,platform,rid,did,label_list,labels,source_file,model_info,benchmark_info`。

### 5. 回答阶段

- 将 `matched_records` 格式化到 `ADELA_QA_PROMPT` 的 `<adela_records>` 上下文。
- 调用 OpenAI 兼容接口生成答案（`enable_thinking=False`）。
- adela 链路不走 documents 的 PDF 映射（`src/rag/reference_service.py`），而是使用 `did` 拼接部署链接并返回 `reference`。

## 当前耗时情况

以下数据来自 2026-04-21 的实测：

```bash
python sample_code/rag_chat_benchmark.py \
  --chat-api-url http://127.0.0.1:6060/api/v1/rag/chat_engine/adela_query \
  --query "有哪些 cuda11.0-trt7.1-fp16-T4 的部署模型？" \
  --repeat 10 \
  --similarity-threshold 0.15 \
  --embedding-models bge_m3 \
  --output-jsonl results/adela_chat_benchmark_20260421_repeat10.jsonl
```

- 样本量：10（1 个问题 x 重复 10 次）
- 成功率：100%（10/10）

### 端到端耗时（客户端视角，ms）

| 指标 | avg | min | p50 | p95 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 客户端总耗时 | 4658.601 | 2083.567 | 3563.099 | 8341.426 | 8737.474 |
| 服务端总耗时 | 4371.532 | 1804.597 | 3285.589 | 7977.212 | 8464.262 |
| 客户端额外开销 | 287.069 | 233.985 | 274.999 | 399.269 | 475.440 |

### 服务端分步耗时（ms）

| 阶段 | avg | min | p50 | p95 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 检索耗时 | 306.751 | 292.159 | 300.191 | 333.759 | 343.322 |
| 回答耗时 | 4064.749 | 1506.577 | 2976.178 | 7673.181 | 8158.921 |

- `retrieve_ms` 和 `answer_ms` 可直接从响应 `timings` 获取。
- 该压测脚本原本为 document 问答设计，`retrieved_count/reference_count` 字段对 adela 语义不完全对应，建议重点看 `timings` 与 `answer_length`。
