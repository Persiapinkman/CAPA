# 表格检索方法说明

本文档用于简单说明当前项目中的表格检索实现，便于快速理解 `POST /api/v1/rag/chat_engine/table_query` 的工作方式。

## 1. 当前能力概览

当前表格问答不是走通用文档分块检索，而是直接对结构化表格的“行”做检索。

整体流程如下：

1. 从 JSONL 文件加载表格数据，每一行对应一条结构化记录。
2. 基于请求参数选择 `keyword`、`vector` 或 `hybrid` 检索。
3. 对所有行计算相关性分数并排序，截取 `top_k` 条命中行。
4. 将命中行交给 LLM，总结生成最终回答。

相关代码位置：

- 路由入口：`src/api/routes.py`
- 表格检索实现：`src/rag/service.py`
- 表格问答生成：`src/rag/qa_service.py`

## 2. 当前数据来源

默认表格数据文件为：

- `data_source/tables/model_release_records.jsonl`

默认配置位置：

- `src/core/config.py`

当前默认配置包括：

- `TABLE_DATA_JSONL_PATH`：默认表格 JSONL 路径
- `TABLE_SEARCHABLE_FIELDS`：默认参与检索的字段
- `TABLE_RETURN_FIELDS`：默认返回给前端和 LLM 的字段

JSONL 中每一行是一条完整记录，典型字段包括：

- `row_id`
- `target_name`
- `algorithm_type`
- `algorithm_name`
- `application_scene`
- `owner`
- `model_name`
- `supported_device`
- `recommended_config`
- `last_updated`
- `last_updated_month`
- `ones_release_link`
- `oid`

其中 `row_id` 是行级唯一标识。

## 3. 检索对象是什么

当前检索粒度是“表格行”，不是单元格，也不是整张表。

服务会先从每一行里抽取一组可检索字段，拼成一段检索文本，格式类似：

```text
target_name: 人体
algorithm_type: 动作
algorithm_name: 扶梯摔倒
application_scene: 香港地铁扶梯摔倒
owner: 肖坤
model_name: KM_falcon_tsm18_elevator_tumble_nart_cuda11.0-trt7.1-fp16-T4_b3_5.0.2.model
supported_device: T4
recommended_config: cuda11.0-trt7.1-int8-T4
last_updated: 2024-11-28
last_updated_month: 2024-11
```

向量检索时，实际就是对这段“行级拼接文本”做 embedding；关键词检索时，也是针对这些字段内容逐字段打分。

当前实现还会把表格行 embedding 落盘到：

- `data_source/embedding_artifacts/tables/`

服务启动后会优先读取该目录下的文件缓存；若缓存不存在或已失效，再重新计算。

## 4. 三种检索方式

### 4.1 `keyword`

关键词检索会逐字段计算匹配分数，当前主要包含三类信号：

- 查询全文是否直接包含在字段值中
- 字段值是否反向包含在查询里
- 查询分词结果与字段分词结果的重叠比例

一个字段只要有命中，就会记录到 `matched_fields` 中。

单行最终关键词分数不是简单取最大值，而是：

```text
final_keyword_score = min(1.0, max_score * 0.7 + avg_score * 0.3)
```

这意味着当前实现同时兼顾“最强命中字段”和“整体命中情况”。

### 4.2 `vector`

向量检索的做法是：

1. 对 query 计算 embedding。
2. 对每一行的拼接文本计算 embedding。
3. 使用余弦相似度计算 query 与每一行的相关性。
4. 按相似度排序后返回。

当前限制：

- `vector` / `hybrid` 模式下，表格检索只支持一个 embedding 模型。

### 4.3 `hybrid`

混合检索会同时计算关键词分数和向量分数，然后做加权融合。

当前实现的融合公式是：

```text
final_score = keyword_score * 0.45 + vector_score * 0.55
```

可以理解为当前实现略微偏向向量召回结果。

## 5. 排序与过滤

在得到每一行的最终分数后，服务会：

1. 按 `similarity_threshold` 做阈值过滤。
2. 按分数从高到低排序。
3. 取前 `top_k` 条作为 `matched_rows`。

默认参数见 `src/api/schemas.py`：

- `top_k = 20`
- `similarity_threshold = 0.15`
- `retrieval_method = hybrid`

## 6. 返回内容

接口返回的核心字段包括：

- `matched_rows`：命中的结构化行
- `answer`：基于命中行生成的最终回答
- `timings`：检索、回答和总耗时

其中每条命中行包含：

- `row_id`：行 ID
- `score`：最终分数
- `matched_fields`：关键词命中的字段列表
- `entity`：返回给前端和 LLM 的结构化字段内容

需要注意：

- 即使使用 `vector` 模式，当前响应结构里也仍会保留 `matched_fields`，但它主要来自关键词匹配阶段的计算结果。

## 7. LLM 回答阶段

检索完成后，服务不会直接把命中行原样返回作为最终答案，而是会再走一层 LLM 总结。

当前做法是：

1. 把 `matched_rows` 格式化成提示词上下文。
2. 连同用户问题一起发送给 LLM。
3. 由 LLM 基于命中行生成自然语言回答。

因此这个接口本质上是“表格检索 + LLM 问答”，而不是纯检索接口。

## 8. 当前实现特点与限制

当前方案的特点：

- 实现简单，适合结构稳定、字段明确的表格数据
- 检索粒度固定为“行”，便于直接返回结构化结果
- 支持字段级关键词解释，也支持语义向量召回

当前限制：

- 只支持 JSONL 行数据，不直接对 Excel 原文件做在线检索
- `vector` / `hybrid` 下只支持一个 embedding 模型
- 混合融合权重是代码写死的，当前没有暴露成配置项
- 关键词匹配是轻量规则打分，不是 BM25
- 当前没有单元格级精排或复杂表结构理解能力

## 9. 调用示例

可直接运行：

```bash
python sample_code/table_chat_api_client.py
```

默认请求示例等价于：

```json
{
  "query": "安全绳有哪些模型？",
  "retrieval_method": "hybrid",
  "top_k": 20,
  "similarity_threshold": 0.15,
  "data_path": "data_source/tables/model_release_records.jsonl",
  "embedding_models": ["bge_m3"]
}
```

接口地址：

```text
POST /api/v1/rag/chat_engine/table_query
```

## 10. 相关文档

- `../../API_USAGE.md`
- `README.md`
