# 文档全文数据源

最后更新：2026-04-27

本文档说明 `data_source/PDFs` 等模型发版文档语料如何进入 RAG，并如何被 `document` 文档问答和 unified 网关复用。

这里的 `document` 是项目里的固定数据源代号，语义不是“任意文档”，而是“模型发版文档正文内容”。当前这批正文内容主要来自 ONES 工作文档同步出的发版 PDF，也包含正文里的 table、image 和页面上下文。

## 当前状态

- 主要语料目录：`data_source/PDFs/`
- 当前 PDF 数量：`75`
- PDF 默认处理方式：`autopdf`
- 默认向量库：`data_source/embedding_artifacts/documents/milvus_data_source_evoqwen_3b.db`
- BGE 向量库：`data_source/embedding_artifacts/documents/milvus_data_source_bge.db`
- collection：`llamacollection`
- PDF 来源链接映射：`data_source/pdf_reference_links.csv`
- 离线入库报告：`results/data_source_embedding_report__evoqwen2_5_vl_retriever_3b_v1__bge_m3.csv`

当前成功入库规模可参考 `CHUNK_METHOD_AND_RESULTS.md`：两套成功库均覆盖 `75` 份 PDF、`1909` 个 chunk。

这些 PDF 在产品语义上主要对应模型发版文档正文，因此 `document` 数据源最适合回答这类问题：

- 当前版本输入输出是什么。
- 默认阈值、精度、优化点、适用场景是什么。
- 这版追加了什么数据、有哪些标签。
- 发版文档正文、页内表格、配图附近的具体描述是什么。

## 离线处理流程

文档数据源的主流程在 `src/rag/pipeline/chenggong_pipeline.py`：

1. 解析数据源目录，优先使用 `DATA_SOURCE_DIR`，否则使用 `/app/data_source` 或仓库内 `data_source`。
2. 递归收集 `.pdf`、`.md`、`.txt`、`.json`、`.jsonl`、`.xlsx`。
3. 按后缀构造 `ChunkingEmbeddingRequest`。
4. 调用 `RAGService.chunking_embedding()` 生成 chunk 和 embedding。
5. 按 embedding 模型写入各自的 Milvus Lite `.db`。
6. 写出 CSV 报告到 `results/` 或 `DATA_SOURCE_RESULT_DIR`。

重新入库示例：

```bash
python -m src.rag.pipeline.chenggong_pipeline
```

常用环境变量：

```bash
DATA_SOURCE_DIR=/path/to/data_source \
DATA_SOURCE_MILVUS_URI=/path/to/milvus.db \
DATA_SOURCE_COLLECTION_NAME=llamacollection \
python -m src.rag.pipeline.chenggong_pipeline
```

注意：当一次请求包含多个 embedding 模型时，pipeline 会按 `src/core/config.py` 中的 `DATA_SOURCE_VECTOR_STORE_CONFIGS` 分库写入，避免不同维度模型混写。

## PDF: `autopdf`

PDF 由 `src/rag/pipeline/chenggong_pipeline.py` 逐页抽取文本，并包装成 AutoPDF 风格的
`content_list` JSON，再以 `input_type=autopdf` 调用 chunking 接口。

每页会生成一个 AutoPDF 内容段，通常包含：

- `page_id`：从 0 开始的页码
- `page_width` / `page_height`：PDF 页面尺寸
- `others`：页眉等附加结构，当前 PDF 抽取为空列表
- `content`：当前页的文本内容，元素类型为 `text`

`autopdf` 路径会继续按结构聚合成父级块，并做小块切分后生成 embedding。

## 其他文档类型

| 后缀 | input_type | 处理方式 |
| --- | --- | --- |
| `.pdf` | `autopdf` | 按页抽取文本并包装为 AutoPDF 风格 JSON 后切分 |
| `.md` | `markdown` | Markdown 结构父块 + 小块 |
| `.txt` / `.jsonl` | `raw` | `HierarchicalParser` 两层切分 |
| `.xlsx` | `json_list` | Excel 转 dict list，每个元素作为 chunk |
| `.json` list | `json_list` | 每个元素作为 chunk |
| `.json` dict | `autopdf` | 按结构聚合后切分 |

更细的 chunk 策略见 `CHUNK_STRATEGY.md`。

## 检索与问答接口

### document 模型发版文档正文问答

```text
POST /api/v1/rag/chat_engine/query
```

特点：一次检索模型发版文档正文后直接调用 LLM 回答，并返回 `retrieved_chunks`、`reference`、`answer`、`timings`。

最小请求示例：

```json
{
  "query": "safety_rope v0.2.1 追加了什么数据，标签有哪些？",
  "retrieval_method": "hybrid",
  "top_k": 5,
  "similarity_threshold": 0.5,
  "embedding_models": ["bge_m3", "EvoQwen2.5-VL-Retriever-3B-v1"]
}
```

## reference 处理

`document` 文档问答完成回答后，会用 `src/rag/reference_service.py` 将命中 chunk 的 `doc_name` 映射到来源链接。

映射文件：

```text
data_source/pdf_reference_links.csv
```

匹配规则支持完整路径和 basename；返回结构为：

```json
{"doc_name": "safety_rope v0.2.1.pdf", "url": "..."}
```

## 相关文档

- `CHUNK_METHOD_AND_RESULTS.md`
- `CHUNK_STRATEGY.md`
- `RAG_CHAT_V1_TIMING.md`
