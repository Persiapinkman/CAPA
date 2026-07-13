# Swift-RAG 说明文档目录

最后更新：2026-04-24

该目录用于集中存放项目说明文档。当前文档组织方式是：具体 RAG 方法和数据源处理链路统一归档到 `data_sources/`，根目录只保留跨数据源共享的 API、服务启动、产物存放和目录索引类说明。

## 推荐阅读路径

1. 先看 `data_sources/README.md`，了解 documents / tables / adela / unified 四类链路的边界；其中 `documents` 专指模型发版文档正文（ONES 工作文档 / 发版 PDF）。
2. 如果要调用接口，看 `API_USAGE.md` 和对应数据源子目录。
3. 如果要重新入库或排查文档召回，看 `data_sources/documents/CHUNK_STRATEGY.md`、`data_sources/documents/CHUNK_METHOD_AND_RESULTS.md`、`EMBEDDING_ARTIFACT_STORAGE.md`。
4. 如果要压测或分析三类数据源问答耗时，分别看：
   - `data_sources/documents/RAG_CHAT_V1_TIMING.md`
   - `data_sources/tables/RAG_CHAT_V1_TIMING.md`
   - `data_sources/adela/RAG_CHAT_V1_TIMING.md`

## 数据源处理文档

| 数据源/链路 | 文档 | 说明 |
| --- | --- | --- |
| 总览 | `data_sources/README.md` | 当前数据源规模、接口映射和代码入口 |
| 文档全文 | `data_sources/documents/README.md` | 模型发版文档正文（ONES 工作文档 / 发版 PDF）、多模态 `pdf_blocks`、Milvus 入库、document 文档问答 |
| 模型发版表 | `data_sources/tables/README.md` | Excel 到 JSONL、行级检索、表格问答、embedding 缓存 |
| adela 部署记录 | `data_sources/adela/README.md` | 原始 adela JSON 到 JSONL、部署记录检索问答 |
| 统一检索网关 | `data_sources/unified/README.md` | documents / tables / adela 的 LLM 路由、并行检索和 RRF 融合 |
| 文档 chunk 策略 | `data_sources/documents/CHUNK_STRATEGY.md` | documents 各 `input_type` 的 chunk 策略 |
| 文档 chunk 结果 | `data_sources/documents/CHUNK_METHOD_AND_RESULTS.md` | documents 当前 chunk 方法和已入库统计结果 |
| document 文档问答耗时 | `data_sources/documents/RAG_CHAT_V1_TIMING.md` | models 发版文档正文一次检索问答耗时统计与分析 |
| table 表格问答耗时 | `data_sources/tables/RAG_CHAT_V1_TIMING.md` | tables v1 流程简介与耗时记录方式 |
| adela 部署记录问答耗时 | `data_sources/adela/RAG_CHAT_V1_TIMING.md` | adela v1 流程简介与耗时记录方式 |
| 表格检索方法 | `data_sources/tables/TABLE_RETRIEVAL_METHOD.md` | tables 行级 keyword / vector / hybrid 检索方法细节 |

## 通用文档

| 文档 | 说明 |
| --- | --- |
| `API_USAGE.md` | 核心 API 总览和最小调用示例 |
| `SERVICE_PORT_AND_API_CLI.md` | 服务启动、端口检查、curl 测试命令 |
| `EMBEDDING_ARTIFACT_STORAGE.md` | embedding 结果文件与缓存存放规则 |

## 当前项目进展摘要

- 文档全文：`75` 份 PDF 已通过 `pdf_blocks` 多模态方式入库，成功库包含 `1889` 个 chunk。
- 表格数据：`data_source/模型发版记录汇总.xlsx` 已标准化为 `293` 行 `model_release_records.jsonl`。
- adela 数据：`data_source/adela/data/` 下 `487` 份原始 JSON 已标准化为 `adela_release_records.jsonl`。
- API 能力：已支持 `document` 模型发版文档正文问答、表格问答、adela 问答和 unified 统一检索问答。
- 检索策略：文档默认 `hybrid`（向量 + BM25 + RRF），表格/adela 默认 `hybrid`（关键词规则 + 向量加权）。
