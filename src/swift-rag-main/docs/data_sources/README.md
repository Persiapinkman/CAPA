# 数据源处理说明

最后更新：2026-04-27

本目录按数据源类型归纳当前项目的处理链路。当前 Swift-RAG 已经形成四类主要问答入口：

| 数据源 | 文档位置 | 主要接口 | 适合回答的问题 |
| --- | --- | --- | --- |
| 模型发版文档正文 `documents` | `documents/README.md`、`documents/RAG_CHAT_V1_TIMING.md` | `/rag/chat_engine/query` | 模型发版文档的具体内容，来源于 ONES 工作文档 / 发版 PDF，适合回答版本细节、阈值、输入输出、优化点、正文/table/image 内容 |
| 模型发版表 `tables` | `tables/README.md`、`tables/RAG_CHAT_V1_TIMING.md` | `/rag/chat_engine/table_query` | 模型清单、负责人、OID、支持设备、更新时间等结构化字段 |
| adela 部署记录 `adela` | `adela/README.md`、`adela/RAG_CHAT_V1_TIMING.md` | `/rag/chat_engine/adela_query` | 部署平台、did/rid、部署版本、部署状态、did 跳转链接 |
| 统一检索网关 `unified` | `unified/README.md` | `/rag/chat_engine/unified_query` | 需要跨文档、表格、adela 交叉核对的问题 |

## 当前数据规模

| 数据源 | 当前文件/记录 | 标准化产物 | 缓存/索引位置 |
| --- | ---: | --- | --- |
| 模型发版文档 | `75` 份 PDF | `autopdf` 结构化 chunk | `data_source/embedding_artifacts/documents/*.db` |
| 模型发版表 | `293` 行 JSONL | `data_source/tables/model_release_records.jsonl` | `data_source/embedding_artifacts/tables/` |
| adela 部署记录 | `487` 行 JSONL | `data_source/adela/adela_release_records.jsonl` | `data_source/embedding_artifacts/adela/` |

## 统一约定

- API 基础路径：`http://127.0.0.1:6060/api/v1`
- 服务入口：`python -m src.main`
- 默认 collection：`llamacollection`
- 默认文档检索方法：`hybrid`，即模型发版文档正文的向量检索 + BM25 后用 RRF 融合
- 表格/adela 默认检索方法：`hybrid`，即规则关键词分数 + 向量相似度加权
- LLM 调用使用 OpenAI 兼容接口；请求里可通过 `llm_config` 覆盖默认配置
- `document` 是历史接口枚举，当前语义固定指“模型发版文档正文（ONES 工作文档 / 发版 PDF）”，不是泛指任意 document。

## 代码入口速查

| 能力 | 代码位置 |
| --- | --- |
| 请求/响应 schema | `src/api/schemas.py` |
| API 路由 | `src/api/routes.py` |
| 核心检索服务 | `src/rag/service.py` |
| 回答生成与统一路由 prompt | `src/rag/qa_service.py` |
| 文档离线入库 pipeline | `src/rag/pipeline/chenggong_pipeline.py` |
| PDF block 抽取 | `doc_loader/pdf_block_loader.py` |
| embedding 缓存落盘 | `src/rag/embedding_artifacts.py` |
