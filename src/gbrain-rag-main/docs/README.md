# GBrain RAG 文档目录

最后更新：2026-05-06

本目录集中说明当前项目相对 `../swift-rag` 的改进、端到端技术路线，以及借鉴开源 `gbrain` 项目的方法。当前项目是一个面向 RD 模型发版资料的本地 RAG 服务：语料来自模型发版 PDF、模型发版汇总表和 Adela 部署记录，索引与检索统一收敛到 SQLite brain。

## 推荐阅读路径

1. [IMPROVEMENTS_OVER_SWIFT_RAG.md](IMPROVEMENTS_OVER_SWIFT_RAG.md)：先了解本项目对传统 swift-rag 三源 RAG 的主要改进。
2. [TECHNICAL_ROUTE.md](TECHNICAL_ROUTE.md)：再看数据源处理、embedding、检索、回答的完整链路。
3. [GBRAIN_METHOD.md](GBRAIN_METHOD.md)：了解开源 gbrain 方法及本项目如何落地其中的思想。
4. [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)：查看当前目录组织、重要文件和后续规范建议。

## 文档说明

| 文档 | 说明 |
| --- | --- |
| `IMPROVEMENTS_OVER_SWIFT_RAG.md` | 对比 `swift-rag`，说明统一 SQLite brain、实体图、结构化意图、字段摘要、降级策略等改进 |
| `TECHNICAL_ROUTE.md` | 说明数据源摄取、chunk、embedding、SQLite 索引、混合检索、证据构造和 LLM 回答流程 |
| `GBRAIN_METHOD.md` | 简述 gbrain 的 local-first memory/graph/RAG 方法，以及本项目的对应实现 |
| `PROJECT_STRUCTURE.md` | 当前项目结构、文件存放规范和不改变功能的整理建议 |
| `history/` | 历史调优和问题修复记录 |

## 代码入口速查

| 模块 | 主要职责 |
| --- | --- |
| `scripts/build_index.py` | 离线构建 SQLite 索引和 embedding |
| `src/gbrain_rag/ingest/` | 数据源发现、PDF/表格/JSONL 解析、chunk 生成 |
| `src/gbrain_rag/retrieval/store.py` | SQLite brain：documents/chunks/embeddings/entities/entity_links/FTS |
| `src/gbrain_rag/retrieval/service.py` | 路由、vector/keyword/structured/graph 混合检索和证据 payload |
| `src/gbrain_rag/retrieval/query_understanding.py` | 本地域名查询理解、同义词扩展、结构化行打分 |
| `src/gbrain_rag/llm/client.py` | OpenAI-compatible LLM 调用、问答 prompt、结构化统计规划 |
| `src/gbrain_rag/api/routes.py` | FastAPI 接口、unified retrieve/query、结构化聚合短路和降级 |

## 与 swift-rag 的关系

`swift-rag` 是本项目的数据与接口风格来源：它已经沉淀了 document/table/adela 三类数据源、统一检索入口和 OpenAI-compatible 问答接口。本项目在此基础上引入 gbrain 式的统一本地 brain，目标不是替换语料，而是让三类语料在同一个索引、实体和证据模型下协同工作。
