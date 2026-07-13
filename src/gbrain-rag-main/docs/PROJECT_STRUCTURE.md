# 项目结构与文件存放规范

最后更新：2026-05-06

## 当前结构

```text
gbrain-rag/
├── README.md
├── pyproject.toml
├── requirements.txt
├── environment.yml
├── scripts/
│   ├── build_index.py
│   └── import_swift_artifacts.py
├── src/gbrain_rag/
│   ├── api/
│   ├── core/
│   ├── ingest/
│   ├── llm/
│   └── retrieval/
├── tests/
├── sample_code/
├── docs/
├── data_source/
├── data/
│   └── index/
└── logs/
```

## 目录职责

| 路径 | 职责 |
| --- | --- |
| `src/gbrain_rag/api/` | FastAPI 路由、请求/响应 schema、统一 retrieve/query 入口 |
| `src/gbrain_rag/core/` | 配置、基础类型、文本工具 |
| `src/gbrain_rag/ingest/` | 数据源加载、PDF 解析、chunk 生成 |
| `src/gbrain_rag/retrieval/` | embedding、SQLite store、检索服务、排序、实体、查询理解 |
| `src/gbrain_rag/llm/` | OpenAI-compatible LLM 客户端、问答 prompt、统计规划 prompt |
| `scripts/` | 离线任务脚本，例如构建索引、导入 swift-rag 产物 |
| `tests/` | 单元测试和回归测试 |
| `sample_code/` | API 调用示例 |
| `docs/` | 项目说明、技术路线、改进记录 |
| `data_source/` | 原始和标准化语料 |
| `data/index/` | SQLite 索引及 WAL/SHM 文件 |
| `logs/` | 本地运行日志 |

## 数据文件存放建议

### 语料

保留在 `data_source/`：

- PDF 发版文档：`data_source/PDFs/`
- 模型发版表：`data_source/模型发版记录汇总.xlsx`
- 标准化模型发版行：`data_source/tables/model_release_records.jsonl`
- Adela 标准化记录：`data_source/adela/adela_release_records.jsonl`
- Adela 原始 JSON：`data_source/adela/data/`

原则：`data_source/` 放可追溯到业务原始来源或标准化来源的数据，不放运行时索引。

### 索引和运行时产物

保留在 `data/`：

- SQLite brain：`data/index/gbrain.sqlite3`
- SQLite WAL/SHM：`data/index/gbrain.sqlite3-wal`、`data/index/gbrain.sqlite3-shm`
- 后续可新增缓存或中间产物：`data/artifacts/`

原则：`data/` 放可重建的运行时产物；如果索引过大，建议不纳入版本管理。

### 文档

保留在 `docs/`：

- 设计和路线：放根目录。
- 历史调优记录：放 `docs/history/`。
- 后续如果按主题增多，可再拆 `docs/api/`、`docs/operations/`、`docs/evaluation/`。

## 本次已做的低风险整理

本次只整理文档，不修改原有功能：

- 新增 `docs/README.md` 作为文档索引。
- 新增技术说明类文档。
- 将历史性能调优记录移动到 `docs/history/`。

## 后续整理建议

以下建议不要求立刻执行，且需要结合是否纳入版本管理决定：

- 将 `logs/` 标记为本地运行产物，避免提交大日志。
- 将 `data/index/*.sqlite3*` 视为可重建索引，按团队需要决定是否提交。
- 将模型目录 `bge-m3`、`EvoQwen2.5-VL-Retriever-3B-v1` 继续用符号链接复用，避免重复占用空间。
- 为 `data_source/` 增加更明确的 README，说明哪些文件是原始数据、哪些是标准化数据、哪些由脚本生成。
- 如果后续脚本增多，可按 `scripts/ingest/`、`scripts/eval/`、`scripts/ops/` 拆分，但当前脚本数量较少，暂不需要过度拆分。

## 快速定位

常见任务与入口：

| 任务 | 入口 |
| --- | --- |
| 重建索引 | `scripts/build_index.py` |
| 修改 PDF 表格解析 | `src/gbrain_rag/ingest/pdf.py` |
| 修改数据源加载规则 | `src/gbrain_rag/ingest/loaders.py` |
| 修改 embedding 后端 | `src/gbrain_rag/retrieval/embeddings.py` |
| 修改 SQLite schema | `src/gbrain_rag/retrieval/store.py` |
| 修改检索融合 | `src/gbrain_rag/retrieval/service.py` |
| 修改领域查询理解 | `src/gbrain_rag/retrieval/query_understanding.py` |
| 修改问答 prompt | `src/gbrain_rag/llm/client.py` |
| 修改 API 行为 | `src/gbrain_rag/api/routes.py` |
