# Embedding 结果文件存放说明

当前项目将可直接查看的 embedding 结果文件统一放在：

- `data_source/embedding_artifacts/`

这样做的目的，是把不同数据类型产生的 embedding 结果和原始 `data_source` 放在同一棵目录下，便于排查、复用和备份。

## 目录划分

当前按产物类型区分为：

- `data_source/embedding_artifacts/documents/`
- `data_source/embedding_artifacts/tables/`
- `data_source/embedding_artifacts/adela/`

## 文档 embedding

文档类 embedding 结果统一放在：

- `data_source/embedding_artifacts/documents/`

当前包括两类产物。

### 文档检索向量库

离线入库后的 Milvus Lite `.db` 文件也属于文档数据源的 embedding 检索产物，现在默认存放在：

- `data_source/embedding_artifacts/documents/milvus_data_source_evoqwen_3b.db`
- `data_source/embedding_artifacts/documents/milvus_data_source_bge.db`

这些 `.db` 会被 `document` 文档问答和 unified 的 `document` 子检索直接读取；其中 `document` 固定表示模型发版文档正文（ONES 工作文档 / 发版 PDF）。

### 可查看的文档分块 embedding

文档分块接口 `chunking_embedding` 在生成分块向量后，会额外写入：

- `*.jsonl`：完整分块结果，每行一个节点
- `*.meta.json`：本次结果的元信息

文件名会带上：

- `doc_id`
- `doc_name`
- `embedding_models`

因此不同文档、不同模型组合的结果会自然区分开。

## 结构化行 embedding

表格检索和 adela 部署记录检索都使用“行级 embedding”，但当前已经按数据源分目录保存：

- 模型发版表：`data_source/embedding_artifacts/tables/`
- adela 部署记录：`data_source/embedding_artifacts/adela/`

当前会写入两类文件：

- `*.npy`：结构化数据所有行的 embedding 矩阵
- `*.meta.json`：矩阵对应的源数据说明

文件名和元信息会区分：

- 源结构化数据文件路径
- embedding 模型名
- searchable_fields

因此不同数据源、不同结构化数据文件、不同模型、不同可检索字段集合的结果不会混在一起。

缓存有效性会检查源文件路径、文件大小、mtime、模型名、可检索字段和 `row_ids`。如果源 JSONL 或字段配置变化，会自动重新生成。

## 用途

这些 embedding 结果文件主要用于：

- 查看实际生成了哪些 embedding
- 让文档、表格和 adela 检索结果按数据源分目录管理
- 让结构化检索支持文件级缓存
- 区分不同数据类型的 embedding 产物
