# Embedding Artifact 目录说明

该目录用于统一存放项目运行过程中生成的 embedding 结果文件，按数据类型区分子目录。

当前目录结构：

- `documents/`：文档分块后的 embedding 结果
- `tables/`：模型发版表行级 embedding 缓存文件
- `adela/`：adela 部署记录行级 embedding 缓存文件

## documents

`documents/` 中的文件对应文档数据源的 embedding 结果，包括文档检索用 Milvus Lite `.db` 和 `chunking_embedding` 可查看结果。

当前常见 Milvus Lite 文件：

- `milvus_data_source_evoqwen_3b.db`
- `milvus_data_source_bge.db`

这些 `.db` 是 document 文档问答、流式 RAG 和 unified 的 `document` 子检索默认读取的文件。

`chunking_embedding` 处理结果包括：

- `*.jsonl`：每行一个分块节点，包含文本、元数据和 `embeddings`
- `*.meta.json`：本次生成的说明信息，例如 `doc_id`、`doc_name`、`input_type`、`embedding_models`

## tables

`tables/` 中的文件对应模型发版表检索的行级 embedding 结果。

- `*.npy`：行向量矩阵
- `*.meta.json`：与矩阵对应的元信息，例如 `data_path`、`model_name`、`searchable_fields`、`row_ids`

表格 embedding 会优先读取这里的文件缓存；当源表格文件、字段集合或模型发生变化时，会重新生成。

## adela

`adela/` 中的文件对应 adela 部署记录检索的行级 embedding 结果。

- `*.npy`：部署记录行向量矩阵
- `*.meta.json`：与矩阵对应的元信息，例如 `artifact_namespace=adela`、`data_path`、`model_name`、`searchable_fields`、`row_ids`

adela embedding 会优先读取这里的文件缓存；当源 JSONL、字段集合或模型发生变化时，会重新生成。
