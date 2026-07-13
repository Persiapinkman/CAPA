# 当前 Chunk 方法与结果（简版）

本文档用于快速说明：当前项目实际在用的 chunk 方法，以及当前库里的 chunk 结果现状。

## 1. 当前 chunk 方法

按 `input_type` 区分：

| input_type | 适用文件 | 当前切分方式（简述） |
| --- | --- | --- |
| `autopdf` | `.pdf` / 部分 `.json` | 先结构聚合，再按规则切小块 |
| `markdown` | `.md` | `MarkdownNodeParser` 先按结构切父块，再切小块 |
| `raw` | `.txt` / `.jsonl` | `HierarchicalParser` 两层切分（`chunk512 -> chunk128`），入库以小块为主 |
| `json_list` | `.xlsx` / 部分 `.json` | 每个元素直接作为一个 chunk |
| `mineru` | `mineru` 数据 | 版面/结构清洗后再形成父子块 |

补充：

- 当前 PDF 默认已走 `autopdf` 链路。
- 历史 `pdf_blocks` 多模态链路已不作为当前 PDF 默认入库方式。

## 2. 当前 chunk 结果（截至 2026-04-28）

### 2.1 统计口径

- 数据源范围：`data_source/PDFs` 当前共 `75` 份 PDF
- 报告来源：
  - `results/data_source_embedding_report__evoqwen2_5_vl_retriever_3b_v1__bge_m3.csv`
  - `results/data_source_embedding_report__evoqwen2_5_vl_retriever_7b_v1.csv`
- 向量库来源：
  - `data_source/embedding_artifacts/documents/milvus_data_source_evoqwen_3b.db`
  - `data_source/embedding_artifacts/documents/milvus_data_source_bge.db`
- “chunk 数”按向量库 `llamacollection` 记录数统计（单库单模型，不需要再按模型去重）
- “PDF 页数”按 `data_source/PDFs/*.pdf` 实际页数统计

### 2.2 离线 embedding 报告概览

| 报告文件 | 覆盖 PDF 数 | 记录数 | 成功 | 失败 | 备注 |
| --- | --- | --- | --- | --- | --- |
| `data_source_embedding_report__evoqwen2_5_vl_retriever_3b_v1__bge_m3.csv` | `75` | `150` | `150` | `0` | 每个 PDF 各跑 `2` 个模型（`EvoQwen-3B` + `bge_m3`） |
| `data_source_embedding_report__evoqwen2_5_vl_retriever_7b_v1.csv` | `75` | `75` | `0` | `75` | 全量失败，主要为 CUDA OOM |

### 2.3 成功入库后的向量库规模

| 向量库 | 模型 | 记录数（=chunk 数） | 覆盖 PDF 数 |
| --- | --- | --- | --- |
| `data_source/embedding_artifacts/documents/milvus_data_source_evoqwen_3b.db` | `EvoQwen2.5-VL-Retriever-3B-v1` | `1909` | `75` |
| `data_source/embedding_artifacts/documents/milvus_data_source_bge.db` | `bge_m3` | `1909` | `75` |

补充：

- 两个成功库的文档集合完全一致（同样的 `75` 份 PDF）
- 当前有效入库链路的 `input_type` 均为 `autopdf`

### 2.4 页数 / chunk 统计（基于成功入库的 75 份 PDF）

- 平均每个 PDF 页数：`7.29`（中位数 `6`，最小 `2`，最大 `31`）
- 平均每个 PDF chunk 数：`25.45`（中位数 `19`，最小 `2`，最大 `117`）
- 平均每页 chunk 数：`3.49`
- 当前 `autopdf` 入库内容来自逐页文本抽取并按结构/字符规则切分。

### 2.5 `autopdf` 验证样例

以 `PDFs/safety_rope v0.2.1.pdf` 为例，当前 artifact 记录：

- `input_type`: `autopdf`
- `node_count`: `11`

## 3. 一句话结论

- 方法层面：代码已切到 `autopdf` chunk。
- 结果层面：`EvoQwen-3B` 与 `bge_m3` 已完成 75 份 PDF 全量入库；当前数据下，平均每个 PDF 约 `7.29` 页、`25.45` 个 chunk（约 `3.49` chunk/页）。
