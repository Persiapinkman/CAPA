# Single-Step RAG Benchmark (100, Unified 三源版)

该目录提供基于 unified 三源语料构建的 100 道单步题集，覆盖：

- `documents`：`data_source/PDFs`（文档语料）
- `tables`：`data_source/tables/model_release_records.jsonl`（表格结构化语料）
- `adela`：`data_source/adela/adela_release_records.jsonl`（部署记录语料）

## Benchmark 简介

- 用途：评估 unified 检索入口在三类数据源上的检索与回答能力。
- 题型：文档事实问答 + 表格字段查询 + adela 部署记录查询。
- 追溯性：每道题都绑定来源文件、证据片段和来源类型，便于人工复核。

## 文件说明

- `benchmark_100.jsonl`：主评测集（推荐）
- `benchmark_100_for_eval.csv`：评测兼容格式
- `questions.txt`：仅问题列表，便于批量压测
- `BENCHMARK_AND_EVAL_METHOD.md`：Benchmark 设计与评测方法说明（指标定义、运行流程、bad case 分析）

## 构建脚本

统一构建脚本：

```bash
python scripts/build_unified_benchmark.py \
  --out-dir benchmark/single_step_rag_100_v1 \
  --doc-benchmark-jsonl /tmp/doc_pool_clean.jsonl \
  --doc-count 8 \
  --table-count 46 \
  --adela-count 46 \
  --total 100
```

可通过参数调节三源配比（`doc-count/table-count/adela-count`）。

## 覆盖统计（当前版本）

- 题目数量：100
- 来源分布：`document=8`、`table=33`、`adela=41`、`unified_cross=18`
- 升级后的跨源核对题：18（需同时命中 `table + adela`）

## 样本字段（JSONL）

- `id`：题目编号（`ssr100-001` ~ `ssr100-100`）
- `question`：问题文本
- `reference_answer`：标准答案
- `source_doc` / `source_page`：来源文件与页码/行号
- `evidence`：证据片段
- `question_type` / `difficulty` / `expected_keywords`：题型、难度与关键词
- `source_type`：来源类型（`document` / `table` / `adela`）
- `source_id`：来源主键（文档名/row_id）
- `retrieval_source_types`：建议检索源

## 快速使用（unified 接口）

```bash
python sample_code/unified_chat_benchmark.py \
  --queries-file benchmark/single_step_rag_100_v1/questions.txt \
  --repeat 1 \
  --route-with-llm
```
