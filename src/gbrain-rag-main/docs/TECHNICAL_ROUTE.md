# 技术路线说明

最后更新：2026-05-06

本文说明当前项目从数据源处理、embedding、检索到回答生成的端到端链路。对应代码以 `src/gbrain_rag` 和 `scripts/build_index.py` 为准。

## 1. 数据源处理

### 1.1 数据源类型

项目默认读取 `data_source/`，支持后缀：

- `.pdf`
- `.md`
- `.txt`
- `.json`
- `.jsonl`
- `.csv`
- `.xlsx`

`ingest/loaders.py` 会根据路径和文件名推断 `source_type`：

- `adela_release_records` -> `adela`
- `model_release_records`、`.csv`、`.xlsx` -> `table`
- 其他文件 -> `document`

`adela/data/` 下的原始 JSON 会被跳过，因为 swift-rag 已经将其标准化为 `adela_release_records.jsonl`。这样可以避免重复索引噪声数据。

### 1.2 文档类 PDF

PDF 处理入口是 `ingest/pdf.py`：

- 优先使用 `pdfplumber` 抽取文本和表格。
- 使用 PyMuPDF block 文本作为补充。
- 最后可回退到 `pypdf` 文本抽取。
- 普通正文按页和 chunk 大小切分。
- 表格同时转换为 Markdown 表格和 `表格结构化行`。

其中 `表格结构化行` 用于保存字段绑定关系，例如：

```text
模型族: Base224
模型名称: ...
组件类型: text_encoder
OID: ...
平台: cuda11.0-trt7.1-fp16-T4
特征维度: 512
```

这种表示可以降低 PDF 表格换行错位对检索和回答的影响。

### 1.3 结构化表格与 JSONL

`table` 和 `adela` 会以“行”为基本 chunk：

- `model_release_records.jsonl`：读取模型发版表的标准化行。
- `adela_release_records.jsonl`：读取 Adela 部署记录。
- `.xlsx`：通过 pandas 读取每个 sheet，每一行生成一个 chunk。
- `.csv`：通过 `csv.DictReader` 逐行生成 chunk。

每行 chunk 的 `metadata` 保留原始字段，`text` 使用可检索字段拼接，`index_text` 保存更完整的行内容。

### 1.4 Chunk 生成

chunk 统一使用 `Chunk` 数据结构：

- `chunk_id`：由 doc、source、block、页码、序号和文本前缀稳定生成。
- `doc_id`：由相对路径、文件大小和 mtime 生成。
- `source_type`：`document` / `table` / `adela`。
- `text`：用于展示和回答的文本。
- `index_text`：更完整的索引文本。
- `metadata`：结构化字段或文件信息。

正文默认配置：

- `CHUNK_SIZE=900`
- `CHUNK_OVERLAP=120`

## 2. Embedding 过程

### 2.1 构建入口

离线索引构建命令：

```bash
PYTHONPATH=src python scripts/build_index.py \
  --reset \
  --embedding-model bge_m3 \
  --embedding-backend sentence-transformers \
  --embedding-device cuda
```

`scripts/build_index.py` 的流程：

1. 遍历 `data_source/` 支持的文件。
2. 调用 `load_file()` 生成 chunk。
3. 将 chunk 拼成 embedding 输入：`doc_name + title + text + index_text`。
4. 每批调用 embedding backend。
5. 将 chunk、embedding、FTS、实体和共现关系写入 SQLite。

### 2.2 Embedding backend

当前支持两种 backend：

- `sentence-transformers`：默认用于真实向量检索。
- `hashing`：轻量 fallback 或 smoke test。

默认配置：

- `EMBEDDING_MODEL=bge_m3`
- `EMBEDDING_MODELS=["bge_m3", "EvoQwen2.5-VL-Retriever-3B-v1"]`
- `EMBEDDING_DEVICE=cuda`

当请求 `cuda` 时，系统会检查可用 GPU，并选择空闲显存最多的设备；CUDA 不可用时回退 CPU。向量会归一化后存入 SQLite BLOB。

### 2.3 SQLite brain 存储

索引默认保存在：

```text
data/index/gbrain.sqlite3
```

核心表：

- `documents`：文档元信息。
- `chunks`：所有 source 的统一 chunk。
- `embeddings`：chunk 向量，按模型区分。
- `entities`：规则抽取的实体。
- `chunk_entities`：chunk 与实体的关联。
- `entity_links`：chunk 内实体共现边。
- `chunks_fts`：SQLite FTS5 全文索引。

## 3. 检索过程

### 3.1 数据源路由

`RetrievalService.route_sources()` 根据问题关键词选择 source：

- 输入、输出、阈值、优化、标签、发版文档、特征维度 -> `document`
- OID、负责人、更新时间、推荐配置、统计、清单 -> `table`
- 部署、上线、did、rid、平台、状态、推荐等 -> `adela`
- 对比、一致、差异、核对、关联 -> 多源联合

请求也可以通过 `sources` 手动指定数据源。

### 3.2 查询理解与扩展

`query_understanding.py` 会构造 `QueryIntent`：

- 保留原 query，并抽取英文、数字、版本、平台等通用 token。
- 抽取平台词，例如 T4/P4/L4/ascend/cuda。
- 判断是否问 OID、部署、推荐、统计、列表、最新信息。
- 为 table/adela 准备字段级打分。

默认还会在检索前调用轻量 LLM query expansion，生成少量可能出现在语料中的检索词，并过滤泛化类别词和 query 中已经存在的词。请求可通过 `expand_query_with_llm=false` 关闭，或通过 `query_expansion_terms` 传入外部明确给定的补充词。

### 3.3 多路召回

`RetrievalService.retrieve()` 按 `retrieval_method` 执行：

- `keyword` / `bm25`：SQLite FTS5 + Python BM25。
- `vector`：加载 SQLite 中对应模型的 embedding matrix，对 query embedding 做内积排序。
- `hybrid`：同时使用 keyword、vector、structured、graph 并融合。

document 默认可以使用多 embedding 模型；table/adela 默认使用主 embedding 模型。

### 3.4 结构化信号

对 `table` 和 `adela`，系统会读取 chunk metadata，并按字段打分：

- 目标名称、算法名称、模型名称。
- OID、负责人、推荐配置、支持设备。
- platform、did、rid、status、label_list。

这条 structured signal 会单独排序，并以权重加入 hybrid 融合。

### 3.5 实体图信号

入库时抽取实体并建立共现边。检索时：

1. 从 query 抽取实体。
2. 对 vector/keyword 候选加载 chunk entities。
3. 计算 overlap score。
4. 将 graph score 加入最终分数。

这对短查询、精确模型名、平台/OID/did/rid 查询有明显帮助。

### 3.6 证据构造

检索结果会被转换成 EvidenceItem：

- `source_type`
- `score`
- `title`
- `snippet`
- `doc_name`
- `page_label`
- `source_path`
- `metadata`
- `matched_entities`
- `retrieval_signals`
- `payload`

字段值问题会额外生成 `payload.field_summary`，把表格行中的主体和值绑定起来，供 LLM 优先读取。

## 4. 回答过程

### 4.1 Unified API

主要接口：

- `POST /api/v1/rag/chat_engine/unified_retrieve`
- `POST /api/v1/rag/chat_engine/unified_query`
- `POST /api/v1/rag/retrieve`
- `POST /api/v1/rag/query`

`unified_retrieve` 返回证据；`unified_query` 在证据基础上调用 LLM 生成答案。
`unified_query` 和 `/rag/query` 支持 `stream=true`，返回 `text/event-stream`：
增量帧格式为 `data: {"content": "..."}`，结束前会返回包含完整 `answer`、`evidences`、`timings` 的 JSON 帧，最后一帧为 `data: [DONE]`。

### 4.2 结构化聚合短路

当问题是统计类问题时，例如“总共有多少个模型”“部署了多少”，系统优先尝试结构化聚合：

- 常见问题由本地规则直接统计。
- 更复杂问题由 LLM 生成 JSON 统计计划。
- 代码根据计划过滤 rows、去重、计数。
- LLM 只负责把执行结果组织成中文答案。

统计数字不由 LLM 自行推断。

### 4.3 LLM 问答

普通问题进入 `answer_with_llm()`：

- 使用 OpenAI-compatible 接口。
- prompt 明确只能依据 evidence 回答。
- 要求中文回答并用 `[证据N]` 标注。
- 字段值问题必须优先读取 `payload.field_summary`。
- 设置 `enable_thinking=false`，并清理可能出现的 thinking 内容。

如果 LLM 不可用，系统返回检索证据片段作为降级答案。

答案生成后，系统会让 LLM 对 `knowledge_base_fully_answered` 输出 `0.0`-`1.0` 的证据充分性置信度，供下游服务直接使用；LLM 评分不可用时才使用本地兜底分。

### 4.4 字段值补齐

LLM 回答后，`_complete_known_field_answer()` 会检查已检索证据中的 `field_summary`。如果问题明确询问特征维度、OID、平台、负责人、did/rid 等字段，而 LLM 漏掉了同一证据中的已知值，系统会追加“补充”信息。

这一步只使用已检索证据，不引入新知识。

## 5. 运行与验证

构建索引：

```bash
PYTHONPATH=src python scripts/build_index.py --reset
```

启动服务：

```bash
PYTHONPATH=src python -m uvicorn gbrain_rag.main:app --host 0.0.0.0 --port 6061
```

健康检查：

```bash
curl --noproxy '*' http://127.0.0.1:6061/api/v1/rag/health
```

单元测试：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
