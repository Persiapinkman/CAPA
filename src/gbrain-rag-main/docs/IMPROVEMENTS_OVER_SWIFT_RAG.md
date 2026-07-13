# 相对 swift-rag 的改进说明

最后更新：2026-05-09

## 背景

原 `../swift-rag` 项目已经支持三类数据源：

- `document`：模型发版 PDF / ONES 文档正文，适合输入输出、阈值、优化点、追加数据等正文细节。
- `table`：模型发版记录汇总表，适合 OID、负责人、更新时间、推荐配置、支持设备等结构化字段。
- `adela`：Adela 部署记录，适合 did/rid、部署平台、部署状态、部署版本等部署信息。

swift-rag 的统一链路主要是“三源分别检索，再通过 RRF 合并”。当前项目在保留这三个数据源边界和 API 风格的基础上，引入开源 gbrain 的统一本地 brain 思路，把 chunk、embedding、实体、关系边和 FTS 放进同一个 SQLite 索引，并在检索和回答阶段加入更多确定性信号。

## 核心改进

### 1. 从多套检索产物收敛为统一 SQLite brain

swift-rag 中 document、table、adela 有各自的入库和检索产物，统一查询阶段再把结果合并。当前项目使用 `BrainStore` 将数据统一保存到 `data/index/gbrain.sqlite3`：

- `documents`：文档级元信息。
- `chunks`：统一 chunk 表，包含 `source_type`、`text`、`index_text`、`metadata_json`。
- `embeddings`：按 `chunk_id + model_name` 保存向量。
- `entities` / `chunk_entities` / `entity_links`：保存确定性实体和 chunk 内共现关系。
- `chunks_fts`：SQLite FTS5 全文索引。

这样 document/table/adela 不再只是最后合并的三条孤立链路，而是在同一个数据模型下共享实体、关键词、向量和证据 payload。

### 2. 引入实体图信号，补强短查询和跨源关联

当前项目在入库时通过规则抽取实体，包括：

- 模型文件名：`.model`、`.onnx`、`.pt` 等。
- 版本号、OID、did/rid、平台。
- 领域短语：输入、输出、阈值、推荐配置、检测/识别/分类/属性/项目/场景等。

同一 chunk 内实体会建立 `co_mentions` 共现边。检索时，候选 chunk 会计算 query 实体与 chunk 实体的 overlap 分数，并作为 graph signal 加入最终排序。这个机制对“某模型在 T4 平台是否部署”“某模型 OID 是什么”“发版文档和部署平台是否一致”这类实体驱动问题更稳定。

### 3. 结构化查询理解不再完全依赖 LLM

当前项目新增 `query_understanding.py`，在本地完成领域查询理解：

- 本地只保留意图、字段、平台、版本、英文 token 等通用解析；显式扩展词可通过 `query_expansion_terms` 传入。
- 默认启用轻量 LLM query expansion：服务会在检索前生成少量候选检索词，并经过通用噪声过滤；请求中设置 `expand_query_with_llm=false` 可关闭。
- 识别 OID、部署、推荐、统计、列表、最新信息等意图。
- 对 table/adela 的结构化行按字段打分，而不只依赖全文 BM25 或向量相似度。
- 对平台字段做边界匹配，避免 `T4` 错误命中 `P4`。

这降低了小模型或弱 LLM 在路由和统计规划上的不确定性，也使部分结构化问题可以不调用 LLM 直接回答。
LLM query expansion 会增加一次 LLM 调用，因此评测和线上观测需要同时看召回收益与延迟成本；如需做控制组实验，可通过环境变量或请求字段关闭。

### 4. 检索融合信号更多

swift-rag 统一检索主要围绕向量、BM25/关键词和 RRF。当前项目的 hybrid 检索融合以下信号：

- vector：基于 `bge_m3` 和可选 `EvoQwen2.5-VL-Retriever-3B-v1` 的向量召回。
- keyword：SQLite FTS5 + Python BM25，兼顾中文、模型名和字段文本。
- structured：对 table/adela 元数据字段做确定性打分。
- graph：实体 overlap 和共现图信号。
- lexical boost：对字段类型和文档结构做轻量补强，例如特征维度、模型文件列表等，不作为业务答案词表使用。
- source-balanced 排序：显式多源检索时优先保证 document/table/adela 在前排证据中都有代表性结果，再做全局排序，降低单源分数尺度挤占上下文的风险。

最终融合仍使用 RRF，但不再只比较单一检索器的原始分。

### 5. PDF 表格被转为“可回答”的结构化文本

传统 PDF 抽取容易把表格列错位，LLM 看到长 snippet 时容易把模型名、平台、特征维度绑定错。当前项目在 `ingest/pdf.py` 中对 PDF 表格做两层表达：

- Markdown 表格：保留可读形态。
- `表格结构化行`：把模型族、模型名称、组件类型、OID、平台、特征维度按行绑定。

检索证据进入 LLM 前还会生成 `payload.field_summary`，用紧凑格式表达“主体 -> 字段 -> 值”。问答 prompt 明确要求字段值问题优先读取 `field_summary`，并在回答后做已知字段漏值补齐。

### 6. 结构化统计和字段查询更可控

对于“RD 部门总共有多少个模型”“部署了多少”“某类模型有哪些”等问题，当前项目有两条路径：

- 本地结构化回答：对常见 OID、部署、推荐、统计问题直接读取 SQLite metadata，去重后回答。
- LLM 结构化统计规划：当问题需要全量统计且本地规则不足时，让 LLM 只输出 JSON 计划，再由代码执行计数，最后生成答案。

关键数字由代码计算，LLM 不直接编造统计结果。
统计行读取优先使用规范化 JSONL 源，避免同一结构化语料被 CSV/XLSX/JSONL 重复导入后污染聚合口径。对于显式三源请求，结构化捷径会补充缺失 source 的支撑证据，让 route/evidence 和请求语义保持一致。

### 7. 答案阶段保留证据中的关键原值

当前项目在 evidence formatting 中加入 `important_values`，从 snippet、`payload.index_text`、`payload.field_summary` 中提取：

- 原始英文 label。
- 百分比和精度指标。
- 版本、模型、平台等关键字符串。

回答 release note、优化点、指标、标签、版本变化等问题时，prompt 要求优先保留这些原值；如果答案漏掉证据里的关键指标，会做基于证据的补全。这个机制不生成新事实，只减少 LLM 把数字、label 或版本概括丢失的问题。

### 8. 降级策略更稳

当前项目支持以下降级：

- embedding 设备默认 CUDA，CUDA 不可用自动回退 CPU。
- sentence-transformers 加载失败且允许 fallback 时，可回退 hashing embedding。
- hybrid 检索中单个向量通道失败时跳过该通道，继续使用 keyword/structured/graph。
- LLM 不可用时返回可核对证据片段，而不是整个接口失败。

这使服务更适合本地 GPU 资源不稳定或索引尚未完整重建的场景。

## 当前评测观察

在 `benchmark/hard-cases` 的 20 道真实语料 hard cases 上，使用 `recall@12` 和 LLM judge 的一次对比结果如下：

| 配置 | client avg | retrieve avg | golden recall @12 | answer ok | answer score |
| --- | ---: | ---: | ---: | ---: | ---: |
| 本地通用解析，关闭 LLM expansion | 8502 ms | 2029 ms | 0.6333 | 0.60 | 0.6225 |
| 本地通用解析，开启过滤后的 LLM expansion | 8829 ms | 2273 ms | 0.6500 | 0.65 | 0.6625 |

结论：

- LLM query expansion 能提升 answer ok 和 answer score，但会增加延迟。
- 当前默认策略是开启 LLM query expansion，并保留 `expand_query_with_llm=false` / `GBRAIN_RAG_ENABLE_LLM_QUERY_EXPANSION=false` 作为控制组开关。
- 后续优化应继续使用留出集或线上 A/B 验证，避免把单轮 benchmark 的问题特征写进规则。

## 不变的部分

当前项目没有改变 swift-rag 的核心语料边界：

- `document` 仍表示模型发版正文。
- `table` 仍表示模型发版汇总表。
- `adela` 仍表示部署记录。
- `/api/v1/rag/chat_engine/unified_retrieve` 和 `/api/v1/rag/chat_engine/unified_query` 仍保留统一接口习惯。

因此本项目更准确地说是“在 swift-rag 语料和接口基础上，引入 gbrain 式本地 brain 和结构化检索增强”，而不是另起一套语料体系。
