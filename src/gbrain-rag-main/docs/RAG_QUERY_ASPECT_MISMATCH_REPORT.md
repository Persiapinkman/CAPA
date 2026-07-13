# RAG 查询方面错配问题分析报告

最后更新：2026-05-12

## 1. 背景

`sample_code/unified_query_api_client.py` 默认向统一问答接口发送如下请求：

```json
{
  "query": "安全绳检测模型的精度如何",
  "top_k": 8,
  "retrieval_method": "hybrid"
}
```

期望答案应围绕安全绳检测模型的精度测试结果、指标数值或评测结论展开。但实际回答容易偏向“哪些情况下精度无法保证”的边界说明。

该现象不是 client 侧 query 传错，也不应被理解为某个具体模型或某个关键词的孤立 bug。它暴露的是当前 RAG 链路对“用户询问方面”的表达不足：系统能检索到与实体相关的材料，但不能稳定区分“精度指标”“算法边界”“输入输出”“模型文件”“部署信息”等不同回答面。

## 2. 现象

对问题“安全绳检测模型的精度如何”，知识库中实际存在两类相关证据：

1. 能直接回答精度问题的证据：
   - Release Note 中包含 `top1 acc 从 0.55 提升到 0.97`。
   - 通用测试集上各项精度指标有 `0.1%-0.5%` 提升。
   - 精度测试表包含 `Acc/mPrec/mA` 以及各类别的 `prec/recall/F1`。

2. 与精度相关但不直接回答“精度如何”的证据：
   - 算法边界中描述夜间、强逆光、雨雪雾等场景下性能暂时无法保证。
   - 目标过小、模糊、遮挡、截断、多人、角度或姿态影响判断时，精度无法保证。
   - 安全绳颜色或款式难以辨别、红色之外安全绳等情况，精度无法保证。

第二类证据包含“安全绳”和“精度”等词，因此在混合检索中具备较高相关性。但它回答的是“适用边界/失败条件”，不是“精度指标/评测表现”。当这类证据排在前列或在 prompt 中缺乏证据分层时，LLM 容易把边界说明作为主答案。

## 3. 当前链路观察

### 3.1 Query 没有形成稳定的方面理解

当前 `build_query_intent()` 主要识别结构化字段、部署、OID、统计、推荐、列表等意图。对“精度如何”这类问题，系统没有显式形成类似如下 query frame：

```json
{
  "entity": "安全绳检测模型",
  "aspect": "accuracy_metric",
  "answer_type": "evaluation_summary",
  "constraints": {}
}
```

结果是检索阶段只能依靠原始 query、别名扩展词、向量相似度、BM25、实体 overlap 等信号。它知道“安全绳”相关，但不知道用户真正要的是“评测指标”，不是“边界条件”。

### 3.2 文档 chunk 缺少可回答方面标签

PDF 文档被切成普通正文 chunk、表格 chunk 和结构化行 chunk，但 chunk metadata 中没有统一标注该 chunk 属于哪种内容，例如：

- `release_note`
- `accuracy_eval`
- `performance_eval`
- `algorithm_boundary`
- `input_output`
- `model_artifact`
- `deployment`

因此，同一个模型文档下的“精度测试表”和“精度无法保证边界”在检索系统里都只是包含相近词汇的普通文本。排序器无法基于内容类型判断谁更适合作为主证据。

### 3.3 混合检索解决相关性，不解决可回答性

当前 hybrid retrieval 融合了向量、关键词、图实体和结构化信号。这些信号能提升“相关材料”的召回，但没有明确衡量“这段证据是否直接回答当前问题”。

对本例而言，以下两类 chunk 都相关：

- 包含 `top1 acc`、`Acc/mPrec/mA` 的评测 chunk。
- 包含 `精度无法保证` 的边界 chunk。

但只有前者是主答案证据，后者应作为补充限制条件。当前排序缺少这层区分。

### 3.4 Prompt 中证据平铺，缺少主次约束

`UNIFIED_QA_PROMPT` 会把 evidence 统一平铺给 LLM，并要求回答 release note、优化点、精度、指标、标签、版本变化等问题时优先查看 `important_values` 和 `payload.index_text`。

这个约束能提醒模型保留数字和指标，但没有告诉模型：

- 哪些 evidence 是 primary evidence。
- 哪些 evidence 只是 caveat 或背景。
- 当主证据和边界证据同时出现时，应先回答指标，再补充边界。

因此，即使检索结果中存在正确精度指标，LLM 仍可能被更易组织成自然语言的“无法保证”段落带偏。

## 4. 根因归纳

根因不是“没有召回正确材料”，而是“召回材料没有按问题方面和可回答性分层”。

具体表现为：

1. Query 表达不足：系统缺少 `entity + aspect + answer_type` 级别的查询理解。
2. Chunk 表达不足：文档片段缺少 section/aspect 类型标签。
3. Ranking 目标不足：排序主要优化相关性，没有显式优化直接可回答性。
4. Evidence 组织不足：传给 LLM 的证据缺少 primary/supporting/caveat 分层。
5. 评测覆盖不足：如果只看最终答案文本，容易忽略“同实体不同方面错配”的检索失败模式。

## 5. 不建议采用的修复方式

不建议为本问题直接增加如下特例：

- 遇到“安全绳”就强行提升 `safety_rope` 文档。
- 遇到“精度”就硬编码提升某几个中文标题。
- 遇到“无法保证”就一律降权。
- 在 prompt 中只针对“安全绳检测模型的精度如何”写规则。

这些做法短期可能修复单个样例，但会带来新的问题：

- 用户问“哪些情况下精度无法保证”时，`无法保证` 反而是正确主证据。
- 其他模型也存在同样的“指标 vs 边界”错配问题，安全绳特例无法覆盖。
- 不同文档标题、PDF 抽取质量和表格格式不一致，硬编码标题容易失效。
- 系统会逐渐堆积领域补丁，降低可维护性。

更合理的方向是把“方面”和“可回答性”作为一等信号，而不是修某个词。

## 6. 改进方案

### 6.1 引入 Query Frame

在检索前将用户问题解析为结构化 query frame。

建议字段：

```json
{
  "entity_mentions": ["安全绳检测模型"],
  "normalized_entities": ["safety_rope"],
  "aspect": "accuracy_metric",
  "answer_type": "evaluation_summary",
  "constraints": {
    "version": null,
    "platform": null
  }
}
```

`aspect` 应覆盖常见问答面：

- `accuracy_metric`：精度、准确率、召回、F1、mAP、评测指标。
- `limitation`：边界、限制、无法保证、适用条件。
- `input_output`：输入、输出、标签。
- `model_artifact`：模型文件、OID、配置、平台。
- `deployment`：Adela、did、rid、部署状态。
- `release_change`：版本变化、优化点、追加数据。
- `owner_metadata`：负责人、更新时间等结构化字段。

Query frame 可以先用本地规则实现，再逐步引入 LLM 或小模型增强。关键是后续检索、排序和 prompt 都消费同一个 frame。

### 6.2 给 Chunk 增加 Section/Aspect Metadata

索引阶段对文档 chunk 增加内容类型标注，例如：

```json
{
  "section_type": "accuracy_eval",
  "aspects": ["accuracy_metric"],
  "doc_entity": ["safety_rope"],
  "section_title": "五、精度测试"
}
```

可以从以下信号提取：

- 标题和邻近标题：`精度测试`、`性能测试`、`算法边界`、`模型文件列表`、`功能介绍`。
- 表格表头：`Acc/mPrec/mA`、`prec/recall/F1`、`耗时`、`OID`、`平台`。
- 内容模式：指标名、百分比、模型文件名、did/rid、输入输出描述。

这不是为了某个模型写规则，而是为所有文档建立统一的“片段能回答什么”的元数据。

### 6.3 改为 Entity-Then-Aspect 两阶段检索

建议把检索拆为两层：

1. 实体召回：先找到与目标实体强相关的文档、表格行或部署记录。
2. 方面排序：在实体候选内部，根据 query frame 的 `aspect` 选择最能直接回答的 chunk。

这样可以避免全库中相似词污染，也能避免同一实体下的不同方面互相抢答。

例如：

- `aspect=accuracy_metric`：优先 `accuracy_eval`、`release_note` 中的指标片段。
- `aspect=limitation`：优先 `algorithm_boundary`、`input_constraint`。
- `aspect=model_artifact`：优先 `model_list`、结构化表格行。

### 6.4 增加 Answerability Reranker

在最终送入 LLM 前，对候选 evidence 增加可回答性评分。

评分维度建议包括：

- 实体匹配：是否与 query 的实体一致。
- 方面匹配：chunk 的 aspect 是否覆盖 query aspect。
- 答案密度：是否包含用户需要的字段、指标、数值或结论。
- 背景惩罚：是否只是边界、说明、上下文，而不是直接答案。
- 冲突识别：是否回答了相邻但不同的问题。

初期可用规则和 metadata 实现，后续可替换为 cross-encoder 或 LLM reranker。

### 6.5 Evidence 分层后再生成

传给 LLM 前，将证据分为：

- `primary_evidence`：直接回答当前问题。
- `supporting_evidence`：补充上下文或结构化字段。
- `caveat_evidence`：限制条件、边界、注意事项。

Prompt 不应只平铺 evidence，而应明确要求：

- 主答案必须来自 `primary_evidence`。
- `caveat_evidence` 只能作为补充说明。
- 如果没有 primary evidence，则回答证据不足，而不是用 caveat 代替主答案。

这能显著降低“相关但不回答”的证据带偏生成结果。

### 6.6 建立方面错配评测集

需要新增一组评测用例，覆盖同一实体下不同方面的问题。

示例类型：

- 某模型的精度如何？
- 某模型在哪些情况下精度无法保证？
- 某模型输入输出是什么？
- 某模型有哪些标签？
- 某模型有哪些模型文件和 OID？
- 某模型是否部署到某平台？
- 某模型相比上一版优化了什么？

评测不应只看最终答案，还应检查：

- `primary_evidence` 是否命中正确 section。
- 是否把 caveat 当成主答案。
- 是否遗漏关键数值。
- 是否跨实体误用证据。

## 7. 建议落地顺序

### 阶段一：观测和评测

1. 为检索结果输出 query frame、chunk section_type、aspect、answerability_score。
2. 建立方面错配评测集。
3. 加入 evidence hit 评测，而不只评估最终答案。

### 阶段二：索引增强

1. 在 PDF 解析和 chunk 生成阶段抽取 section title。
2. 为 chunk 增加 `section_type` 和 `aspects`。
3. 对表格 chunk 识别指标表、模型列表、性能表等类型。

### 阶段三：检索和排序改造

1. 实现 query frame。
2. 引入 aspect-aware rerank。
3. 增加 answerability rerank。
4. 输出 primary/supporting/caveat evidence。

### 阶段四：生成约束

1. 修改 prompt，使主答案严格基于 primary evidence。
2. 对 caveat evidence 只允许补充，不允许替代主答案。
3. 在无 primary evidence 时明确返回证据不足。

## 8. 预期收益

该方案解决的是一类通用 RAG 问题，而非单个样例：

- 同一实体下不同问题方面的证据不再互相抢答。
- 检索结果从“相关材料集合”升级为“可回答证据集合”。
- LLM 接收的上下文主次更清晰，降低被背景段落带偏的概率。
- 后续新增模型、文档和问题类型时，不需要持续堆积关键词特例。

## 9. 结论

“安全绳检测模型的精度如何”被回答成“精度无法保证的情况”，本质是查询方面错配问题：系统召回了同实体、相邻方面的证据，但没有识别哪个证据应作为主答案。

短期可以通过 prompt 或关键词 rerank 缓解，但长期应把 query frame、chunk aspect metadata、answerability reranker 和 evidence 分层纳入主链路。这样才能系统性解决“相关但不回答”的 RAG 失败模式。
