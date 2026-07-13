# Chunk 策略说明

本文档简要说明当前项目的 chunk 策略，包括不同 `input_type` 的切分路径、父子 chunk 关系，以及这些 chunk 在 embedding 和检索阶段如何使用。

## 总体思路

当前项目的核心思路是“父子两层”：

- 父级 chunk 通常记为 `chunk512`
- 子级 chunk 通常记为 `chunk128`
- 实际入库和向量化的主要对象是子级 chunk
- 子级 chunk 会通过 `index_id` 指回父级 chunk
- 检索命中子级 chunk 后，后处理阶段通常会回到父级 `index_text`

需要特别注意：

- `chunk512` / `chunk128` 是项目里的层级命名，不总是严格等于 512 / 128 token
- `raw` 路径更接近“按 token 控制大小”
- `autopdf`、`markdown`、`mineru` 路径里，很多地方更接近“按结构或字符长度控制大小”

## 当前使用方式

当前仓库里，实际在用的是 data source 离线入库这条链路：

1. 从 `data_source/` 目录收集源文件
2. 在 `src/rag/pipeline/chenggong_pipeline.py` 里构造 `ChunkingEmbeddingRequest`
3. 调用 `RAGService.chunking_embedding()`
4. 将子级 chunk 按 embedding 模型写入本地 Milvus Lite
5. 检索时先召回子级 chunk，再回到父级 `index_text`

当前默认配置可以概括为：

- 数据源目录：`data_source/`
- 本地向量库：`data_source/embedding_artifacts/documents/milvus_data_source_evoqwen_3b.db`
- collection：`llamacollection`
- 默认 embedding 模型：`EvoQwen2.5-VL-Retriever-3B-v1`

如果按当前仓库的默认方式重新离线入库，通常对应的是：

- PDF 文件走 `autopdf` 路径
- `.md` 文件走 `markdown` 路径
- `.txt` / `.jsonl` 走 `raw` 路径
- `.xlsx` 走 `json_list` 路径
- `.json` 根据内容自动判定为 `json_list` 或 `autopdf`

一个常见的理解方式是：

- 建库阶段：用子级 chunk 做 embedding
- 检索阶段：先命中子级 chunk
- 回答阶段：把命中的子级 chunk 还原到父级 `index_text` 再喂给大模型

## 当前结果

基于当前仓库里的本地数据，现状如下：

- `data_source/PDFs` 当前包含 `75` 份 PDF
- 当前有效文档入库链路已切换为 `autopdf`
- `results/data_source_embedding_report__evoqwen2_5_vl_retriever_3b_v1__bge_m3.csv` 覆盖 `75` 份 PDF、`150` 条模型级处理记录，成功 `150`、失败 `0`
- `results/data_source_embedding_report__evoqwen2_5_vl_retriever_7b_v1.csv` 覆盖 `75` 份 PDF，但当前 `75` 条均失败，主要原因是 CUDA OOM
- 成功入库后的 `EvoQwen2.5-VL-Retriever-3B-v1` 和 `bge_m3` 两套库均为 `1909` 条记录，覆盖同一批 `75` 份 PDF

当前主要向量库为：

- `data_source/embedding_artifacts/documents/milvus_data_source_evoqwen_3b.db`
- `data_source/embedding_artifacts/documents/milvus_data_source_bge.db`

当前配置中，文档检索可通过 `vector_store_configs` 按 embedding 模型路由到对应向量库，避免不同维度模型混写。

单模型写入关系为：

- embedding 后实际入库条数 = 原始子级 chunk 数
- 每个原始子级 chunk 只会写入一条记录

如果显式启用多模型，当前推荐按模型分库写入；若写入同一个兼容维度的库，则：

- 总条数 = 原始 chunk 数 × embedding 模型数

可以直接用下面的命令复查当前结果：

```bash
# EvoQwen-3B 总入库条数
sqlite3 data_source/embedding_artifacts/documents/milvus_data_source_evoqwen_3b.db "select count(*) from llamacollection;"

# BGE 总入库条数
sqlite3 data_source/embedding_artifacts/documents/milvus_data_source_bge.db "select count(*) from llamacollection;"

# EvoQwen-3B 原始子级 chunk 数
sqlite3 data_source/embedding_artifacts/documents/milvus_data_source_evoqwen_3b.db "select count(distinct substr(milvus_id, 1, instr(milvus_id, '_EMB_') - 1)) from llamacollection;"

# 每个 embedding 模型的条数（同库多模型时才有对比意义）
sqlite3 data_source/embedding_artifacts/documents/milvus_data_source_evoqwen_3b.db "select substr(milvus_id, instr(milvus_id, '_EMB_') + 5) as embedding_model, count(*) from llamacollection group by embedding_model order by embedding_model;"
```

如果只关心“现在有多少个 chunk”，应该看原始子级 chunk 数。

## 各输入类型的切分方式

### 1. `raw`

`raw` 文本走标准的两层切分：

1. 先用 `HierarchicalParser(chunk_sizes=[128, 512])`
2. 底层实际使用 `ZHSentenceSplitter`
3. 先切父级 `chunk512`
4. 再在每个父级块内部切子级 `chunk128`
5. 最终只保留子级块进入 embedding

特点：

- `ZHSentenceSplitter` 以 tokenizer 计算长度，属于 token 口径
- 优先按段落、句号、分号、逗号等边界切分
- `chunk_overlap=0`，默认没有重叠
- 子级 chunk 的 `metadata["index_text"]` 会保存父级全文
- 检索命中后可以回到父级文本

ID 形式：

- 父级：`doc_{doc_id}_chunk512_{i}`
- 子级：`doc_{doc_id}_chunk512_{i}_chunk128_{j}`

## 2. `autopdf`

`autopdf` 不是直接按 token 切，而是先按文档结构聚合，再做子切分。

流程大致如下：

1. 从 AutoPDF 输出的 JSON 中抽取内容
2. 按 `header` / `heading` / `page_id` 生成 `group_key`
3. 把同组文本聚合成父级块 `all_text`
4. 如果父块太短：
   - 小于 40 字时优先向后合并
   - 如果后面不合适，再向前合并
5. 如果父块过长：
   - 仅当 `all_text_len > 6000` 时触发二次切分
   - 切成不超过 1024 字符的小段
6. 这些父级块被命名为 `chunk512`
7. 然后再用 `RecursiveCharacterTextSplitter2` 切成子块

特点：

- 父级块首先是“结构块”，不是严格 512 token
- 超长父块会被额外切到约 1024 字符以内
- 子级切分是字符和规则驱动，不是严格 token 切分
- 子级文本会额外拼上：
  - `header`
  - `heading`
- 子级 `metadata["index_text"]` 保存父级全文

`RecursiveCharacterTextSplitter2` 的行为：

- 默认先按双换行切
- 再尝试按中文编号、小节标题、括号编号等模式切
- 对超过 `force_chunk=300` 的内容继续硬切
- 长度小于 `chunk_size=128` 的碎片会向前合并

所以这里的 `chunk128` 更准确地说是：

- 一个“目标较小的子块”
- 常见上限接近 300 字符
- 过小片段会被合并
- 不是严格 128 token

## 3. `markdown`

`markdown` 走“结构切分 + 子切分”：

1. 先使用 `MarkdownNodeParser` 按 Markdown 结构切出父级块
2. 父级块统一命名为 `doc_{doc_id}_chunk512_{i}`
3. 再调用 `create_small_nodes()`
4. 子切分仍使用 `RecursiveCharacterTextSplitter2`

特点：

- 父级块由 Markdown 结构决定，不是严格 512 token
- 子级块规则与 `autopdf` 一致
- 子级文本会拼接 `header` / `heading` 信息
- `metadata["index_text"]` 仍保存父级文本

## 4. `json_list`

`json_list` 最简单：

- 输入会被解析成列表
- 列表里的每个字典直接变成一个 chunk
- 每个元素既是一个子块，也是它自己的父块

特点：

- 不做额外分段
- `index_id` 直接等于当前 `id`
- `index_text` 就是该 JSON 对象本身的字符串

ID 形式：

- `doc_{doc_id}_chunk512_0_chunk128_{idx}`

这里的 `chunk512` / `chunk128` 只是为了兼容统一命名，并不表示真的经过两层切分。

## 5. `mineru`

`mineru` 路径有自己的一套预处理规则，结构最复杂。

主要流程：

1. 连续短正文先合并
2. 超长正文先拆分，默认阈值约为 6000 字
3. 跨页短文本可合并
4. 图片 caption / footnote 会补进文本
5. 根据标题层级生成 `group_text`
6. 短 group 会继续合并，阈值约 128 字
7. 如果整篇文档很短，总字数不超过 2000，可整体合并
8. 超长 group 再按 6000 字做拆分
9. 表格会单独清洗并拼接 caption / footnote / body
10. 最后再把正文切成不超过 128 字的子块

格式化为 LlamaIndex 节点时：

- `index_text` 使用 group 级文本
- `index_id` 形如 `doc_{doc_id}_chunk512_{group_id}`
- `id` 形如 `doc_{doc_id}_chunk512_{group_id}_chunk128_{chunk_id}`
- embedding 实际使用的是 `embedding_text`

`embedding_text` 会拼接：

- `doc_name`
- `header`
- `heading`
- 当前子块正文

因此 `mineru` 路径下，向量召回更偏向“带结构上下文的子块”。

## 6. `pdf_blocks`（历史方案：block 级多模态）

`pdf_blocks` 是此前 EvoQwen2.5-VL-Retriever-3B-v1 用于 PDF 的链路，目标是让 PDF 中的文字、表格、图片都能进入同一套检索流程。

整体流程：

1. 使用 `pypdf` 逐页读取 PDF。
2. 对每页文本按行做轻量规则拆分：
   - 命中 `|`、`\t`、或“多空格列对齐”模式的行判为 `table`；
   - 其余非空行判为 `text`。
3. 从 `page.images` 提取图片并落盘到 `.cache/pdf_blocks/...`，生成 `image` block。
4. 形成统一 block 列表并作为 `ChunkingEmbeddingRequest.text`（JSON list）传入，`input_type=pdf_blocks`。
5. `chunk_into_small_nodes()` 直接把每个 block 转成一个 `IndexNode`（不再走 512/128 二层切分）。
6. embedding 阶段按 block 类型分流：
   - `text/table`：走文本 embedding；
   - `image`：走 image embedding（模型不支持时回退为文本 embedding）。

每个 block 是怎么得到的（对应你关心的“方案 4 每个 block 来源”）：

- `text block`：来自当前页中“非表格行”连续片段合并。
- `table block`：来自当前页中“表格样式行”连续片段合并（启发式，不是版面级精确表格识别）。
- `image block`：来自当前页可提取图片对象，每张图一个 block，并写入 `image_path` 供 image embedding 使用。

此前 `pdf_blocks` 的 trunk/chunk 划分可以理解为：

- 以 “page 内 block” 作为最小检索单元；
- 不再额外拆出 `chunk512/chunk128` 子层级；
- 每个 block 通过 `content_type` 标记模态（`text`/`table`/`image`）。

局限与注意：

- 表格识别是行级启发式，复杂版面可能误判。
- 图片块默认不做 OCR，主要依赖视觉 embedding。
- 若页面文本为空且图片无有效提取结果，该页可能不会产出 block。

## embedding 时真正写入的是什么

无论哪种输入类型，最终向量化的主要都是子级 chunk，而不是父级 chunk。

项目中会把父级内容放到子级节点的 `metadata["index_text"]` 里。后续入库时，每个子级 chunk 会按每个 embedding 模型各写一条记录。

默认配置（单模型 `EvoQwen2.5-VL-Retriever-3B-v1`）下：

- 原始子级 chunk 数量为 `N`
- 实际 Milvus 记录数为 `N`

若同时启用两个 embedding 模型：

- 实际 Milvus 记录数为 `N x 2`

记录 ID 形式类似：

- `doc_xxx_chunk512_3_chunk128_2_EMB_EvoQwen2.5-VL-Retriever-3B-v1`
- `doc_xxx_chunk512_3_chunk128_2_EMB_bge_m3`（多模型时）

## 检索时为什么会回到父级文本

检索命中的是子级 chunk，但后处理时会按 `index_id` 去重，并把返回节点的文本替换为父级 `index_text`。

这样做的目的有两个：

- embedding 时保留小块粒度，提升召回精度
- 回答时回到父级块，给大模型更多完整上下文

因此可以把当前策略理解成：

- 用小块召回
- 用父块喂给大模型

## 一句话总结

当前项目不是单一的固定 chunk 策略，而是“按输入类型选择切分器”的混合方案：

- `raw`：两层 token 切分
- `autopdf`：按文档结构聚合，再按字符规则切小块
- `markdown`：按 Markdown 结构聚合，再按字符规则切小块
- `json_list`：每个元素一个 chunk
- `mineru`：先做较重的版面/结构清洗，再形成父子块
- 历史 `pdf_blocks`：按 PDF page block 切分，并按模态路由 embedding

如果后续需要统一策略，最值得优先明确的是：

- `chunk512` / `chunk128` 是否要改成真实大小口径
- 各路径是否统一为 token 切分
- `index_text` 是否继续保留父级全文，还是改成更短的父级摘要
