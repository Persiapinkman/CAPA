# gbrain 方法简述

最后更新：2026-05-06

## gbrain 的核心思想

开源 `garrytan/gbrain` 项目强调给 AI agent 一个长期记忆层：用可读的 brain repo 作为事实来源，用 Postgres/pgvector 或 PGLite 做检索层，并通过 MCP/CLI 暴露给 agent 使用。与传统“只把文档切块后做向量检索”的 RAG 相比，它更关注以下几点：

- 可读事实源：markdown brain repo 是 system of record，人可以直接查看和修改。
- 本地/轻量持久化：数据库可用 PGLite 快速启动，也可使用 Postgres + pgvector 承载检索。
- 结构化记忆：不只保存原文 chunk，也保存实体、关系和元数据。
- 自动链接：写入页面时抽取实体关系，形成 typed links 和 backlinks。
- 混合检索：语义向量、关键词、链接/图信号一起参与召回与排序。
- 低耦合实现：用轻量数据库和确定性规则完成一部分“记忆组织”，避免所有步骤都交给 LLM。

本项目没有直接复制 gbrain 的全部代码，而是借鉴其方法论，将其落地到 RD 模型发版知识库场景。

## 本项目对 gbrain 思路的落地

### 1. SQLite 作为 local brain

本项目使用 `BrainStore` 将所有检索资产保存到 SQLite：

- chunk 原文和索引文本。
- 多模型 embedding。
- FTS 全文索引。
- 规则抽取实体。
- chunk 与实体关联。
- 实体共现关系边。

这对应 gbrain 的“持久化 brain + 检索层”思想。区别是：上游 gbrain 以 markdown repo + Postgres/pgvector/PGLite 为主，本项目为了部署简单和复用现有 Python 栈，使用 SQLite 保存全部索引资产。

### 2. 实体和关系自动构建

入库时，系统不调用 LLM，而是用确定性规则从 RD 模型发版语料中抽取实体：

- model：模型文件名。
- version：版本号。
- oid：模型 OID。
- deployment：did/rid。
- platform：部署平台。
- field/scene：输入、输出、阈值、检测/识别/分类等领域短语。

同一 chunk 内出现的实体会建立 `co_mentions` 关系。这是本项目中的轻量知识图谱。

### 3. 图信号参与检索

查询阶段，系统也会从 query 中抽取实体，并与候选 chunk 的实体做 overlap。该分数作为 graph signal 参与最终排序。相比单纯向量召回，图信号更适合以下问题：

- 精确模型名、OID、did/rid 查询。
- 平台查询，例如 T4/P4/L4。
- 跨源核对，例如发版文档、汇总表和部署记录是否一致。
- 短查询或术语型查询。

### 4. 结构化元数据是一等公民

gbrain 方法强调“记忆不只是文本”。本项目的 table/adela 行不会只被压成普通文本，还会保留 metadata，并在检索时按字段计算 structured score。这样可以更可靠地回答：

- “OID 是多少？”
- “负责人是谁？”
- “T4 平台有没有部署？”
- “did/rid 是什么？”
- “总共有多少模型？”

### 5. LLM 负责表达，不负责凭空组织事实

本项目将事实组织尽量前置到索引和检索阶段：

- 数字统计由代码执行。
- 字段值从 metadata 或 `field_summary` 中抽取。
- 证据来源带 `source_type` 和 `[证据N]`。
- LLM prompt 限定只能依据 evidence 回答。

这与 gbrain 的理念一致：LLM/agent 是读取和维护 brain 的接口，而不是唯一的知识存储和事实来源。

## 与传统 RAG 的差异

传统 RAG 常见流程是：

```text
文档 -> chunk -> embedding -> 向量召回 -> LLM 回答
```

本项目采用的 gbrain-inspired 流程是：

```text
多源数据 -> chunk/row/table -> SQLite brain
                           -> embedding
                           -> FTS
                           -> metadata
                           -> entities
                           -> entity links

query -> source route -> query intent -> vector + keyword + structured + graph
      -> evidence payload / field_summary -> LLM 或结构化回答
```

主要差别是：知识组织不止发生在 embedding 中，也发生在 metadata、实体和关系层。

## 当前实现边界

当前项目采用的是轻量规则图谱，不是完整知识图谱系统：

- 实体抽取依赖正则和领域词，不做开放域实体识别。
- `entity_links` 当前主要表示 chunk 内共现，不表示复杂语义关系。
- 图信号用于检索增强，不做多跳图推理。
- 所有数据仍保存在单机 SQLite 中，适合本地和中小规模企业知识库。

这个边界是有意选择的：当前任务更需要稳定、可维护、可解释的 RD 资料问答，而不是构建复杂图数据库。

## 参考

- `garrytan/gbrain` README：https://github.com/garrytan/gbrain
- `garrytan/gbrain` 设计文档：https://github.com/garrytan/gbrain/blob/master/docs/GBRAIN_V0.md
