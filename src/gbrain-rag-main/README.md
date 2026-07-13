# GBrain RAG

参考 `garrytan/gbrain` 思路实现的本地 RAG 服务：SQLite 保存 chunk、embedding、实体和关系边；检索时融合 vector、keyword/BM25、实体图信号，再通过 OpenAI-compatible LLM 生成带证据的回答。

## 当前状态

- 语料已从 `../swift-rag/data_source` 复制到本项目 `data_source/`
- 本地模型目录用符号链接复用，避免重复占用 20GB+ 空间：
  - `bge-m3 -> ../swift-rag/bge-m3`
  - `EvoQwen2.5-VL-Retriever-3B-v1 -> ../swift-rag/EvoQwen2.5-VL-Retriever-3B-v1`
- Python 环境使用 conda 管理；推荐先复用已有 `rag-api`
- 默认 embedding 已改为 `bge_m3` + `sentence-transformers`，设备优先使用 `cuda`，CUDA 不可用时回退 CPU
- 当前已有 `data/index/gbrain.sqlite3` 是之前用于快速验证的 `hashing` 索引；切到 `bge_m3` 后需要重建索引才能启用真正的向量召回，未重建时 hybrid 仍会使用 keyword/BM25 + entity graph

## Conda 环境

复用已有环境：

```bash
cd gbrain-rag
conda activate rag-api
python -m pip install -r requirements.txt
```

或者创建独立 conda 环境：

```bash
cd gbrain-rag
conda env create -f environment.yml
conda activate gbrain-rag
```

## 重建 BGE-M3 索引

```bash
conda activate rag-api
cd gbrain-rag
PYTHONPATH=src python scripts/build_index.py \
  --reset \
  --embedding-model bge_m3 \
  --embedding-backend sentence-transformers \
  --embedding-device cuda
```

如果只是快速 smoke test，可以临时覆盖为 hashing：

```bash
PYTHONPATH=src python scripts/build_index.py \
  --reset \
  --embedding-model hashing \
  --embedding-backend hashing
```

## 启动

```bash
conda activate rag-api
cd gbrain-rag
PYTHONPATH=src python -m uvicorn gbrain_rag.main:app --host 0.0.0.0 --port 6061
```

打开：

- `http://127.0.0.1:6061/docs`
- `http://127.0.0.1:6061/api/v1/rag/health`

## 多源检索策略

系统把语料拆成三类 source，并按问题意图选择数据源：

- `document`：ONES 工作文档 / PDF 发版正文。适合回答当前版本输入输出、阈值、算法边界、功能介绍、优化点、追加数据、标签等正文细节。
- `table`：模型发版记录汇总。适合回答模型清单、负责人、OID、更新时间、推荐配置、支持设备、数量统计等结构化字段。
- `adela`：Adela 部署信息。适合回答 did/rid、部署平台、部署状态、部署版本、部署记录细节。

统一接口会先做确定性路由，再按源独立召回，最后跨源排序展示证据。这样跨源问题不会因为单一大池子的高分结果把弱源挤掉。

示例：

- “安全绳 v0.2.1 输入输出是什么？” -> `document`
- “这个模型 OID 和负责人是谁？” -> `table`
- “这个模型在哪些平台部署了，did/rid 是多少？” -> `adela`
- “发版文档和部署平台是否一致？” -> `document + table + adela`

也可以手动指定源：

```json
{
  "query": "安全绳检测 v0.2.1 的输出是什么？",
  "sources": ["document"],
  "top_k": 5
}
```

每个源可单独配置：

```json
{
  "query": "横幅标语模型的发版信息和部署情况",
  "top_k": 5,
  "table": {"top_k": 8, "retrieval_method": "hybrid"},
  "adela": {"top_k": 8, "retrieval_method": "hybrid"}
}
```

## API 示例

```bash
curl --noproxy '*' -s http://127.0.0.1:6061/api/v1/rag/chat_engine/unified_retrieve \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "安全绳检测 v0.2.1 的输出是什么？",
    "top_k": 5,
    "retrieval_method": "hybrid"
  }' | python -m json.tool
```

问答接口：

```bash
curl --noproxy '*' -s http://127.0.0.1:6061/api/v1/rag/chat_engine/unified_query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "安全绳检测 v0.2.1 的输出是什么？",
    "top_k": 5,
    "retrieval_method": "hybrid"
  }' | python -m json.tool
```

问答响应中的 `knowledge_base_fully_answered` 是下游可直接使用的 float 置信度字段，范围为 `0.0`-`1.0`，由 LLM 评估当前知识库证据是否足以全面回答；未命中、证据不足或仅返回降级片段时会给低分。

流式问答接口使用 SSE，设置 `stream=true`：

```bash
curl --noproxy '*' -N http://127.0.0.1:6061/api/v1/rag/chat_engine/unified_query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "安全绳检测 v0.2.1 的输出是什么？",
    "top_k": 5,
    "retrieval_method": "hybrid",
    "stream": true
  }'
```

默认 LLM 配置在 `.env.example`，调用方式与 `swift-rag` 的 OpenAI-compatible 接口保持一致。

## 测试

```bash
conda activate rag-api
cd gbrain-rag
PYTHONPATH=src python -m unittest discover -s tests -v
```

## 参考

- `garrytan/gbrain`: https://github.com/garrytan/gbrain
- 本地参考项目：`../swift-rag`
- 项目说明文档：`docs/README.md`
