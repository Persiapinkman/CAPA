# 服务启动与接口测试命令（CLI）

本文档给出本项目常用命令行：

- 启动服务
- 查看端口
- 关闭端口/停止服务
- 测试接口

## 1. 启动服务

### 1.1 前台启动（当前终端占用）

```bash
cd swift-rag
conda activate rag-api
export PYTHONPATH="$PWD"
python -m src.main
```

默认监听：`0.0.0.0:6060`

### 1.2 后台启动（tmux 推荐）

```bash
tmux new-session -d -s swift-rag-api \
  'cd swift-rag && \
   conda activate rag-api && \
   export PYTHONPATH="$PWD" && \
   python -m src.main > logs/api_stdout.log 2>&1'
```

查看会话：

```bash
tmux ls
```

查看日志：

```bash
tail -f logs/api_stdout.log
```

## 2. 查看端口

查看 `6060` 是否已监听：

```bash
lsof -i :6060 -P -n
```

探活检查：

```bash
curl http://127.0.0.1:6060/openapi.json
curl http://127.0.0.1:6060/docs
```

## 3. 关闭端口/停止服务

### 3.1 如果是 tmux 启动

```bash
tmux kill-session -t swift-rag-api
```

### 3.2 按端口杀进程（兜底）

先查 PID：

```bash
lsof -i :6060 -P -n
```

结束进程（把 `<PID>` 替换成实际 PID）：

```bash
kill <PID>
```

强制结束（必要时）：

```bash
kill -9 <PID>
```

## 4. 测试接口（curl）

基础地址：

```bash
BASE_URL="http://127.0.0.1:6060/api/v1"
```

### 4.1 document 文档问答接口

这里的 `document` 指模型发版文档正文内容，来源于 ONES 工作文档及对应发版 PDF，不是泛指任意文档。

```bash
curl -X POST "$BASE_URL/rag/chat_engine/query" \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "safety_rope v0.2.1 追加了什么数据，标签有哪些？",
    "retrieval_method": "hybrid",
    "top_k": 5,
    "similarity_threshold": 0.5,
    "embedding_models": ["bge_m3", "EvoQwen2.5-VL-Retriever-3B-v1"]
  }'
```

响应重点：

- `retrieved_chunks`：命中的模型发版文档正文 chunk
- `reference`：命中文档对应的来源链接
- `answer`：LLM 生成的答案
- `timings`：检索、回答、reference 和总耗时

### 4.2 table 表格问答接口

```bash
curl -X POST "$BASE_URL/rag/chat_engine/table_query" \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "安全绳有哪些模型？",
    "retrieval_method": "hybrid",
    "top_k": 20,
    "similarity_threshold": 0.15,
    "embedding_models": ["bge_m3"]
  }'
```

响应重点：

- `matched_rows`：命中的结构化表格行
- `answer`：LLM 生成的答案
- `timings`：检索、回答和总耗时

### 4.3 adela 部署记录问答接口

```bash
curl -X POST "$BASE_URL/rag/chat_engine/adela_query" \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "有哪些 cuda11.0-trt7.1-fp16-T4 的部署模型？",
    "retrieval_method": "hybrid",
    "top_k": 20,
    "similarity_threshold": 0.15,
    "embedding_models": ["bge_m3"]
  }'
```

响应重点：

- `matched_records`：命中的 adela 部署记录
- `matched_records[].entity.reference`：每条命中记录的部署跳转链接（基于 did 拼接）
- `reference`：去重后的 adela 部署链接列表（和 document 一样作为顶层 reference 返回）
- `answer`：LLM 生成的答案
- `timings`：检索、回答和总耗时

### 4.4 unified 统一检索问答接口

```bash
curl -X POST "$BASE_URL/rag/chat_engine/unified_query" \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "安全绳使用到的相关模型有哪些记录？",
    "fused_top_k": 12,
    "rrf_k": 60,
    "stream": false,
    "route_with_llm": true
  }'
```

开启流式返回时：

```bash
curl -N -X POST "$BASE_URL/rag/chat_engine/unified_query" \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "安全绳使用到的相关模型有哪些记录？",
    "stream": true
  }'
```

响应重点：

- `route_plan`：LLM 路由选择的数据源；其中 `document` 表示模型发版文档正文（ONES 工作文档 / 发版 PDF）
- `source_status`：各来源检索状态与耗时
- `fused_evidences`：融合后的证据
- `answer`：最终答案
- `timings`：路由、检索、融合、回答和总耗时
- `stream=true` 时会变为 SSE 分片输出，最后一条是 `data: [DONE]`
