# Swift-RAG

## Highlights

支持 `document` 模型发版文档正文问答、`table` 表格问答、`adela` 部署记录问答、`unified` 统一检索网关问答（基于 query 的 LLM 数据源路由）、`unified` 统一检索直出（只返回融合证据，不走最终 LLM answer）。

## 项目概述

本项目是一个基于 FastAPI 的高性能 RAG 服务框架，提供文档处理、向量化和检索等核心能力。
当前 API 默认 embedding 模型已统一为 `EvoQwen2.5-VL-Retriever-3B-v1`。

这里的 `document` 不是泛指“任意文档”，而是第一个核心数据源的固定代号：它表示模型发版文档的正文内容，来源于 ONES 工作文档及对应发版 PDF。相比 `table` 这种结构化汇总表，`document` 承担的是“具体发版内容本身”的检索，包括输入输出、阈值、优化点、追加数据、标签、背景说明，以及正文里的 table/image 细节。

## 项目运行

### 直接测试 document 模型发版文档正文问答

```bash
python sample_code/rag_chat_api_client.py
```

### 直接测试 table 表格问答

```bash
python sample_code/table_chat_api_client.py
```

### 直接测试 adela 部署记录问答

```bash
python sample_code/adela_chat_api_client.py
```

### 直接测试 unified 统一检索网关问答

```bash
python sample_code/unified_chat_api_client.py
```

### 直接测试 unified 统一检索网关流式问答

```bash
python sample_code/unified_stream_api_client.py
```

### 直接测试 unified 统一检索直出

```bash
python sample_code/unified_retrieve_api_client.py
```

### 模拟真实 API 调用并统计耗时

```bash
python sample_code/rag_chat_benchmark.py \
  --query "安全绳检测用哪个模型？"
```

批量问题压测：

```bash
python sample_code/rag_chat_benchmark.py \
  --queries-file sample_code/questions.txt \
  --repeat 3
```

### Docker 镜像（python 代码加密）

```bash
registry.st-sh-01.sensecore.cn/scg_rdbp_ccr/swift-rag:v1.0
```

### 一键部署并运行 RAG 服务

```bash
docker run -itd \
  --ipc=host \
  --name swift_rag_crypt \
  --gpus all \
  --privileged=true \
  --network default \
  -p 6060:6060 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  registry.st-sh-01.sensecore.cn/scg_rdbp_ccr/swift-rag:v1.0 \
/bin/bash -c "source activate rag-api && python -m src.main"
```

### 启动并进入容器环境

```bash
sh ./docker_run.sh
```

### 进入容器运行 API 服务

```bash
conda activate rag-api
python -m src.main
```

服务默认运行在 `http://0.0.0.0:6060`。

启动后探活：

```bash
curl http://127.0.0.1:6060/openapi.json
curl http://127.0.0.1:6060/docs
```

说明：

- API 服务进程启动后会先监听 `6060` 端口
- 向量模型按请求懒加载，首次检索/问答请求通常更慢
- 若本机网络较慢，启动阶段可能出现 `nltk` 缓存检查日志，不影响服务可用性

### 直接执行 RAG 处理流程（非 API 方式）

```bash
python -m src.rag.pipeline.chenggong_pipeline
```

该流程会默认递归读取 `data_source` 目录（优先 `/app/data_source`，其次仓库根目录 `data_source`）并执行离线入库。
若数据源包含 PDF，请先安装 `pypdf` 或 `PyPDF2`。

## 本地模型准备

默认会从仓库根目录读取以下向量模型目录：

- `./EvoQwen2.5-VL-Retriever-3B-v1`（默认）
- `./bge-m3`（可选）

请确保需要使用的模型目录中已存在实际模型文件，而不是空目录。

- 若模型实际放在其他目录，请修改 [src/core/config.py](src/core/config.py) 中的 `model_path` 配置

模型目录中至少应包含 `config.json` 或 `modules.json`，否则无法被 `sentence-transformers` 正常加载。

## 项目测试

### 测试 API 接口

```bash
python -m src.test_rag_api
```

测试脚本会自动测试文档处理、检索和 `document` 模型发版文档正文问答三个接口，并输出结果。

## API 接口

详细接口说明、请求示例和响应示例见：

- [API_USAGE.md](docs/API_USAGE.md)

重点接口：

- `POST /api/v1/rag/doc_engine/chunking_embedding`
- `POST /api/v1/rag/chat_engine/query`
- `POST /api/v1/rag/chat_engine/table_query`
- `POST /api/v1/rag/chat_engine/adela_query`
- `POST /api/v1/rag/chat_engine/unified_retrieve`
- `POST /api/v1/rag/chat_engine/unified_query`
- `POST /api/v1/rag/embedding`

## 项目依赖

- **fastapi**: Web 框架
- **pydantic**: 数据模型定义
- **pydantic_settings**: 配置管理
- **uvicorn**: 运行服务
- **llama-index-core**: 文档切分、Node 与 Embedding 抽象
- **llama-index-vector-stores-milvus**: 向量数据库
- **pymilvus**: 向量数据库
- **numpy**: 数值计算
- **pandas**: 数据处理
- **openpyxl**: Excel 文件读取
- **beautifulsoup4**: HTML 解析
- **requests**: HTTP 请求
- **aiohttp**: 异步 HTTP 请求
- **openai**: 对接 OpenAI 兼容接口
- **sentence-transformers**: 向量模型加载
- **torch**: 向量模型推理
- **tqdm**: 进度条显示

### 依赖安装

优先使用仓库根目录下的 [requirements.txt](requirements.txt) 安装依赖。

补充说明：

- `torch` 请根据机器环境选择对应的 CPU / CUDA 安装方式
- 若需要处理 PDF 数据源，请额外安装 `pypdf` 或 `PyPDF2`
- 若需要运行前端 demo，请额外安装 `streamlit`
- 若需要运行评测脚本，请补充安装 `jieba`、`nltk`、`transformers`、`rouge-chinese`、`loguru`

## 环境配置

项目配置通过 `src/core/config.py` 中的 `Settings` 类管理，主要配置项包括：

- API 配置：路径前缀、项目名称等
- 服务器配置：主机、端口、是否热重载
- 向量数据库配置：URI、collection 名称、是否覆盖等
- 检索配置：`DEFAULT_RETRIEVAL_METHOD`（默认 `hybrid`）、`BM25_K1`、`BM25_B`、`BM25_CANDIDATE_LIMIT`
- PDF 来源映射配置：`PDF_REFERENCE_MAPPING_PATH`，默认指向 `data_source/pdf_reference_links.csv`
- adela 部署链接模板配置：`ADELA_DEPLOYMENT_URL_TEMPLATE`，默认
  `http://adela.sensetime.com/mainpage/project/3/models?deployment_id={did}`
