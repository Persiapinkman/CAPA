# Search Agent

## 核心功能

- **多轮智能检索**：支持多轮自适应检索，确保找到足够的相关信息
- **实时流式输出**：支持流式响应，提供更好的用户体验
- **智能结果过滤**：自动删除与用户问题无关的检索结果，提高答案质量
- **自适应查询**：根据检索结果自动调整查询策略

## 文件结构

```
search_agent/
├── agent.py           # 核心逻辑
├── config.py          # 配置文件和系统提示词
├── llm_utils.py       # LLM接口函数
├── tool_utils.py      # 检索工具函数
├── test_streaming.py  # 流式输出测试文件
└── README.md          # 本文档
```

## 文件说明

### config.py
配置管理模块，包含：
- `DEFAULT_LLM_API_CONFIG`: 默认LLM API配置
- `DEFAULT_SEARCH_CONFIG`: 默认检索引擎配置
- `DEFAULT_SEARCH_AGENT_CONFIG`: 默认智能代理配置
- `SYSTEM_PROMPT`: 系统提示词模板
- `CHUNK_SEARCH_TOOL`: chunk检索工具定义
- `CHUNK_DELETE_TOOL`: chunk删除工具定义

### test_streaming.py
测试模块，包含：
- `test_function_streaming()`: 测试函数级流式输出
- `test_api_streaming()`: 测试API级流式输出

## 使用方法

### 测试示例

```bash
# 运行功能测试
python test_streaming.py
```

## 工作流程

1. **初始检索**：使用用户原始查询进行第一轮检索
2. **结果分析**：LLM分析检索结果是否足够回答问题
3. **迭代检索**：如果需要更多信息，调整查询词进行后续检索
4. **结果过滤**：在多轮检索后，删除与问题无关的结果
5. **答案生成**：基于所有相关检索结果生成最终答案

## 特性说明

### 智能结果过滤
- 从第2轮开始，如果有多个chunk历史，会触发智能过滤
- 自动识别和删除与用户问题无关的检索结果
- 每轮检索至少保留一条结果，确保信息连续性

### 自适应检索策略
- 根据检索结果质量自动调整查询关键词
- 支持多轮检索，确保找到足够的相关信息
- 第一轮会同时使用原始查询和模型生成的查询进行检索

### 实时流式输出
- 支持Server-Sent Events (SSE)流式输出
- 实时显示检索和推理过程
- 提供更好的用户交互体验