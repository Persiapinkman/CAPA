# test-skills/demo 架构说明（当前实现）

目标：围绕一次 `POST /run` 请求，完成「多步规划 -> 工具执行 -> 记忆沉淀 -> 最终回答 -> NDJSON 流式返回」。

---

## 1) 一句话总览

前端把 `text/image/session_id` 发到 `demo_server.py`，服务端通过 `AgentOrchestrator` 进行多步决策，`ToolExecutor` 调用具体技能脚本，`memory_system` 持久化会话账本并回灌下一步上下文，最后输出 `final_answer` 并持续流式推送中间事件。

---

## 2) 分层与职责

### A. 接入层（HTTP + 会话）

**核心文件**
- `demo_server.py`
- `frontend_page.py`

**做什么**
- 提供 `GET /`（页面）和 `POST /run`（`multipart/form-data`）。
- 处理上传图片、创建 `demo/runs/<run_stamp>/` 产物目录。
- 管理会话文件 `demo/sessions/<session_id>.json`，并用会话锁避免并发写冲突。
- 以 `application/x-ndjson` 持续推送执行过程。

**关键产物**
- 运行期文件：`intent.json`、`prompts.json`、`prediction.json`、`evaluation.json`、图片等。
- 会话持久化：`raw_ledger`、`working_trajectory`、`session_state`、`summary_history`、`last_image_path`。

### B. 编排层（Agent Brain）

**核心文件**
- `src/capa/agent.py`
- `src/capa/prompts.py`
- `src/capa/tools/registry.py`
- `src/capa/tools/contracts.py`

`demo/agent.py`、`demo/prompts.py` 与 `demo/tools/*` 仅保留兼容导入。

**做什么**
- Planner 以结构化 JSON 决策：`thought + action + action_input + final_answer`。
- 支持最多 `DEMO_AGENT_MAX_STEPS`（默认 5）轮循环。
- 统一新旧 action 名称映射，兼容历史分支。
- 在需要时触发 Answerer 生成面向用户的最终答复（含 fallback）。
- 支持 LLM 调试落盘到 `demo/llm_debug/`。

**当前可用工具（tool name）**
- `rag_answer`
- `flux-image-generation`
- `qwen_detection`
- `rexomni_detection`
- `pipeline_eval`
- 以及结束动作：`final_answer`

### C. 执行层（Tool Gateway）

**核心文件**
- `src/capa/tools/executor.py`
- `demo_server.py` 内各 `_run_*_streaming` 实现

**做什么**
- 把 `ToolCall` 分发到具体执行分支：`rag / flux / qwen_detection / rexomni_detection / pipeline`。
- 执行前做参数和资源校验（例如图片是否存在、`label` 是否为空）。
- 返回统一结构 `ToolResult(action, observation, ok, error_message)`。

**执行方式**
- 大部分能力通过子进程调用 `skills/*/scripts/*.py`。
- 执行日志写入 `demo/runs/run_log.txt` 和本次 `run_dir`。

### D. 记忆层（Ledger + 投影）

**核心文件**
- `src/capa/memory.py`
  - `LedgerStore`
  - `MemoryProjector`
  - `ContextBuilder`

**做什么**
- 维护 append-only 事件账本 `raw_ledger`。
- 维护步骤轨迹 `working_trajectory`（动作、状态、关联事件）。
- 从 observation 投影 `working_memory`、更新 `session_state`（如 `global_kv_state`、`asset_registry`）。
- 构造 `planner_context`（`history + working_trajectory + session_state`）给下一轮 Planner。

### E. 技能层（外部脚本能力）

**典型脚本**
- `skills/rag-retrieve-answer/scripts/run_rag.py`
- `skills/user-intent-understanding/scripts/run_intent.py`
- `skills/llm-prompts-generation/scripts/run_prompt_generation.py`
- `skills/flux-image-generation/scripts/run_generation.py`
- `skills/qwen-vlm-open-set-delection/scripts/run_detection.py`
- `skills/rexomni-open-set-detection/scripts/run_detection.py`
- `skills/eval-reports-generation/scripts/run_eval_report_generation.py`

**特点**
- 通过 CLI 参数输入，输出 JSON/图片文件。
- 与主流程通过文件交换，解耦清晰但依赖外部服务可用性。

---

## 3) 一次请求的数据流（端到端）

1. 前端提交 `text`、可选 `image`、可选 `session_id` 到 `POST /run`。  
2. 服务端创建 `run_dir`，加载会话，写入 `USER_INPUT` 事件。  
3. `ContextBuilder` 生成 `planner_context`。  
4. `AgentOrchestrator` 选择下一步 `action`（或直接 `final_answer`）。  
5. 若是工具动作：`ToolExecutor` 执行并得到 `observation`，同时推送中间事件。  
6. `MemoryProjector.persist_step` 把计划与观察写入 ledger/trajectory，刷新会话状态。  
7. 重复 3~6，直到 planner 结束或达到最大步数。  
8. 生成 `final_answer`，写入 `ASSISTANT_OUTPUT`，保存会话，发送 `done`。  

---

## 4) 前端事件协议（NDJSON）

`POST /run` 返回逐行 JSON。常见事件：
- `session`：返回会话 ID。
- `meta`：当前步骤路由与决策信息。
- `direct_reply`：RAG 或证据综合的直接文本。
- `generated_one`：生成图 URL。
- `annotated`：标注图 URL 列表。
- `evaluation`：评测结果摘要。
- `final_answer`：最终答复。
- `error` / `done`：失败或完成信号。

---

## 5) 当前架构结论

- **优点**：编排、执行、记忆分层清晰；新增工具只需补 registry + executor 分支。  
- **关键约束**：强依赖子进程脚本和外部 API；I/O 与网络抖动会直接影响链路稳定性。  
- **适用场景**：多工具编排、可解释中间步骤、需要会话连续性的 Demo/实验系统。  
