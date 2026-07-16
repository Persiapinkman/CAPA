# Demo Agent 架构

## 端到端路径

```text
Browser / API
  -> demo/demo_server.py (HTTP, upload, NDJSON, session lock)
  -> AgentOrchestrator (Planner decision + deterministic state machine)
  -> ToolExecutor / inline Answerer
  -> skill scripts and external services
  -> MemoryProjector + LedgerStore
  -> NDJSON events and persisted session
```

一次 `POST /run` 的主要步骤：

1. 接入层解析文本、图片和 `session_id`，创建 `demo/runs/<run_stamp>/`。
2. `ContextBuilder` 将当前 query、历史轨迹、资产和状态构造成 Planner 上下文。
3. Planner 输出 `tool`、`clarify` 或 `end` 的结构化 JSON。
4. 前置条件检查缺失图片、标签、模型等参数，必要时进入澄清状态。
5. Executor 执行工具；`answerer` 由 Orchestrator 直接调用模型。
6. `MemoryProjector.persist_step` 将动作和 observation 写入 append-only ledger 及 working trajectory。
7. 非终止工具返回后重新规划，直至最终回答、固定终止分支或最大步数。

## 分层职责

| 层 | 核心文件 | 职责 |
|---|---|---|
| HTTP/会话 | `demo/demo_server.py` | 上传、NDJSON、会话隔离、反馈和测试端点 |
| 编排 | `src/capa/agent.py` | Planner/Answerer、澄清、循环和固定状态流转 |
| Prompt/协议 | `src/capa/prompts.py`, `src/capa/tools/schemas.py` | 决策协议、工具 schema、参数约束 |
| 注册表 | `src/capa/tools/registry.py` | 8 个默认规范工具、历史别名、flow 映射 |
| 执行 | `src/capa/tools/executor.py` | 工具分发、校验、统一 `ToolResult` |
| 记忆 | `src/capa/memory.py` | ledger、trajectory、working memory 和上下文投影 |
| 能力审计 | `src/capa/capabilities.py`, `src/capa/service_health.py` | 静态资产和外部服务状态分离 |

## 策略所有权

模型可学习的 Planner 决策：

- 首步选哪个工具或是否澄清。
- 工具参数是否完整且忠实于用户输入。
- `finish_after_tool`，即工具结果是终点还是中间证据。
- Qwen/Rex 视觉探针后是否继续 `migration_advisor`。
- 何时使用 `answerer`、记忆结束或直接工具结果。

编排器硬编码、不能用当前 Planner GRPO 学习的流转：

- RAG miss 后强制 `re_question`。
- `re_question` 完成后同轮立即调用 `rag_answer`。
- 最多三轮 RAG，第三轮 miss 后固定发出迁移顾问选项。
- `pipeline_eval`、`migration_advisor` 的专用终止行为。

因此，训练强制 RAG 中间动作只能证明奖励和采样有可学习性，不能证明当前 Demo 策略改善。当前 RL 主实验选择“显式 Qwen 探针后继续迁移”作为运行时真实可控的两步场景。

## 服务与契约

`GET /health` 只验证代码资产和工具契约；`GET /health/capabilities?live=1` 才探测外部服务。静态通过与 live 通过必须分别报告，避免把脚本存在误写成能力可用。

生产 RAG 的语料和索引不在仓库内，通过 `pipelines/demo/open_rag_tunnel.sh` 转发远端 6061/6062 服务。2026-07-14 已验证真实生成式问答、Flux 和完整检测 pipeline；Adela 已从默认 Planner、Executor 与 Agent RL action space 排除。详细证据见 `reports/DEMO_CAPABILITY_LIVE_CHECK_2026-07-14.md` 和 `reports/DEMO_FULL_PIPELINE_LIVE_CHECK_2026-07-14.md`。
