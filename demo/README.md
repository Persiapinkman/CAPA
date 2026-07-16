# CAPA Demo Agent

`demo/demo_server.py` 是 CAPA 的 HTTP 接入与会话服务。它不是关键词触发的固定脚本：每次请求由 Planner 在结构化上下文上选择动作，编排器执行状态机约束，工具网关调用具体 skill，最终通过 NDJSON 流返回过程和结果。

## 启动

模型网关需要 SOCKS5 代理，生产 RAG 通过 SSH 转发到本机端口。先在终端 1 执行并保持运行：

```bash
bash pipelines/demo/open_rag_tunnel.sh
```

SSH 会交互式询问密码，脚本不会保存密码。然后在终端 2 执行：

```bash
source init_env.sh
.venv/bin/python demo/demo_server.py --port 18080
```

`init_env.sh` 导出 `socks5h://127.0.0.1:8888` 代理、模型服务地址、Qwen 检测地址和本地 RAG 转发地址。必须使用 `source` 或 `. init_env.sh`，否则环境变量不会进入当前 shell。

默认绑定 `0.0.0.0`：

- 页面：`http://127.0.0.1:18080/`
- 静态契约：`GET /health`
- 能力清单：`GET /health/capabilities`
- 非副作用服务探活：`GET /health/capabilities?live=1`
- Flux 只读模型列表探活：额外加 `include_flux=1`

## 请求协议

`POST /run` 只接受 `multipart/form-data`：

- `text`：用户问题。
- `session_id`：可选，用于延续会话。
- 图片字段：可选，最多 10 张。

响应类型为 `application/x-ndjson`。常见事件包括 `session`、`meta`、`direct_reply`、`generated_one`、`annotated`、`evaluation`、`migration_advisor_offer`、`final_answer`、`error` 和 `done`。

会话保存在 `demo/sessions/`，运行产物保存在 `demo/runs/<run_stamp>/`，LLM 调试记录保存在 `demo/llm_debug/`。这些目录可能包含真实使用痕迹，不应直接复制进训练集或报告。

## Agent 能力

Planner 默认可见的 8 个规范工具为：

| Tool | 用途 | 运行时所有者 |
|---|---|---|
| `rag_answer` | 公司私有知识检索 | Executor |
| `re_question` | RAG miss 后最小改写 | Executor |
| `answerer` | 通用回答或证据综合 | Orchestrator |
| `flux-image-generation` | 文生图/参考图扩增 | Executor |
| `qwen_detection` | Qwen 单图快速检测 | Executor |
| `rexomni_detection` | Rex-Omni 单图快速检测 | Executor |
| `pipeline_eval` | 扩增、双模型检测和评测报告 | Executor |
| `migration_advisor` | 能力边界和迁移方案 | Executor |

Adela 已退出当前 Demo 与 Agent RL 范围。旧执行实现仅为历史兼容保留，默认不会进入 Planner schema、有效 action 集、能力清单或服务探活；只有显式设置 `CAPA_ENABLE_ADELA=1` 才会恢复。

Planner 决定首步路由、参数、`finish_after_tool`，以及非终止工具之后的下一步。RAG miss 后的 `re_question -> rag_answer` 重试和第三次 miss 后的迁移顾问询问是编排器硬约束，不是当前 Planner/GRPO 可以改变的动作。

## 外部依赖边界

- Planner、Answerer、query rewrite 和 Rex-Omni 使用 `DEMO_LLM_API_BASE` 模型网关。
- RAG 默认访问本机 `6061/6062`；当前生产索引由 SSH 转发提供，仓库内 gbrain/ace 实现只用于开发或服务故障时的独立启动。
- Qwen detection 由 `init_env.sh` 配置为 `10.111.32.254:9012/v1`，通过 SOCKS 代理访问。
- Flux 使用凭据化外部 API，可能产生费用，不在默认探活中执行生成。

静态脚本存在只表示能力契约可复现，不表示外部服务在线。当前逐项验收结果见 `reports/demo_agent_capability_reproduction.md`。

## 后端迁移顾问测试

`POST /test/migration-advisor` 接受 JSON，可用单条 `query`，或通过 `csv_path` 加 `index/indices` 批量运行。设置 `force_migration_advisor=true` 可跳过前端确认；结果写入对应 run 目录的 `migration_advisor_report.json` 和 `migration_advisor_report.md`。

迁移报告只落盘有界证据摘要，并对 `evidence_id/doc_id + quote` 做确定性校验。无同集人工 GT 时，性能、周期和成本必须显示为证据不足；工程建议与知识库事实分开标记。

## 完整复验

无副作用的默认复验：

```bash
source init_env.sh
uv run python pipelines/demo/run_full_demo_smoke.py --include-migration
```

Flux 与完整 pipeline 会产生外部调用，必须显式授权：

```bash
source init_env.sh
uv run python pipelines/demo/run_full_demo_smoke.py \
  --include-migration --include-flux --include-pipeline --allow-side-effects
```

脚本分别报告 runtime 状态和 RL readiness。pipeline 即使无运行错误，只要扩增多样性门未通过，也不会被标记为可用于 RL。

## 阅读入口

- Agent 的真实职责、状态机与 RL 边界：`reports/DEMO_AGENT_RUNTIME_ANALYSIS.md`
- 2026-07-14 RAG/Qwen/Rex 实时端到端复验：`reports/DEMO_CAPABILITY_LIVE_CHECK_2026-07-14.md`
- 2026-07-14 Flux/完整 pipeline 与 RL readiness：`reports/DEMO_FULL_PIPELINE_LIVE_CHECK_2026-07-14.md`
- 能力服务复现矩阵：`reports/demo_agent_capability_reproduction.md`
- 当前路由训练集的人工样例：`data/datasets/planner_runtime_probe_curriculum_v1/HUMAN_REVIEW.md`
