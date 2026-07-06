# Demo 评测与 Trace 审计（demo/eval）

本目录用于 **从真实会话挖掘评测素材**，并对已录制的 Agent trace 做**离线审计**（工具路由、observation 忠实性、final 报告等）。偏 Planner / 路由迭代与回归分析；面向发布门禁的精简集见 [`../../dataset/README.md`](../../dataset/README.md)。

> 仓库根 `.gitignore` 默认忽略 `demo/eval/` 与 `demo/sessions/`。脚本可保留在本地或单独提交；重新生成的大 JSON/CSV 通常不入库。

## 目录一览

```
demo/eval/
├── build_from_sessions.py              # 仅从 sessions 抽取去重单轮问题
├── build_eval_suite.py                 # 完整套件：打标签 + 模板 case + 挖掘多轮 case
├── run_planner_routing_eval.py         # P0 Planner 首步路由评测（不执行工具）
├── build_planner_routing_dpo.py        # 从 routing eval 失败样本构造 DPO pair
├── tag_questions.py                    # 单轮问题的 suggested_tags 规则
├── run_trace_audit.py                  # 将会话 trace 与 case 对齐并输出审计报告
│
├── session_questions_eval.json         # 单轮题库（去重用户问题 + suggested_tags）
├── session_questions_eval.csv          # 上表 CSV 视图
├── session_cases_eval.json             # 多轮/澄清/闭环 case（模板 + 会话挖掘）
│
├── target_detection_agent_cases.json   # 目标检测 Skill 专用小集（4 case）
├── target_detection_agent_trace_report.json  # 对上述 case 的审计结果（示例）
│
└── trace_audit_report.json             # 对 session_cases 子集的审计结果（示例）
```

**输入依赖**：`demo/sessions/**/*.json`（Demo Server 录制的会话，含 `raw_ledger`、`query_trajectories`）。**对照产物**：`demo/runs/<run_stamp>/` 中 observation 外部 JSON（`external_ref` 指向的文件）。

## 数据文件说明

| 文件 | 典型规模 | 说明 |
|------|----------|------|
| `session_questions_eval.json` | ~100+ 题 | 从 sessions 去重后的**单轮用户问题**；`ground_truth` 多为空，供人工补全 |
| `session_questions_eval.csv` | 同上 | 表格视图：`id`、`question`、`suggested_tags`、`occurrence_count` 等 |
| `session_cases_eval.json` | ~26 case | **模板 case**（如 MT-01 RAG 多轮）+ **session_mined** 多轮片段；含 `expect.tool_sequence`、`suggested_tags` |
| `target_detection_agent_cases.json` | 4 case | `target-detection-evaluation` / `pipeline_eval` 路由与边界（生图、单模型检测 vs 完整 pipeline） |
| `trace_audit_report.json` | 报告 | `run_trace_audit.py` 默认对 `session_cases_eval.json` 前 N 条跑出的汇总 |
| `target_detection_agent_trace_report.json` | 报告 | 对 `target_detection_agent_cases.json` 跑审计的示例输出 |

### `session_cases_eval.json` 字段（节选）

- **`interaction`**：`single_turn` \| `multi_turn` \| `clarification_resume` \| `fragment`
- **`source.type`**：`template`（手写标准）\| `session_mined`（从 ledger 挖掘）
- **`turns`**：用户轮次文本，用于与会话 thread 匹配
- **`expect.tool_sequence`** / **`must_not`**：期望工具与禁止工具
- **`expect.suggested_tags`**：与 `tag_questions.py` 一致的标签（interaction / intent / planner / finish）
- **`audit.required_tools`**（可选）：审计时优先使用的工具列表；未写则从 `tool_sequence` 推导

### `target_detection_agent_cases.json` case 列表

| case_id | 要点 |
|---------|------|
| `TD-PIPE-001` / `TD-PIPE-002` | 「钓鱼的人，生成精度评估报告」→ 必须 `pipeline_eval` |
| `TD-BOUNDARY-001` | 「生成 3 张钓鱼的图片」→ 仅 `flux-image-generation`，不得 `pipeline_eval` |
| `TD-BOUNDARY-002` | 「用 qwen 标注钓鱼的人」→ 仅 `qwen_detection`，不得 `pipeline_eval` |

case 中 `setup.image` 如 `fixture/fishing_scene.jpg` 为占位路径；实际跑 Demo 时需换成仓库内图片（如 `examples/images/fisherman.jpg`）。

## 脚本用法

在仓库根目录执行：

```bash
# 1) 仅从 demo/sessions 抽取单轮问题库
python3 demo/eval/build_from_sessions.py \
  --sessions-dir demo/sessions \
  --out-dir demo/eval

# 2) 生成完整套件（打 suggested_tags + 模板 case + 挖掘多轮，默认再挖 12 条）
python3 demo/eval/build_eval_suite.py \
  --sessions-dir demo/sessions \
  --out-dir demo/eval \
  --mine-limit 12

# 3) 对已有 case 文件做 trace 审计（默认读 session_cases_eval.json，审计前 12 条）
python3 demo/eval/run_trace_audit.py \
  --cases demo/eval/session_cases_eval.json \
  --sessions-dir demo/sessions \
  --limit 12 \
  --out demo/eval/trace_audit_report.json

# 目标检测专用小集
python3 demo/eval/run_trace_audit.py \
  --cases demo/eval/target_detection_agent_cases.json \
  --sessions-dir demo/sessions \
  --limit 4 \
  --out demo/eval/target_detection_agent_trace_report.json

# P0 Planner 路由评测（只调用 Planner 首步，不执行重工具）
python3 demo/eval/run_planner_routing_eval.py \
  --model Qwen3.5-9B \
  --out results/planner_routing_eval/planner_routing_report_Qwen3.5-9B.json

# 从 4B/9B 路由失败样本构造 repair-style DPO pairs
python3 demo/eval/build_planner_routing_dpo.py \
  --out-dir results/planner_routing_eval/dpo
```

`tag_questions.py` 由 `build_eval_suite.py` 调用，一般无需单独运行；规则基于关键词推断 `lookup_fact`、`vision_pipeline`、`migration_advisor` 等标签。

## Trace 审计项（与 dataset T1–T6 对应）

`run_trace_audit.py` 在**已匹配**到 `demo/sessions` 中某 thread 的 case 上统计以下检查（报告里为英文 key）：

| 审计 key | 含义 | 对应 dataset |
|----------|------|----------------|
| `called_required_tools` | 是否调用期望工具 | T1 |
| `observation_reflects_output` | ledger observation 与 `demo/runs` 外部 JSON 是否一致 | T2 |
| `evaluator_disclosed_failure_or_uncertainty` | 失败/空结果是否在 final 中披露 | T3 |
| `final_report_supported_by_observation` | final 是否有 observation 依据 | T4 |
| `working_memory_carried_early_observation` | 多轮是否延续早期摘要 | T5 |
| `critical_failure_exposed` | 有关键失败时不得只报成功 | T6 |

报告 `summary.counts` 形如 `"4/6"`：分子为通过数，分母为适用 case 数；`cases_matched` 低于 `cases_total` 表示会话中未找到相同用户轮次（需新录 session 或改 case 文本）。

## 与 `dataset/`、`results/` 的分工

| 目录 | 角色 |
|------|------|
| **`demo/eval/`** | 从 **`demo/sessions`** 生长 case、离线 trace 审计；体量大、随会话更新 |
| [`dataset/`](../../dataset/README.md) | 发布门禁用**精简、稳定** case（如 `agent_trace_eval.json` 16 条、`migration_advisor_eval` 30 条） |
| [`results/`](../../results/README.md) | pipeline / 迁移顾问等**脚本运行产物** |
| `demo/runs/` | 单次 Demo / API 调用的完整落盘 |

迭代建议：在 `demo/eval` 验证路由与 trace 质量后，将稳定、可重复的 case **下沉**到 `dataset/agent_trace_eval.json`，并同步 CSV 与 `scores.template.csv`。

## 维护注意

- 更新 `demo/sessions` 后重新跑 `build_eval_suite.py`，`generated_at` 与 `stats` 会变化。
- `ground_truth` 在 questions / cases 中多为空，需人工或另 pipeline 补全后再做答案级 judge。
- 勿将绝对路径、API key 或未脱敏客户原文写入生成的 JSON；`session_path` 字段仅用于本地调试定位。
