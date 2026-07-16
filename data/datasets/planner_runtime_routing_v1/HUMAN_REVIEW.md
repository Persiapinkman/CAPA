# Human Review: planner_runtime_routing_v1

## 阅读顺序

先读 `DATASET_CARD.md`，再读 `manifest.json`；只有核对标签时才打开 case JSONL，不要从 ChatML step 行开始。

## 必查项目

1. 每个 split 按 13 个 category 各抽查至少 3 个 case。
2. `qwen_probe_then_migration` 和 `rex_probe_then_migration` 的 step 1 必须为检测且 `finish_after_tool=false`，step 2 必须为 `migration_advisor`。
3. `qwen_probe_only_contrast` 必须在检测后结束，不能因为文本出现“迁移”而调用迁移顾问。
4. `pipeline_eval` 仅用于明确要求样本扩增、双模型对比和评估报告的请求。
5. `adela_eval` 必须同时保留模型名、平台串与 `eval_type`；精度为 0，性能为 1。
6. 缺模型、平台和评测类型的请求应为 `clarify`，不得编造参数或启动副作用工具。
7. `memory_end` 必须有充分的 prior trajectory，并标为 `end(memory_hit)`。
8. 核对 train/dev/test 的 entity、template、case ID 和精确 query 均无重叠。

## 拒绝条件

- 把单图快速检测标成完整 pipeline，或把完整评测标成单图检测。
- 探针到迁移路径提前结束、重复检测，或跳过探针。
- 在用户未明确请求时调用 Flux、pipeline 或 Adela。
- 将真实 session query、内部模型 ID、客户端信息或 RAG 原文复制进数据集。
- 使用 test 结果选择模型、训练步数、学习率或 reward。

## 抽查命令

```bash
jq -c 'select(.category=="qwen_probe_then_migration") | {case_id,user_query,expected_decisions,mock_observations}' \
  training/planner_grpo_seed_v1/cases/planner_runtime_routing_v1_dev_cases.jsonl | head -n 3
```

```bash
jq '{integrity, splits: (.splits | map_values({cases: .cases.cases, steps: .steps.rows, actions: .cases.expected_actions}))}' \
  data/datasets/planner_runtime_routing_v1/manifest.json
```
