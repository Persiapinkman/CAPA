# Human Review: planner_stateful_retrieval_v2

## 人工阅读重点

首要阅读本文档，然后查看 `DATASET_CARD.md` 的五步实例。只有需要抽查原始标签时才打开 case JSONL；不要直接从 6144-token ChatML 训练行开始读。

## 推荐审阅顺序

1. 在 `manifest.json` 核对 split 数量、文件 SHA256、零重叠和测试封存状态。
2. 在 `planner_stateful_retrieval_v2_dev_cases.jsonl` 按七个 `category` 各抽查至少 3 个 case。
3. 对 `rag_double_miss_recovery` 逐步检查 observation 与动作是否严格交替，尤其是 step 4 的 `retrieval_round=3`。
4. 对 `rag_hit_then_synthesize` 确认第一步 `finish_after_tool=false`，第二步为 `answerer(mode=rag_evidence)`。
5. 对三个 guardrail 确认没有把 `re_question` 或 `rag_answer` 当作通用默认动作。
6. 核对所有行的 `wrong_action_cap=0.20`，并确认训练命令使用 task/format 外层权重 `0.95/0.05`。
7. 核对 `development_replication_gate.json` 为通过后，再阅读 test prediction 或 test 指标；本研究已按该顺序完成一次性 test 确认。

## 拒绝条件

- observation 说“命中”却标成 `re_question`，或说“未命中”却提前结束。
- 第三轮检索仍设置 `finish_after_tool=false`，造成无界循环。
- train/dev/test 有实体、模板、case ID 或精确 query 重叠。
- 把同一实体的多个 case 当作独立 bootstrap 单位。
- 错误动作可获得超过 `0.20` 的 task reward。
- 开发复现门通过前读取 test prediction，或根据 test 结果调整超参数。

## 人工抽查命令

```bash
jq -c 'select(.category=="rag_double_miss_recovery") | {case_id,user_query,expected_decisions,mock_observations}' \
  training/planner_grpo_seed_v1/cases/planner_stateful_retrieval_v2_dev_cases.jsonl | head
```

```bash
jq -c '{case_id,category,step_index,expected:(.expected_step|fromjson)}' \
  training/planner_grpo_seed_v1/sft_data_stateful_retrieval_v2_chatml/support_audit.jsonl
```
