# Planner DPO 偏好对挖掘报告

- planner 步骤总数：617（来自 llm_debug 真实 prompt）
- 涉及会话数：418
- 决策解析失败：54
- 生成偏好对：**98**

## 偏好对按类型

| pair_type | 数量 | 说明 |
|-----------|------|------|
| contrastive | 50 | chosen=真实好决策，rejected=合成反例（chosen 真实，更安全） |
| repair | 48 | rejected=真实坏决策，chosen=合成修复（需复核） |

## 各规则命中数

| rule_id | 偏好对数 |
|---------|----------|
| R5_degenerate_decision | 26 |
| B1_platform_eval_correct | 25 |
| B3_generic_direct_correct | 17 |
| R3_clarify_loop | 17 |
| B2_coref_rewrite_correct | 8 |
| R2_coref_no_rewrite | 3 |
| R1_platform_misroute | 2 |

## 步骤弱标签分布

| label | 数量 |
|-------|------|
| skipped_no_response | 27 |
| good | 491 |
| bad | 48 |
| good_anchor | 50 |
| neutral | 1 |

## 决策动作分布

| action | 数量 |
|--------|------|
| rag_answer | 317 |
| answerer | 59 |
| unparsed | 54 |
| clarify | 51 |
| adela_cli_eval | 48 |
| qwen_detection | 24 |
| flux-image-generation | 19 |
| re_question | 18 |
| end | 9 |
| pipeline_eval | 8 |
| final_answer | 5 |
| rexomni_detection | 3 |
| adela_cli_benchmark | 1 |
| transfer_advisory | 1 |

## sessions 序列失败诊断（不直接造训练对）

- 命中序列失败：13 条
- 按问题类型：{'clarify_loop': 12, 'empty_action': 3}
- 按疑似根因：{'planner': 12, 'rag_judger': 1}

> 注：repair 对的 chosen 为合成修复，contrastive 对的 rejected 为合成反例；
> 建议按 confidence 降序人工抽检后再用于 DPO。clarify 循环若根因为 rag_judger，
> 应优先调整编排/judger 阈值，而非用 planner DPO 修复。
