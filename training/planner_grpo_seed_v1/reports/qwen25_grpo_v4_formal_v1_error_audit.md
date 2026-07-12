# Planner GRPO Eval Audit

- Baseline: `qwen25_sft_v3_chatml`
- Candidate: `qwen25_grpo_v4_formal_v1`
- Compared rows: 49
- Mean score delta: 0.003343

## Category Delta

| Category | N | Baseline | Candidate | Delta | Low<0.8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| clarify_intent_ambiguity | 2 | 0.100000 | 0.100000 | 0.000000 | 2 |
| full_detection_eval | 1 | 1.000000 | 0.750000 | -0.250000 | 1 |
| general_answer | 1 | 0.900000 | 0.900000 | 0.000000 | 0 |
| historical_asset_qa | 1 | 1.000000 | 1.000000 | 0.000000 | 0 |
| probe_only_contrastive | 6 | 1.000000 | 1.000000 | 0.000000 | 0 |
| probe_then_migration | 24 | 0.848334 | 0.862702 | 0.014368 | 12 |
| probe_then_migration_strict | 12 | 0.875205 | 0.880952 | 0.005747 | 6 |
| single_image_probe | 2 | 1.000000 | 1.000000 | 0.000000 | 0 |

## Main Regressions

- `GRPO-EXP-BOUNDARY-005` step 1 `full_detection_eval`: 1.0000 -> 0.7500; expected `pipeline_eval`, got `pipeline_eval`; failures: arg 'task_text' expected to contain ['检测'], got 'person_with_bag.png'

## Persistent Low Scores

- `GRPO-EXP-CLARIFY-004` step 1 `clarify_intent_ambiguity`: 0.1000; expected `clarify`, got `flux-image-generation`; forbidden_hit=True; sample: { "decision_type": "tool", "action": "flux-image-generation", "action_input": { "task_text": "处理烟雾的图片", "source_image_required": true, "num_images": 1, "finish_after_tool": fals...
- `GRPO-EXP-CLARIFY-005` step 1 `clarify_intent_ambiguity`: 0.1000; expected `clarify`, got `flux-image-generation`; forbidden_hit=True; sample: { "decision_type": "tool", "action": "flux-image-generation", "action_input": { "task_text": "生成一张类似于 banner.jpg 的横幅图片", "source_image_required": false, "num_images": 1, "finish...
- `GRPO-OBS-001` step 2 `probe_then_migration`: 0.3929; expected `migration_advisor`, got `re_question`; forbidden_hit=False; sample: { "decision_type": "tool", "thought": "当前检测结果置信度较低，需要进一步验证。", "action": "re_question", "action_input": { "query": "这张图里有没有钓鱼的人？先用已有模型试一下。如果结果不确定，再给客户一个低成本验证方案。", "rewrite_reason...
- `GRPO-EXP-BOUNDARY-005` step 1 `full_detection_eval`: 0.7500; expected `pipeline_eval`, got `pipeline_eval`; forbidden_hit=False; sample: { "decision_type": "tool", "action": "pipeline_eval", "action_input": { "task_text": "person_with_bag.png", "finish_after_tool": true }, "final_answer": "" }
- `GRPO-EXP-PROBE-MIG-001` step 2 `probe_then_migration`: 0.7619; expected `migration_advisor`, got `migration_advisor`; forbidden_hit=False; sample: { "decision_type": "tool", "action": "migration_advisor", "action_input": { "user_query": "这张图里有没有钓鱼的人？先用已有模型试一下。如果不确定，再给客户一个钓鱼人员检测的低成本验证方案。", "use_image": true, "use_visual_pro...
- `GRPO-EXP-PROBE-MIG-003` step 2 `probe_then_migration`: 0.7619; expected `migration_advisor`, got `migration_advisor`; forbidden_hit=False; sample: { "decision_type": "tool", "action": "migration_advisor", "action_input": { "user_query": "这张图里有没有反光条？先用已有模型试一下。如果不确定，再给客户一个工服反光条检测的低成本验证方案。", "use_image": true, "use_visual_pro...
- `GRPO-EXP-PROBE-MIG-009` step 2 `probe_then_migration`: 0.7619; expected `migration_advisor`, got `migration_advisor`; forbidden_hit=False; sample: { "decision_type": "tool", "thought": "根据上一步的观察，开放集检测结果不稳定，需要结合已有资产做低成本可行性判断。", "action": "migration_advisor", "action_input": { "user_query": "客户给了这张样例图，先检测有没有反光条；如果结果不稳，再判断工服反...
- `GRPO-EXP-PROBE-MIG-011` step 2 `probe_then_migration`: 0.7619; expected `migration_advisor`, got `migration_advisor`; forbidden_hit=False; sample: { "decision_type": "tool", "thought": "根据上一步的观察，开放集检测结果不稳定，需要结合已有资产做低成本可行性判断。", "action": "migration_advisor", "action_input": { "user_query": "客户给了这张样例图，先检测有没有横幅；如果结果不稳，再判断横幅违规...
- `GRPO-EXP-PROBE-MIG-015` step 2 `probe_then_migration`: 0.7619; expected `migration_advisor`, got `migration_advisor`; forbidden_hit=False; sample: { "decision_type": "tool", "thought": "根据上一步的观察结果，需要进一步评估工服反光条检测的可行性。", "action": "migration_advisor", "action_input": { "user_query": "请先对图里的反光条做快速探针，探针结果不确定的话，再给一个工服反光条检测的可行性方...
- `GRPO-EXP-PROBE-MIG-016` step 2 `probe_then_migration`: 0.7619; expected `migration_advisor`, got `migration_advisor`; forbidden_hit=False; sample: { "decision_type": "tool", "action": "migration_advisor", "action_input": { "user_query": "请先对图里的烟雾做快速探针，探针结果不确定的话，再给一个烟雾异常检测的可行性方案。", "use_image": true, "use_visual_probe": fal...
- `GRPO-EXP-PROBE-MIG-026` step 2 `probe_then_migration`: 0.7619; expected `migration_advisor`, got `migration_advisor`; forbidden_hit=False; sample: { "decision_type": "tool", "action": "migration_advisor", "action_input": { "user_query": "这张图作为样例，先用开放集模型试厨师帽；如果信心不足，再输出后厨帽子佩戴检测迁移建议。", "use_image": true, "use_visual_probe": f...
- `GRPO-EXP-PROBE-MIG-030` step 2 `probe_then_migration`: 0.7619; expected `migration_advisor`, got `migration_advisor`; forbidden_hit=False; sample: { "decision_type": "tool", "action": "migration_advisor", "action_input": { "user_query": "这张图作为样例，先用开放集模型试垃圾车；如果信心不足，再输出垃圾车识别迁移建议。", "use_image": true, "use_visual_probe": fals...
- `GRPO-EXP-PROBE-MIG-047` step 2 `probe_then_migration`: 0.7619; expected `migration_advisor`, got `migration_advisor`; forbidden_hit=False; sample: { "decision_type": "tool", "action": "migration_advisor", "action_input": { "user_query": "这不是完整评测，先检查横幅；如果检测没把握，再转成横幅违规悬挂检测迁移评估。", "use_image": true, "use_visual_probe": false,...
- `GRPO-EXP-PROBE-MIG-052` step 2 `probe_then_migration`: 0.7619; expected `migration_advisor`, got `migration_advisor`; forbidden_hit=False; sample: { "decision_type": "tool", "thought": "根据上一步 open-set 检测结果不稳定，需要结合已有资产做低成本可行性判断。", "action": "migration_advisor", "action_input": { "user_query": "先跑单图检测看烟雾，再根据结果判断是否需要给客户烟雾异常检测...
- `GRPO-EXP-PROBE-MIG-054` step 2 `probe_then_migration`: 0.7619; expected `migration_advisor`, got `migration_advisor`; forbidden_hit=False; sample: { "decision_type": "tool", "thought": "根据上一步 open-set 检测结果不稳定，需要结合已有资产做低成本可行性判断。", "action": "migration_advisor", "action_input": { "user_query": "先跑单图检测看垃圾车，再根据结果判断是否需要给客户垃圾车识别...

## Action Confusions

- `answerer -> answerer`: 1
- `clarify -> flux-image-generation`: 2
- `migration_advisor -> migration_advisor`: 17
- `migration_advisor -> re_question`: 1
- `pipeline_eval -> pipeline_eval`: 1
- `qwen_detection -> qwen_detection`: 26
- `rag_answer -> rag_answer`: 1

## Recommendation

- Do not extend GRPO v4 blindly: clarify samples show no action-level improvement, so equal wrong rewards can yield no GRPO advantage.
- Create a hard-case refresh path that first raises clarify/pipeline parameter behavior with SFT or high-exploration GRPO, then run GRPO on a balanced hard subset.
- Track parameter regressions separately from action regressions; full_detection_eval kept pipeline_eval but degraded task_text quality.
