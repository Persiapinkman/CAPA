# Human Review: planner_rag_miss_action_reward_v1

## Review Order

1. Read `DATASET_CARD.md` and `manifest.json`.
2. Review trajectory labels in `planner_rag_miss_state_machine_v1/HUMAN_REVIEW.md`.
3. Verify every derived row has `reward_profile=action_dominant_v1`.
4. Verify the wrong-action task cap is `0.20` and outer format weight is only `0.05`.
5. Keep test predictions sealed until the registered confirmation trigger fires.

## Rejection Conditions

- A wrong action can receive task reward above `0.20`.
- Correct and incorrect actions have overlapping reward ranges on the historical sampling audit.
- Prompt selection weights differ from `planner_rag_miss_state_machine_v1`.
- A sampling replica is counted as an independent evaluation unit.
