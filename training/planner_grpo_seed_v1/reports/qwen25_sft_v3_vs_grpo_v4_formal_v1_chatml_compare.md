# Qwen2.5 ChatML SFTv3 vs GRPOv4 Formal-v1

Train cases: `training/planner_grpo_seed_v1/cases/planner_grpo_focused_train_v3_cases.jsonl`
Eval file: `training/planner_grpo_seed_v1/sft_data_v3_chatml/val.jsonl`

| model | mean_score | delta_vs_sft_v3 | json_valid | extra_text_rate | mean_extra_chars |
|---|---:|---:|---:|---:|---:|
| sft_v3_chatml | 0.8564 | +0.0000 | 1.0 | 0.0 | 0.0 |
| grpo_v4_smoke_8step | 0.8578 | +0.0014 | 1.0 | 0.0 | 0.0 |
| grpo_v4_formal_v1_60step | 0.8597 | +0.0033 | 1.0 | 0.0 | 0.0 |

Conclusion:

- GRPOv4 formal-v1 is trained only on the SFTv3 train-case split, then evaluated on the held-out SFTv3 val split.
- Stopping remains fixed: all compared ChatML models have extra_text_rate 0.0 on this eval.
- Formal-v1 improves mean score slightly over SFTv3 (+0.0033) and the 8-step smoke (+0.0019).
- The gain is modest; the run mostly validates the stable GRPO training path rather than proving broad generalization.
