# Qwen2.5 ChatML SFTv3 vs GRPOv4 Smoke

Eval file: `training/planner_grpo_seed_v1/sft_data_v3_chatml/val.jsonl`

| model | mean_score | delta_vs_sft_v3 | json_valid | extra_text_rate | mean_extra_chars |
|---|---:|---:|---:|---:|---:|
| sft_v3_chatml | 0.8564 | +0.0000 | 1.0 | 0.0 | 0.0 |
| grpo_v4_chatml_format_smoke | 0.8578 | +0.0014 | 1.0 | 0.0 | 0.0 |

Conclusion:

- The previous stopping failure was primarily prompt/template mismatch plus missing explicit ChatML EOS in SFT targets.
- Correct path: build ChatML SFT data, train SFTv3, merge the SFT adapter, then run GRPO LoRA with ChatML prompts and combined task/format reward.
- GRPOv4 smoke is not a final quality run, but it validates that rollout termination is fixed: training clipped_ratio was 0.0 for most observed steps and eval extra_text_rate is 0.0.
