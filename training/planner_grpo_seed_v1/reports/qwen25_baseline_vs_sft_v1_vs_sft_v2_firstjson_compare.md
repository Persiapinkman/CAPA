# Qwen2.5 First-JSON Stop-Aware Eval Compare

Eval file: `training/planner_grpo_seed_v1/sft_data/val.jsonl`

| model | mean_score | delta_vs_baseline | json_valid | raw_extra_rate | raw_extra_chars | effective_extra_rate |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.8454 | +0.0000 | 1.0 | 1.0 | 160.96 | 0.0 |
| sft_v1 | 0.8580 | +0.0126 | 1.0 | 1.0 | 157.14 | 0.0 |
| sft_v2 | 0.8495 | +0.0040 | 1.0 | 1.0 | 131.47 | 0.0 |

Conclusion:

- First-complete-JSON postprocessing fixes execution/scoring tail pollution, not model stopping.
- SFT v1 is slightly best by first-JSON score.
- SFT v2 still has the shortest raw tail, but remains at raw extra text rate 1.0.
- Do not run long GRPO with first-JSON reward alone; add an explicit stopping objective or rollout postprocessor before treating GRPO v2 as formal quality training.
