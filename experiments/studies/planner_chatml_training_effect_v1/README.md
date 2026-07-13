# Planner ChatML Training-Effect Study

## Purpose

Separate the effects of prompt format, SFT warmup, and GRPO on the Qwen2.5-7B Planner. The missing control is the untrained base model evaluated with the same native ChatML prompt used by SFTv3 and GRPOv4.

## Design

| Arm | Prompt | Training state | Comparison role |
|---|---|---|---|
| `base_pseudo` | pseudo tags | untrained base | historical prompt control |
| `base_chatml` | native ChatML | untrained base | prompt-corrected control |
| `sft_v3_chatml` | native ChatML | SFTv3 | SFT effect |
| `grpo_v4_chatml` | native ChatML | SFTv3 + GRPOv4 | GRPO effect |

All arms use greedy generation, seed 42, 128 maximum new tokens, and three repeats. Comparisons are paired and bootstrap-clustered by `case_id`.

## Decision Rule

The case-macro 95% interval must exclude zero before calling an effect supported. Results remain development evidence because this split has already influenced model selection.

## Result

- Native ChatML is retained because raw tail-text failures changed from 49/49 to 0/49 in every repeat; its routing-score change is inconclusive.
- SFTv3 versus Base ChatML: case-macro delta `+0.001503`, 95% CI `[-0.006674, +0.013408]`.
- GRPOv4 versus SFTv3: case-macro delta `-0.001390`, 95% CI `[-0.020857, +0.010011]`.
- Neither SFTv3 nor GRPOv4 is promoted as a quality improvement.

Full tables are in `reports/planner_chatml_training_effect_v1.md`.
