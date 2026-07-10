# Qwen2.5 Baseline vs SFT v1 vs SFT v2 on SFT Val

| Run | Adapter | Mean Score | JSON Valid | Extra Text Rate | Mean Extra Chars |
|---|---|---:|---:|---:|---:|
| Baseline | `none` | 0.6407 | 1.000 | 1.000 | 160.96 |
| SFT v1 | `outputs/planner-sft-qwen25-7b-trl-lora-warmup-v1` | 0.6173 | 1.000 | 1.000 | 157.14 |
| SFT v2 | `outputs/planner-sft-qwen25-7b-trl-lora-warmup-v2` | 0.6931 | 1.000 | 1.000 | 131.47 |

SFT v1 delta vs baseline: -0.0234 mean score.
SFT v2 delta vs baseline: 0.0524 mean score.
SFT v2 delta vs SFT v1: 0.0758 mean score.

SFT v2 improves score and reduces extra text length, but does not solve stopping: extra text after the first JSON remains 100%.
