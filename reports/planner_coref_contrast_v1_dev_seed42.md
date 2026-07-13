# Planner Coreference-Contrast GRPO Seed-42 Development Comparison

## Run Summary

| Run | Step mean | Case macro | Category macro | Action match | Exact pass | JSON valid | Extra text |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sft_v3` | 0.758697 | 0.822220 | 0.822220 | 0.687500 | 0.343750 | 1.000000 | 0.031250 |
| `coref42` | 0.747148 | 0.817147 | 0.817147 | 0.671875 | 0.328125 | 0.984375 | 0.046875 |

## Paired Comparisons

| Comparison | Step delta [95% CI] | Case-macro delta [95% CI] | +/=/- steps | Conclusion |
|---|---:|---:|---:|---|
| `sft_v3_to_coref42` | -0.011549 [-0.046875, +0.012228] | -0.005072 [-0.025000, +0.009783] | 1/62/1 | inconclusive |

Intervals use a paired bootstrap clustered by `entity_id`. Claim scope follows the registered study split.
