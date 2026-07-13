# Planner Stateful Retrieval GRPO Step-40 Development Comparison

## Run Summary

| Run | Step mean | Case macro | Category macro | Action match | Exact pass | JSON valid | Extra text |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sft_v3` | 0.758697 | 0.822220 | 0.822220 | 0.687500 | 0.343750 | 1.000000 | 0.031250 |
| `grpo40` | 0.751903 | 0.819684 | 0.819684 | 0.687500 | 0.343750 | 0.984375 | 0.046875 |

## Paired Comparisons

| Comparison | Step delta [95% CI] | Case-macro delta [95% CI] | +/=/- steps | Conclusion |
|---|---:|---:|---:|---|
| `sft_v3_to_grpo40` | -0.006793 [-0.042799, +0.018342] | -0.002536 [-0.022464, +0.012319] | 2/61/1 | inconclusive |

Intervals use a paired bootstrap clustered by `entity_id`. Claim scope follows the registered study split.
