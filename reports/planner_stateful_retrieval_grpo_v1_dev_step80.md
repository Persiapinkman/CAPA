# Planner Stateful Retrieval GRPO Step-80 Development Comparison

## Run Summary

| Run | Step mean | Case macro | Category macro | Action match | Exact pass | JSON valid | Extra text |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sft_v3` | 0.758697 | 0.822220 | 0.822220 | 0.687500 | 0.343750 | 1.000000 | 0.031250 |
| `grpo80` | 0.767528 | 0.828017 | 0.828017 | 0.703125 | 0.359375 | 1.000000 | 0.031250 |

## Paired Comparisons

| Comparison | Step delta [95% CI] | Case-macro delta [95% CI] | +/=/- steps | Conclusion |
|---|---:|---:|---:|---|
| `sft_v3_to_grpo80` | +0.008832 [+0.000000, +0.021060] | +0.005797 [+0.000000, +0.013406] | 2/62/0 | inconclusive |

Intervals use a paired bootstrap clustered by `entity_id`. Claim scope follows the registered study split.
