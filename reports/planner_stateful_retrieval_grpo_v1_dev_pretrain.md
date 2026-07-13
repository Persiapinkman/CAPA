# Planner Stateful Retrieval GRPO Study: Pretraining Baselines

## Run Summary

| Run | Step mean | Case macro | Category macro | Exact pass | JSON valid | Extra text |
|---|---:|---:|---:|---:|---:|---:|
| `base` | 0.791236 | 0.837627 | 0.837627 | 0.390625 | 1.000000 | 0.000000 |
| `sft_v3` | 0.758697 | 0.822220 | 0.822220 | 0.343750 | 1.000000 | 0.031250 |

## Paired Comparisons

| Comparison | Step delta [95% CI] | Case-macro delta [95% CI] | +/=/- steps | Conclusion |
|---|---:|---:|---:|---|
| `base_to_sft_v3` | -0.032539 [-0.057439, -0.002951] | -0.015407 [-0.026766, -0.000587] | 3/53/8 | regressed |

Intervals use a paired bootstrap clustered by `entity_id`. Claim scope follows the registered study split.
