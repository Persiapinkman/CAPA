# Planner ChatML Training-Effect Study

## Run Summary

| Run | Step mean | Case macro | Category macro | Exact pass | JSON valid | Extra text |
|---|---:|---:|---:|---:|---:|---:|
| `base_pseudo` | 0.845436 | 0.847199 | 0.835715 | 0.530612 | 1.000000 | 1.000000 |
| `base_chatml` | 0.854475 | 0.854343 | 0.839957 | 0.510204 | 1.000000 | 0.000000 |
| `sft_v3_chatml` | 0.856377 | 0.855846 | 0.840442 | 0.428571 | 1.000000 | 0.000000 |
| `grpo_v4_chatml` | 0.859720 | 0.854456 | 0.811707 | 0.530612 | 1.000000 | 0.000000 |

## Paired Comparisons

| Comparison | Step delta [95% CI] | Case-macro delta [95% CI] | +/=/- steps | Conclusion |
|---|---:|---:|---:|---|
| `base_pseudo_to_base_chatml` | +0.009040 [-0.018333, +0.038056] | +0.007144 [-0.014745, +0.030265] | 3/41/5 | inconclusive |
| `base_chatml_to_sft_v3_chatml` | +0.001902 [-0.007958, +0.016940] | +0.001503 [-0.006674, +0.013408] | 1/44/4 | inconclusive |
| `sft_v3_chatml_to_grpo_v4_chatml` | +0.003343 [-0.010776, +0.012931] | -0.001390 [-0.020857, +0.010011] | 6/42/1 | inconclusive |

Intervals use a paired bootstrap clustered by `case_id`. This is a development-set study, not a final generalization claim.
