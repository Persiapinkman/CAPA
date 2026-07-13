# Planner five-step double-miss endpoint (seed 42)

## Run Summary

| Run | Step mean | Case macro | Category macro | Action match | Exact pass | JSON valid | Extra text |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sft_v3` | 0.795625 | 0.795625 | 0.795625 | 0.762500 | 0.412500 | 1.000000 | 0.000000 |
| `grpo42` | 0.855625 | 0.855625 | 0.855625 | 0.837500 | 0.487500 | 1.000000 | 0.000000 |

## Category Results

| Category | sft_v3 score | sft_v3 action | grpo42 score | grpo42 action |
|---|---:|---:|---:|---:|
| `rag_double_miss_recovery` | 0.795625 | 0.762500 | 0.855625 | 0.837500 |

## Step Results

| Category step | sft_v3 score | sft_v3 action | grpo42 score | grpo42 action |
|---|---:|---:|---:|---:|
| `rag_double_miss_recovery#step1` | 0.650000 | 0.562500 | 1.000000 | 1.000000 |
| `rag_double_miss_recovery#step2` | 0.425000 | 0.312500 | 0.425000 | 0.312500 |
| `rag_double_miss_recovery#step3` | 0.950000 | 0.937500 | 0.900000 | 0.875000 |
| `rag_double_miss_recovery#step4` | 0.975000 | 1.000000 | 0.975000 | 1.000000 |
| `rag_double_miss_recovery#step5` | 0.978125 | 1.000000 | 0.978125 | 1.000000 |

## Paired Comparisons

| Comparison | Step delta [95% CI] | Case-macro delta [95% CI] | Action delta [95% CI] | +/=/- steps | Conclusion |
|---|---:|---:|---:|---:|---|
| `sft_v3_to_grpo42` | +0.060000 [+0.020000, +0.100000] | +0.060000 [+0.020000, +0.100000] | +0.075000 [+0.025000, +0.125000] | 7/72/1 | supported |

Intervals use a paired bootstrap clustered by `entity_id`. Claim scope follows the registered study split.
