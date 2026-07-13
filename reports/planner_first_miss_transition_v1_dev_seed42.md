# Planner first-miss transition GRPO development screen (seed 42)

## Run Summary

| Run | Step mean | Case macro | Category macro | Action match | Exact pass | JSON valid | Extra text |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sft_v3` | 0.803521 | 0.845781 | 0.845781 | 0.795833 | 0.362500 | 1.000000 | 0.000000 |
| `focus42` | 0.844792 | 0.873810 | 0.873810 | 0.841667 | 0.404167 | 1.000000 | 0.000000 |

## Category Results

| Category | sft_v3 score | sft_v3 action | focus42 score | focus42 action |
|---|---:|---:|---:|---:|
| `coref_rewrite_then_rag` | 0.672656 | 0.656250 | 0.676563 | 0.656250 |
| `direct_rag_guardrail` | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| `general_answer_guardrail` | 0.950000 | 1.000000 | 0.950000 | 1.000000 |
| `memory_hit_end_guardrail` | 0.968750 | 1.000000 | 0.968750 | 1.000000 |
| `rag_double_miss_recovery` | 0.795625 | 0.762500 | 0.828125 | 0.800000 |
| `rag_hit_then_synthesize` | 0.789687 | 0.812500 | 0.820312 | 0.843750 |
| `rag_single_miss_recovery` | 0.743750 | 0.729167 | 0.872917 | 0.875000 |

## Step Results

| Category step | sft_v3 score | sft_v3 action | focus42 score | focus42 action |
|---|---:|---:|---:|---:|
| `coref_rewrite_then_rag#step1` | 0.379688 | 0.312500 | 0.393750 | 0.312500 |
| `coref_rewrite_then_rag#step2` | 0.965625 | 1.000000 | 0.959375 | 1.000000 |
| `direct_rag_guardrail#step1` | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| `general_answer_guardrail#step1` | 0.950000 | 1.000000 | 0.950000 | 1.000000 |
| `memory_hit_end_guardrail#step1` | 0.968750 | 1.000000 | 0.968750 | 1.000000 |
| `rag_double_miss_recovery#step1` | 0.650000 | 0.562500 | 0.650000 | 0.562500 |
| `rag_double_miss_recovery#step2` | 0.425000 | 0.312500 | 0.525000 | 0.437500 |
| `rag_double_miss_recovery#step3` | 0.950000 | 0.937500 | 1.000000 | 1.000000 |
| `rag_double_miss_recovery#step4` | 0.975000 | 1.000000 | 0.975000 | 1.000000 |
| `rag_double_miss_recovery#step5` | 0.978125 | 1.000000 | 0.990625 | 1.000000 |
| `rag_hit_then_synthesize#step1` | 0.721875 | 0.687500 | 0.725000 | 0.687500 |
| `rag_hit_then_synthesize#step2` | 0.857500 | 0.937500 | 0.915625 | 1.000000 |
| `rag_single_miss_recovery#step1` | 0.981250 | 1.000000 | 0.993750 | 1.000000 |
| `rag_single_miss_recovery#step2` | 0.300000 | 0.187500 | 0.671875 | 0.625000 |
| `rag_single_miss_recovery#step3` | 0.950000 | 1.000000 | 0.953125 | 1.000000 |

## Paired Comparisons

| Comparison | Step delta [95% CI] | Case-macro delta [95% CI] | Action delta [95% CI] | +/=/- steps | Conclusion |
|---|---:|---:|---:|---:|---|
| `sft_v3_to_focus42` | +0.041271 [+0.023646, +0.060292] | +0.028028 [+0.015394, +0.042783] | +0.045833 [+0.025000, +0.066667] | 29/208/3 | supported |

Intervals use a paired bootstrap clustered by `entity_id`. Claim scope follows the registered study split.
