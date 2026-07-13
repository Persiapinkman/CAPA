# Planner complex retrieval GRPO v2 development screen (seed 42)

## Run Summary

| Run | Step mean | Case macro | Category macro | Action match | Exact pass | JSON valid | Extra text |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sft_v3` | 0.803521 | 0.845781 | 0.845781 | 0.795833 | 0.362500 | 1.000000 | 0.000000 |
| `grpo42` | 0.836458 | 0.865126 | 0.865126 | 0.833333 | 0.395833 | 1.000000 | 0.000000 |

## Category Results

| Category | sft_v3 score | sft_v3 action | grpo42 score | grpo42 action |
|---|---:|---:|---:|---:|
| `coref_rewrite_then_rag` | 0.672656 | 0.656250 | 0.674219 | 0.656250 |
| `direct_rag_guardrail` | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| `general_answer_guardrail` | 0.950000 | 1.000000 | 0.950000 | 1.000000 |
| `memory_hit_end_guardrail` | 0.968750 | 1.000000 | 0.968750 | 1.000000 |
| `rag_double_miss_recovery` | 0.795625 | 0.762500 | 0.855625 | 0.837500 |
| `rag_hit_then_synthesize` | 0.789687 | 0.812500 | 0.820312 | 0.843750 |
| `rag_single_miss_recovery` | 0.743750 | 0.729167 | 0.786979 | 0.770833 |

## Step Results

| Category step | sft_v3 score | sft_v3 action | grpo42 score | grpo42 action |
|---|---:|---:|---:|---:|
| `coref_rewrite_then_rag#step1` | 0.379688 | 0.312500 | 0.392188 | 0.312500 |
| `coref_rewrite_then_rag#step2` | 0.965625 | 1.000000 | 0.956250 | 1.000000 |
| `direct_rag_guardrail#step1` | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| `general_answer_guardrail#step1` | 0.950000 | 1.000000 | 0.950000 | 1.000000 |
| `memory_hit_end_guardrail#step1` | 0.968750 | 1.000000 | 0.968750 | 1.000000 |
| `rag_double_miss_recovery#step1` | 0.650000 | 0.562500 | 1.000000 | 1.000000 |
| `rag_double_miss_recovery#step2` | 0.425000 | 0.312500 | 0.425000 | 0.312500 |
| `rag_double_miss_recovery#step3` | 0.950000 | 0.937500 | 0.900000 | 0.875000 |
| `rag_double_miss_recovery#step4` | 0.975000 | 1.000000 | 0.975000 | 1.000000 |
| `rag_double_miss_recovery#step5` | 0.978125 | 1.000000 | 0.978125 | 1.000000 |
| `rag_hit_then_synthesize#step1` | 0.721875 | 0.687500 | 0.725000 | 0.687500 |
| `rag_hit_then_synthesize#step2` | 0.857500 | 0.937500 | 0.915625 | 1.000000 |
| `rag_single_miss_recovery#step1` | 0.981250 | 1.000000 | 0.990625 | 1.000000 |
| `rag_single_miss_recovery#step2` | 0.300000 | 0.187500 | 0.420313 | 0.312500 |
| `rag_single_miss_recovery#step3` | 0.950000 | 1.000000 | 0.950000 | 1.000000 |

## Paired Comparisons

| Comparison | Step delta [95% CI] | Case-macro delta [95% CI] | Action delta [95% CI] | +/=/- steps | Conclusion |
|---|---:|---:|---:|---:|---|
| `sft_v3_to_grpo42` | +0.032938 [+0.019167, +0.046521] | +0.019345 [+0.010476, +0.029122] | +0.037500 [+0.020833, +0.054167] | 27/207/6 | supported |

Intervals use a paired bootstrap clustered by `entity_id`. Claim scope follows the registered study split.
