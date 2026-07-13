# Complex Retrieval GRPO v2 Multi-Seed Development

| Arm | Case macro | Action match | Score delta | Score 95% CI | Action delta | Action 95% CI | Conclusion |
|---|---:|---:|---:|---:|---:|---:|---|
| `sft_v3` | 0.845781 | 0.795833 | - | - | - | - | baseline |
| `seed42` | 0.865126 | 0.833333 | +0.019345 | [+0.010476, +0.029122] | +0.037500 | [+0.020833, +0.054167] | supported |
| `seed43` | 0.859985 | 0.812500 | +0.014204 | [+0.005647, +0.024621] | +0.016667 | [+0.004167, +0.033333] | supported |
| `seed44` | 0.877597 | 0.862500 | +0.031815 | [+0.020022, +0.044606] | +0.066667 | [+0.037500, +0.095833] | supported |
| `mean_policy` | 0.867569 | 0.836111 | +0.021788 | [+0.013673, +0.030873] | +0.040278 | [+0.025000, +0.055556] | supported |

## Mean Category Deltas

| Category | Delta |
|---|---:|
| `coref_rewrite_then_rag` | +0.004948 |
| `direct_rag_guardrail` | +0.000000 |
| `general_answer_guardrail` | +0.000000 |
| `memory_hit_end_guardrail` | +0.000000 |
| `rag_double_miss_recovery` | +0.062083 |
| `rag_hit_then_synthesize` | +0.032188 |
| `rag_single_miss_recovery` | +0.053299 |

Intervals use a paired bootstrap clustered by `entity_id`.
