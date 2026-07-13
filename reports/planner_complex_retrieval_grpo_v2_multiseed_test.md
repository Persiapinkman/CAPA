# Complex Retrieval GRPO v2 Multi-Seed Test Confirmation

| Arm | Case macro | Action match | Score delta | Score 95% CI | Action delta | Action 95% CI | Conclusion |
|---|---:|---:|---:|---:|---:|---:|---|
| `sft_v3` | 0.755749 | 0.708333 | - | - | - | - | baseline |
| `seed42` | 0.771841 | 0.743750 | +0.016092 | [+0.000765, +0.031711] | +0.035417 | [+0.006250, +0.064583] | supported |
| `seed43` | 0.774906 | 0.747917 | +0.019157 | [+0.005740, +0.033956] | +0.039583 | [+0.012500, +0.068750] | supported |
| `seed44` | 0.786509 | 0.770833 | +0.030760 | [+0.014032, +0.047865] | +0.062500 | [+0.029167, +0.097917] | supported |
| `mean_policy` | 0.777752 | 0.754167 | +0.022003 | [+0.008292, +0.036338] | +0.045833 | [+0.018750, +0.075000] | supported |

## Mean Category Deltas

| Category | Delta |
|---|---:|
| `coref_rewrite_then_rag` | -0.016484 |
| `direct_rag_guardrail` | +0.000000 |
| `general_answer_guardrail` | -0.001563 |
| `memory_hit_end_guardrail` | +0.000000 |
| `rag_double_miss_recovery` | +0.109604 |
| `rag_hit_then_synthesize` | +0.026875 |
| `rag_single_miss_recovery` | +0.035590 |

Intervals use a paired bootstrap clustered by `entity_id`.
