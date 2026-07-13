# Complex Retrieval GRPO v2 Multi-Seed Test Double-Miss Endpoint

| Arm | Case macro | Action match | Score delta | Score 95% CI | Action delta | Action 95% CI | Conclusion |
|---|---:|---:|---:|---:|---:|---:|---|
| `sft_v3` | 0.697219 | 0.650000 | - | - | - | - | baseline |
| `seed42` | 0.787938 | 0.756250 | +0.090719 | [+0.045656, +0.138344] | +0.106250 | [+0.056250, +0.162500] | supported |
| `seed43` | 0.806188 | 0.775000 | +0.108969 | [+0.054594, +0.169313] | +0.125000 | [+0.062500, +0.193750] | supported |
| `seed44` | 0.826344 | 0.806250 | +0.129125 | [+0.077750, +0.186188] | +0.156250 | [+0.093750, +0.225000] | supported |
| `mean_policy` | 0.806823 | 0.779167 | +0.109604 | [+0.063219, +0.160938] | +0.129167 | [+0.075000, +0.189583] | supported |

## Mean Category Deltas

| Category | Delta |
|---|---:|
| `rag_double_miss_recovery` | +0.109604 |

Intervals use a paired bootstrap clustered by `entity_id`.
