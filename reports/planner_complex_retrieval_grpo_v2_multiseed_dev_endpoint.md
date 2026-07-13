# Complex Retrieval GRPO v2 Multi-Seed Double-Miss Endpoint

| Arm | Case macro | Action match | Score delta | Score 95% CI | Action delta | Action 95% CI | Conclusion |
|---|---:|---:|---:|---:|---:|---:|---|
| `sft_v3` | 0.795625 | 0.762500 | - | - | - | - | baseline |
| `seed42` | 0.855625 | 0.837500 | +0.060000 | [+0.020000, +0.100000] | +0.075000 | [+0.025000, +0.125000] | supported |
| `seed43` | 0.810000 | 0.775000 | +0.014375 | [+0.002500, +0.036250] | +0.012500 | [+0.000000, +0.037500] | supported |
| `seed44` | 0.907500 | 0.900000 | +0.111875 | [+0.055000, +0.170313] | +0.137500 | [+0.062500, +0.212500] | supported |
| `mean_policy` | 0.857708 | 0.837500 | +0.062083 | [+0.030104, +0.096458] | +0.075000 | [+0.033333, +0.120833] | supported |

## Mean Category Deltas

| Category | Delta |
|---|---:|
| `rag_double_miss_recovery` | +0.062083 |

Intervals use a paired bootstrap clustered by `entity_id`.
