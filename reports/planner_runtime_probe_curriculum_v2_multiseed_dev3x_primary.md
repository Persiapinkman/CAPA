# Runtime Probe Primary Multi-Seed Development (3x)

| Arm | Case macro | Action match | Score delta | Score 95% CI | Action delta | Action 95% CI | Conclusion |
|---|---:|---:|---:|---:|---:|---:|---|
| `sft_v3` | 0.646627 | 0.812500 | - | - | - | - | baseline |
| `seed42` | 0.876389 | 1.000000 | +0.229762 | [+0.139841, +0.344802] | +0.187500 | [+0.062500, +0.375000] | supported |
| `seed43` | 0.916071 | 1.000000 | +0.269444 | [+0.179524, +0.384484] | +0.187500 | [+0.062500, +0.375000] | supported |
| `seed44` | 0.791667 | 0.937500 | +0.145040 | [+0.055119, +0.245198] | +0.125000 | [+0.000000, +0.312500] | supported |
| `mean_policy` | 0.861376 | 0.979167 | +0.214749 | [+0.128135, +0.311468] | +0.166667 | [+0.041667, +0.333333] | supported |

## Mean Category Deltas

| Category | Delta |
|---|---:|
| `qwen_probe_then_migration` | +0.214749 |

Intervals use a paired bootstrap clustered by `entity_id`.
