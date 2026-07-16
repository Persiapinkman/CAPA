# Runtime Probe Curriculum Multi-Seed Development (3x)

| Arm | Case macro | Action match | Score delta | Score 95% CI | Action delta | Action 95% CI | Conclusion |
|---|---:|---:|---:|---:|---:|---:|---|
| `sft_v3` | 0.688895 | 0.683333 | - | - | - | - | baseline |
| `seed42` | 0.811416 | 0.808333 | +0.122521 | [+0.102592, +0.141725] | +0.125000 | [+0.100000, +0.150000] | supported |
| `seed43` | 0.830220 | 0.850000 | +0.141325 | [+0.120229, +0.161810] | +0.166667 | [+0.125000, +0.208333] | supported |
| `seed44` | 0.771499 | 0.791667 | +0.082604 | [+0.044496, +0.125418] | +0.108333 | [+0.033333, +0.183333] | supported |
| `mean_policy` | 0.804378 | 0.816667 | +0.115483 | [+0.093791, +0.137630] | +0.133333 | [+0.091667, +0.177778] | supported |

## Mean Category Deltas

| Category | Delta |
|---|---:|
| `adela_eval` | +0.059524 |
| `clarify_incomplete` | +0.000000 |
| `flux_generation` | +0.000000 |
| `general_answer` | +0.311508 |
| `memory_end` | +0.000000 |
| `migration_advisor` | -0.005952 |
| `pipeline_eval` | -0.047619 |
| `private_lookup` | +0.000000 |
| `qwen_probe` | +0.000000 |
| `qwen_probe_only_contrast` | +0.000000 |
| `qwen_probe_then_migration` | +0.214749 |
| `rex_probe` | +0.800000 |
| `rex_probe_then_migration` | +0.169074 |

Intervals use a paired bootstrap clustered by `entity_id`.
