# Planner Runtime Probe Curriculum v2 Seed-42 Development Comparison

## Run Summary

| Run | Step mean | Case macro | Category macro | Action match | Exact pass | JSON valid | Extra text |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sft_v3` | 0.671915 | 0.688895 | 0.688895 | 0.683333 | 0.391667 | 1.000000 | 0.000000 |
| `grpo42` | 0.794577 | 0.811416 | 0.811416 | 0.808333 | 0.575000 | 1.000000 | 0.000000 |

## Category Results

| Category | sft_v3 score | sft_v3 action | grpo42 score | grpo42 action |
|---|---:|---:|---:|---:|
| `adela_eval` | 0.920635 | 1.000000 | 0.990079 | 1.000000 |
| `clarify_incomplete` | 0.200000 | 0.000000 | 0.200000 | 0.000000 |
| `flux_generation` | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| `general_answer` | 0.083333 | 0.000000 | 0.523810 | 0.500000 |
| `memory_end` | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| `migration_advisor` | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| `pipeline_eval` | 0.458333 | 0.625000 | 0.494048 | 0.625000 |
| `private_lookup` | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| `qwen_probe` | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| `qwen_probe_only_contrast` | 0.970238 | 1.000000 | 0.970238 | 1.000000 |
| `qwen_probe_then_migration` | 0.646627 | 0.812500 | 0.876389 | 1.000000 |
| `rex_probe` | 0.200000 | 0.000000 | 1.000000 | 1.000000 |
| `rex_probe_then_migration` | 0.476468 | 0.500000 | 0.493849 | 0.500000 |

## Step Results

| Category step | sft_v3 score | sft_v3 action | grpo42 score | grpo42 action |
|---|---:|---:|---:|---:|
| `adela_eval#step1` | 0.920635 | 1.000000 | 0.990079 | 1.000000 |
| `clarify_incomplete#step1` | 0.200000 | 0.000000 | 0.200000 | 0.000000 |
| `flux_generation#step1` | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| `general_answer#step1` | 0.083333 | 0.000000 | 0.523810 | 0.500000 |
| `memory_end#step1` | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| `migration_advisor#step1` | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| `pipeline_eval#step1` | 0.458333 | 0.625000 | 0.494048 | 0.625000 |
| `private_lookup#step1` | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| `qwen_probe#step1` | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| `qwen_probe_only_contrast#step1` | 0.970238 | 1.000000 | 0.970238 | 1.000000 |
| `qwen_probe_then_migration#step1` | 0.575000 | 0.625000 | 0.975000 | 1.000000 |
| `qwen_probe_then_migration#step2` | 0.718254 | 1.000000 | 0.777778 | 1.000000 |
| `rex_probe#step1` | 0.200000 | 0.000000 | 1.000000 | 1.000000 |
| `rex_probe_then_migration#step1` | 0.195000 | 0.000000 | 0.200000 | 0.000000 |
| `rex_probe_then_migration#step2` | 0.757937 | 1.000000 | 0.787698 | 1.000000 |

## Paired Comparisons

| Comparison | Step delta [95% CI] | Case-macro delta [95% CI] | Action delta [95% CI] | +/=/- steps | Conclusion |
|---|---:|---:|---:|---:|---|
| `sft_v3_to_grpo42` | +0.122661 [+0.104630, +0.139921] | +0.122521 [+0.102592, +0.141725] | +0.125000 [+0.100000, +0.150000] | 41/79/0 | supported |

Intervals use a paired bootstrap clustered by `entity_id`. Claim scope follows the registered study split.
