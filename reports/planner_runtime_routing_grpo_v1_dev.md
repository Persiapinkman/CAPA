# Planner Runtime Routing GRPO v1 Development Comparison

## Run Summary

| Run | Step mean | Case macro | Category macro | Action match | Exact pass | JSON valid | Extra text |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sft_v3` | 0.768611 | 0.773157 | 0.773157 | 0.683333 | 0.391667 | 1.000000 | 0.000000 |
| `route_grpo_v4` | 0.781667 | 0.784295 | 0.784295 | 0.691667 | 0.408333 | 1.000000 | 0.000000 |

## Category Results

| Category | sft_v3 score | sft_v3 action | route_grpo_v4 score | route_grpo_v4 action |
|---|---:|---:|---:|---:|
| `adela_eval` | 0.916667 | 1.000000 | 0.916667 | 1.000000 |
| `clarify_incomplete` | 0.200000 | 0.000000 | 0.200000 | 0.000000 |
| `flux_generation` | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| `general_answer` | 0.075000 | 0.000000 | 0.181250 | 0.125000 |
| `memory_end` | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| `migration_advisor` | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| `pipeline_eval` | 0.462500 | 0.625000 | 0.450000 | 0.625000 |
| `private_lookup` | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| `qwen_probe` | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| `qwen_probe_only_contrast` | 0.968750 | 1.000000 | 0.968750 | 1.000000 |
| `qwen_probe_then_migration` | 0.675000 | 0.812500 | 0.704167 | 0.812500 |
| `rex_probe` | 0.950000 | 0.000000 | 0.950000 | 0.000000 |
| `rex_probe_then_migration` | 0.803125 | 0.500000 | 0.825000 | 0.500000 |

## Step Results

| Category step | sft_v3 score | sft_v3 action | route_grpo_v4 score | route_grpo_v4 action |
|---|---:|---:|---:|---:|
| `adela_eval#step1` | 0.916667 | 1.000000 | 0.916667 | 1.000000 |
| `clarify_incomplete#step1` | 0.200000 | 0.000000 | 0.200000 | 0.000000 |
| `flux_generation#step1` | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| `general_answer#step1` | 0.075000 | 0.000000 | 0.181250 | 0.125000 |
| `memory_end#step1` | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| `migration_advisor#step1` | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| `pipeline_eval#step1` | 0.462500 | 0.625000 | 0.450000 | 0.625000 |
| `private_lookup#step1` | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| `qwen_probe#step1` | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| `qwen_probe_only_contrast#step1` | 0.968750 | 1.000000 | 0.968750 | 1.000000 |
| `qwen_probe_then_migration#step1` | 0.606250 | 0.625000 | 0.643750 | 0.625000 |
| `qwen_probe_then_migration#step2` | 0.743750 | 1.000000 | 0.764583 | 1.000000 |
| `rex_probe#step1` | 0.950000 | 0.000000 | 0.950000 | 0.000000 |
| `rex_probe_then_migration#step1` | 0.831250 | 0.000000 | 0.843750 | 0.000000 |
| `rex_probe_then_migration#step2` | 0.775000 | 1.000000 | 0.806250 | 1.000000 |

## Paired Comparisons

| Comparison | Step delta [95% CI] | Case-macro delta [95% CI] | Action delta [95% CI] | +/=/- steps | Conclusion |
|---|---:|---:|---:|---:|---|
| `sft_v3_to_route_grpo_v4` | +0.013056 [+0.002778, +0.028472] | +0.011138 [+0.000160, +0.028686] | +0.008333 [+0.000000, +0.025000] | 9/110/1 | supported |

Intervals use a paired bootstrap clustered by `entity_id`. Claim scope follows the registered study split.
