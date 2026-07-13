# Planner Coreference-Contrast GRPO Seed-42 Coreference Step-1 Endpoint

## Run Summary

| Run | Step mean | Case macro | Category macro | Action match | Exact pass | JSON valid | Extra text |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sft_v3` | 0.327446 | 0.327446 | 0.327446 | 0.000000 | 0.000000 | 1.000000 | 0.000000 |
| `coref42` | 0.360054 | 0.360054 | 0.360054 | 0.000000 | 0.000000 | 1.000000 | 0.000000 |

## Paired Comparisons

| Comparison | Step delta [95% CI] | Case-macro delta [95% CI] | +/=/- steps | Conclusion |
|---|---:|---:|---:|---|
| `sft_v3_to_coref42` | +0.032609 [+0.000000, +0.097826] | +0.032609 [+0.000000, +0.097826] | 1/7/0 | inconclusive |

Intervals use a paired bootstrap clustered by `entity_id`. Claim scope follows the registered study split.
