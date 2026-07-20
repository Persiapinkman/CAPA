# N65: selected GRPO multi-repeat development validation

The n64 learning-rate-`2e-8` adapter was evaluated three times on each of the
opened V13 and V14 24-case development cohorts. Run 1 for each cohort was the
complete n64 selection run; n65 added runs 2 and 3 without changing inference
settings.

| Cohort | Run 1 (%) | Run 2 (%) | Run 3 (%) | Mean (%) | Range (pp) |
|---|---:|---:|---:|---:|---:|
| V13 query-style-2 | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 0.0000 |
| V14 explicit structured-state wording | 100.0000 | 92.6000 | 92.6000 | 95.0667 | 7.4000 |

All six runs passed current-success 12/12. V13 also passed metric-veto 12/12
in every run. V14 runs 2 and 3 each failed only
`PRLV14-SC-001-QWEN-PV3`, reproducing checkpoint 6's single fixed residual.
There were no runtime errors.

The hypothesis of six exact, zero-range runs is not supported. Nevertheless,
the small-LR GRPO candidate is materially more robust than n60 on V14
(95.0667% versus 85.2000%, range 7.4 versus 14.8 points) and is perfectly
stable on the actual V13 query-style-2 structure where the target ladder was
observed. A fresh confirmation may therefore reuse only that structural
template, with new entity names, error aliases, fixture lexicon, and case IDs.
It must not mix in V14's easier wording, which made the original SFT score
100% and destroyed the desired comparator ordering.

Artifacts: `/raid/zkq/artifacts/CAPA/arbor/ladder_n65/lr2e-8_multirepeat_dev_20260720T0820Z`.
