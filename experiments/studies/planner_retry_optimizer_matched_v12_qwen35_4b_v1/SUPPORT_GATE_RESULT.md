# V12 optimizer-matched support gate result

Status: **PASS — optimizer authorized**

The V10 checkpoint-10 continuation initializer was sampled on all 576 entity-disjoint V12 support prompts with four completions per prompt at the frozen `temperature=0.9`, `top_p=0.9`, and `max_new_tokens=320`. The V12 support composition exactly matches the optimizer's 288 non-migration / 288 migration target-row balance.

| Gate | Observed | Threshold | Result |
| --- | ---: | ---: | :---: |
| Complete prompt groups | 576/576 | 576 | Pass |
| Complete samples | 2,304/2,304 | 2,304 | Pass |
| JSON valid | 100% | >=99% | Pass |
| Clipped | 0% | <=1% | Pass |
| Primary gold-action support | 74.65% | >=70% | Pass |
| Primary task-reward variance | 25.69% (74/288) | >=20% | Pass |
| Control gold-action support | 96.53% | >=50% | Pass |
| Control task-reward variance | 24.65% (71/288) | >=20% | Pass |
| Support block A gold / variance | 85.07% / 26.39% | >=50% / >=15% | Pass |
| Support block B gold / variance | 86.11% / 23.96% | >=50% / >=15% | Pass |
| Forbidden-action sample rate | 18.27% | 2%-20% | Pass |
| Safety variance, current success | 21 groups | >=6 | Pass |
| Safety variance, fresh retry | 28 groups | >=6 | Pass |
| Safety variance, post-retry success | 32 groups | >=6 | Pass |
| Safety variance, overall | 84 groups | >=43 | Pass |
| Primary safety-variance rate | 28.13% (81/288) | >=15% | Pass |

All 59 preregistered hard checks passed, including every scenario x detector task-support stratum. The V11 overall safety count gate was not relaxed: optimizer-matched support increased the observed count from 32 to 84 while retaining the original threshold of 43.

The all-or-none optimizer artifact contains exactly 576 rows over 24 train entities. It has 192 `end`, 96 `retry`, and 288 `migrate` targets, so non-migration and migration rows are balanced 1:1. Its SHA-256 is `c144197d932ae0b76dd9596fd5b6f8f2c3bd9594ec97f3cddf2876ba64b37053`.

Evidence:

- Support decision: `support_decision.json` (`5298a6dde1891dd48dbfb6c8655e02f9e4a217dd9b5e132701e06d851d7b47cb`)
- Combined samples: `experiments/runs/20260717_qwen35_4b_v12_support6x_v10ckpt10/samples.jsonl` (`c769914a2f24e4f5dbca4346f8a946ace75fe9be545040aa6e39d0121b891afc`)
- Combined summary: `experiments/runs/20260717_qwen35_4b_v12_support6x_v10ckpt10/summary.json` (`d093e419710176c5e679bb4c57b1ca9bf1b7df5be8c1b52cc408f3599d42bfbe`)
- Optimizer manifest: `training/planner_grpo_seed_v1/step_data/planner_retry_optimizer_matched_v12_optimizer_qwen35_4b_nothinking_mixed_steps.manifest.json` (`d510591a30d99db01311837a3e5f5862d8b0a749d884d7537415e8fedeb4e783`)
- Sealed-test commitment remains unopened: `6e9413858d6e9cc09672d00d80dbda593e13e190457915498f5240c97b16c149`

The next allowed action is the preregistered two-step GRPO canary. The eight-step screen remains blocked until the V12-specific runtime and safety-telemetry audit passes.
