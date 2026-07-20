# N63: n58 targeted-SFT checkpoint stability screen

The already opened V14 cohort was used as development evidence only. Each n58
targeted-SFT checkpoint was evaluated for three complete 24-case runs with the
same temperature-zero, top-p 1, max-4096 protocol. All three checkpoints had
previously passed all 24 V13 style-2 development cases once.

| Checkpoint | V14 run 1 (%) | V14 run 2 (%) | V14 run 3 (%) | Mean (%) | Range (pp) | V13 prior |
|---|---:|---:|---:|---:|---:|---:|
| 6 | 92.6000 | 92.6000 | 92.6000 | 92.6000 | 0.0000 | 24/24 |
| 12 | 92.6000 | 100.0000 | 92.6000 | 95.0667 | 7.4000 | 24/24 |
| 18 | 77.8000 | 70.4000 | 85.2000 | 77.8000 | 14.8000 | 24/24 |

Checkpoint 6 is the robustness selection despite its lower mean: every run
passed all 12 current-success cases and 11/12 metric-veto cases, always failing
only `PRLV14-SC-001-QWEN-PV3`. Checkpoint 12 had a higher mean but crossed the
decision boundary on different cases across runs. Checkpoint 18 was rejected.

The n63 hypothesis that a deeper checkpoint would achieve exact and zero-range
V14 coverage was therefore not supported. The reusable conclusion is that the
n58 curriculum itself can be stable at checkpoint 6, while both extra SFT
steps and n60's learning-rate-`5e-8` GRPO update increase boundary sensitivity.
The next intervention must start from checkpoint 6 and use a smaller genuine
GRPO update, selected on complete V13+V14 development cohorts before any fresh
confirmation is materialized.

Artifacts: `/raid/zkq/artifacts/CAPA/arbor/ladder_n63/v14_open_dev_n58_cp_three_repeat_20260720T0715Z`.
