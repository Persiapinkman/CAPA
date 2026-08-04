#!/bin/bash
# v8_retry3 SFT, action-balanced (2026-08-04, round 2).
#
# Round 1 (_bg_v8_sft.sh) trained on the raw v8 mix and failed the support gate
# at every checkpoint. The checkpoint sweep on the v8 pool showed why:
#
#   ckpt   gold    var    forbid   note
#     50  0.5117  0.9000  0.3800   retry not learned yet
#    100  0.6183  0.4200  0.3333   best gold, still< 0.80
#    150  0.5767  0.6067  0.6667   detector over-generalised:
#                                  P6 gold 0.694 -> 0.000
#
# Root cause: step-1 supervision was 640 of 1440 rows and every step-1 target
# is a detector call, because every trajectory starts by probing. That is
# uninformative supervision -- there is no decision to make at step 1 -- yet it
# pushed detectors to 55.6% of all supervision. Under-train and retry is never
# learned; over-train and retry is applied everywhere.
#
# This is the n52 conditional collapse, and n58 fixed it by mechanically
# balancing the target action classes rather than by training longer.
#
# The balanced set (CAPA_SFT_BALANCE_ACTIONS=1):
#
#   step1 rows      640 -> 80    (balanced down-sample over category x detector)
#   action mix      detector 55.6% / migrate 33.3% / end 11.1%
#                -> detector 36.8% / migrate 31.6% / end 31.6%
#   rows            1440 -> 1520 (minority classes up-sampled, 640 repeats)
#
# Repeats are tagged with balance_repeat_index and counted separately in the
# audit, so the accidental-duplicate gate still protects against real
# duplication (organic duplicates: 0).
set -euo pipefail

cd /apdcephfs_hzlf/share_1227201/zkq/projects/CAPA

export FORCE=1
export SFT_DATA_DIR=/apdcephfs_hzlf/share_1227201/zkq/projects/CAPA/training/planner_grpo_seed_v1/sft_data_planner_retry_migrate_v8_retry3_qwen35_nothinking_nohint_balanced
export SFT_EXPECTED_TRAIN_ROWS=1520
export SFT_EXPECTED_DEV_ROWS=440
export SFT_EXPECTED_DATASET_ID=planner_retry_migrate_v8_retry3
export CAPA_SKIP_TOKEN_COUNT_DRIFT=1
export CAPA_ALLOW_MAX_LENGTH=1
export MAX_LENGTH=12288
# Dense grid. On the unbalanced set the useful range was 50-150; keep 50-step
# granularity so the earliest healthy checkpoint can be located.
export SFT_SAVE_STEPS=50
export SFT_SAVE_TOTAL_LIMIT=8

bash scripts/reproduce/run_h20_repro.sh sft
