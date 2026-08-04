#!/bin/bash
# v8_retry3 SFT: install support for the *retry* action class.
#
# Why a new SFT is required (2026-08-04):
#
# The v8 optimizer pool restores a genuine 3-step retry trajectory, so step 2
# now has a 4-way decision space (detector 20% / migration_advisor 60% /
# end 20%).  Auditing the v7-trained ckpt-50 against that pool gave:
#
#     nonzero_variance = 0.8667   (excellent -- far above the 0.25 gate)
#     gold_support     = 0.5250   (FAIL -- gate needs >= 0.80)
#     forbidden_groups = 0.3267   (premature migrate / third probe)
#
# The variance is there but the initializer cannot reliably *produce* the gold
# action, because every v7 trajectory was 2 steps and the retry action never
# appeared as a step-2 target.  Per the playbook, GRPO must not be started from
# a policy that cannot sample the correct action: SFT installs support, GRPO
# moves boundaries.  This run installs that support.
#
# The v8 SFT set adds the 160 step-3 rows and rebalances step-2 targets, so the
# action distribution moves toward the n58 recipe that historically fixed rule
# induction (retry/migrate/end balanced rather than migrate-dominated).
#
# Prompts are hint-stripped (the data dir carries the _nohint suffix), so the
# policy keeps entropy instead of memorising the routing literal.
set -euo pipefail

cd /apdcephfs_hzlf/share_1227201/zkq/projects/CAPA

export FORCE=1
export SFT_DATA_DIR=/apdcephfs_hzlf/share_1227201/zkq/projects/CAPA/training/planner_grpo_seed_v1/sft_data_planner_retry_migrate_v8_retry3_qwen35_nothinking_nohint
# v8 rows: 1440 train / 360 dev (v7 was 1280 / 320).
export SFT_EXPECTED_TRAIN_ROWS=1440
export SFT_EXPECTED_DEV_ROWS=360
export SFT_EXPECTED_DATASET_ID=planner_retry_migrate_v8_retry3
export CAPA_SKIP_TOKEN_COUNT_DRIFT=1
# step-3 prompts carry a second long observation: p95 ~9.4k, max ~9.6k.
export CAPA_ALLOW_MAX_LENGTH=1
export MAX_LENGTH=12288
# Dense checkpoint grid: the earliest healthy checkpoint is the one we want.
export SFT_SAVE_STEPS=50
export SFT_SAVE_TOTAL_LIMIT=8

bash scripts/reproduce/run_h20_repro.sh sft
