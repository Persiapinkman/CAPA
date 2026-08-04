#!/bin/bash
# No-hint SFT: rebuild the v7 initializer on routing-hint-stripped prompts.
#
# Why: `CAPA_STRIP_ROUTING_HINT` was documented but never implemented, so the
# shipped v7 SFT/GRPO data leaked an explicit routing sentence into
# `observation.summary`.  The resulting ckpt-100 saturates the GRPO optimizer
# pool (gold_support=1.000, nonzero_variance=0.000) which makes GRPO a
# mathematical no-op.  See reports/V7_GRPO_SUPPORT_AUDIT_20260803.md.
#
# This run keeps every other contract identical to20260802_155804(same base
# model, same 1280/320 rows, same LoRA geometry, 400 steps, lr 2e-5) and only
# changes the prompt variant, so the two initializers are comparable.
set -euo pipefail

cd /apdcephfs_hzlf/share_1227201/zkq/projects/CAPA

export FORCE=1
export SFT_DATA_DIR=/apdcephfs_hzlf/share_1227201/zkq/projects/CAPA/training/planner_grpo_seed_v1/sft_data_planner_retry_migrate_v7_longobs_qwen35_nothinking_nohint
# Denser checkpoint grid than the default 100: the "earliest healthy checkpoint"
# rule needs candidates below step 100 now that the task is harder.
export SFT_SAVE_STEPS=50
export SFT_SAVE_TOTAL_LIMIT=8

bash scripts/reproduce/run_h20_repro.sh sft
