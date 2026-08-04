#!/bin/bash
# GRPO on the first support-authorised configuration (2026-08-04).
#
# Lineage (all four cells of the 2x2 support matrix are recorded in
# reports/V7_GRPO_SUPPORT_AUDIT_20260803.md):
#
#   initializer            pool     gold_support  nonzero_variance  gate
#   hint-SFT   ckpt-100    hint        1.0000         0.0000        FAIL
#   hint-SFT   ckpt-100    nohint      0.7479         0.2333        FAIL
#   noHint-SFT ckpt-50     nohint      0.6208         0.8667        FAIL
#   noHint-SFT ckpt-50     hint        0.9729         0.6500        PASS  <-- this run
#
# Why this cell: the initializer was trained on hint-stripped prompts, so it
# does not depend on the routing-hint literal and keeps policy entropy
# (eval_entropy 1.05 vs 0.006 for the hint-trained ckpt-100).  The optimizer
# pool keeps the hint because dev/test evaluation also keeps it -- changing
# both initializer and evaluation protocol at once would make the comparison
# uninterpretable.
#
# Guard rails active in this run:
#   * status/support.done is required by phase_grpo (SUPPORT_REQUIRED=1)
#   * VanishingSignalCallback aborts on 10 consecutive degenerate steps
#     (frac_reward_zero_std >= 0.99 AND grad_norm == 0)
set -euo pipefail

cd /apdcephfs_hzlf/share_1227201/zkq/projects/CAPA

export SFT_CHECKPOINT_STEP=50
export SEEDS="42"
export GRPO_MAX_STEPS=100
export GRPO_SAVE_STEPS=25
# A stale grpo-42.done exists from the 2026-08-02 saturated-pool run; FORCE=1
# lets phase_grpo re-run this seed (it clears the per-seed marker itself).
# The support hard gate is still enforced via status/support.done.
export FORCE=1

bash scripts/reproduce/run_h20_repro.sh grpo
