#!/usr/bin/env bash
# GRPO x3 - attempt 5: extreme high-temp sampling to force reward variance.
set -euo pipefail
cd /apdcephfs_hzlf/share_1227201/zkq/projects/CAPA

LOG_DIR=/apdcephfs_hzlf/share_1227201/zkq/capa_h20/artifacts/CAPA/logs/train
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/grpo_all_v7_${STAMP}.log"
exec > "${LOG_FILE}" 2>&1

# SFT ckpt-100 is very confident on GRPO stand-alone step2 prompts
# (frac_reward_zero_std ≈ 1.0 at temp=0.9). Push to temp=1.5 top_p=1.0
# to force the 4 gens to diverge in argument_exact / finish_after_tool.
# We add num_generations override attempt in case it becomes tunable.
SFT_CHECKPOINT_STEP=100 \
GENERATION_TEMPERATURE=1.5 \
GENERATION_TOP_P=1.0 \
bash scripts/reproduce/run_h20_repro.sh grpo grpo-eval compare gate
