#!/usr/bin/env bash
# GRPO x3 seeds + eval + compare + gate — attempt 2, high-temp sampling.
set -euo pipefail
cd /apdcephfs_hzlf/share_1227201/zkq/projects/CAPA

LOG_DIR=/apdcephfs_hzlf/share_1227201/zkq/capa_h20/artifacts/CAPA/logs/train
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/grpo_all_v7_${STAMP}.log"

exec > "${LOG_FILE}" 2>&1

# High-temp sampling to overcome SFT ckpt-100 saturation. argument_exact
# was 0.75-0.87 at SFT ckpt-100 with 4 gens giving identical outputs at
# temp=0.7; bumping to 1.1 with top_p=0.95 to induce reward variance.
# num_generations must stay 4 (trainer G3-gate).
SFT_CHECKPOINT_STEP=100 \
GENERATION_TEMPERATURE=1.1 \
GENERATION_TOP_P=0.95 \
bash scripts/reproduce/run_h20_repro.sh grpo grpo-eval compare gate
