#!/usr/bin/env bash
# Launch SFT merge + SFT eval in background.
set -euo pipefail
cd /apdcephfs_hzlf/share_1227201/zkq/projects/CAPA

LOG_DIR=/apdcephfs_hzlf/share_1227201/zkq/capa_h20/artifacts/CAPA/logs/train
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/sft_eval_v7_${STAMP}.log"

exec > "${LOG_FILE}" 2>&1

# Pin SFT to checkpoint-100: eval_loss<0.005 by then, later steps overfit.
SFT_CHECKPOINT_STEP=100 \
bash scripts/reproduce/run_h20_repro.sh sft-merge sft-eval
