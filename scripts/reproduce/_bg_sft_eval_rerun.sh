#!/usr/bin/env bash
# Rerun SFT eval with max_steps fix in rollout.
set -euo pipefail
cd /apdcephfs_hzlf/share_1227201/zkq/projects/CAPA

LOG_DIR=/apdcephfs_hzlf/share_1227201/zkq/capa_h20/artifacts/CAPA/logs/train
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/sft_eval_v7_rerun_${STAMP}.log"
exec > "${LOG_FILE}" 2>&1

# Force re-run of sft-eval by clearing the marker.
rm -f /apdcephfs_hzlf/share_1227201/zkq/capa_h20/artifacts/CAPA/repro_h20/status/eval-sft.done

SFT_CHECKPOINT_STEP=100 \
bash scripts/reproduce/run_h20_repro.sh sft-eval
