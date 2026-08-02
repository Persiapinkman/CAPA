#!/usr/bin/env bash
# Launch the CAPA SFT phase in background (bypass tool 300s timeout).
set -euo pipefail
cd /apdcephfs_hzlf/share_1227201/zkq/projects/CAPA

LOG_DIR=/apdcephfs_hzlf/share_1227201/zkq/capa_h20/artifacts/CAPA/logs/train
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/sft_v7_${STAMP}.log"

exec > "${LOG_FILE}" 2>&1

bash scripts/reproduce/run_h20_repro.sh sft
