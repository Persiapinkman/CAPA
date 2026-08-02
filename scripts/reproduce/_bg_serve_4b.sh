#!/usr/bin/env bash
# Wrap the vLLM 4B launch so it survives the caller's shell timeout.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

MODELS_ROOT="/apdcephfs_hzlf/share_1227201/zkq/capa_h20/models"
LOG_DIR="/apdcephfs_hzlf/share_1227201/zkq/capa_h20/artifacts/CAPA/logs/vllm"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/vllm_4b_${STAMP}.log"

exec > "${LOG_FILE}" 2>&1

export CUDA_VISIBLE_DEVICES=0
.venv-h20-infer/bin/python -m vllm.entrypoints.openai.api_server \
  --model "${MODELS_ROOT}/Qwen3.5-4B" \
  --served-model-name Qwen3.5-4B \
  --host 127.0.0.1 --port 8001 \
  --dtype bfloat16 --tensor-parallel-size 1 \
  --max-model-len 32768 --gpu-memory-utilization 0.90 \
  --trust-remote-code &
echo $! > "${LOG_DIR}/vllm_4b.pid"
wait
