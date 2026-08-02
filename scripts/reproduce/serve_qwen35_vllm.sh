#!/usr/bin/env bash
# Launch a vLLM OpenAI-compatible server for a Qwen3.5 variant on H20.
#
# Alias selects the config profile:
#   4b   -> single H20, tensor-parallel-size=1, port 8001
#   35b  -> 4x H20  , tensor-parallel-size=4, port 8002 (MoE)
#
# Server picks up chat_template.jinja and tokenizer_config.json shipped with
# the model, matching the frozen contract in
# training/planner_grpo_seed_v1/sft_data_planner_retry_migrate_v6_qwen35_nothinking/metadata.json.
#
# Env overrides:
#   MODELS_ROOT              default /raid/zkq/models
#   HOST                     default 127.0.0.1
#   PORT                     default 8001 / 8002 per alias
#   DTYPE                    default bfloat16
#   MAX_MODEL_LEN            default 8192
#   GPU_MEMORY_UTILIZATION   default 0.90
#   TENSOR_PARALLEL_SIZE     default 1 / 4 per alias
#   CUDA_VISIBLE_DEVICES     required for multi-GPU runs
#   LOG_DIR                  default /raid/zkq/artifacts/CAPA/logs/vllm

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

ALIAS="${1:?usage: serve_qwen35_vllm.sh <4b|35b>}"
MODELS_ROOT="${MODELS_ROOT:-/raid/zkq/models}"
HOST="${HOST:-127.0.0.1}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

case "${ALIAS}" in
  4b)
    SERVED_NAME="${SERVED_NAME:-Qwen3.5-4B}"
    MODEL_DIR="${MODELS_ROOT}/Qwen3.5-4B"
    PORT="${PORT:-8001}"
    TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
    ;;
  35b)
    SERVED_NAME="${SERVED_NAME:-Qwen3.5-35B-A3B}"
    MODEL_DIR="${MODELS_ROOT}/Qwen3.5-35B-A3B"
    PORT="${PORT:-8002}"
    TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
    ;;
  *) echo "unknown alias '${ALIAS}' (use 4b or 35b)"; exit 2 ;;
esac

[[ -f "${MODEL_DIR}/config.json" ]] || { echo "model missing at ${MODEL_DIR}"; exit 2; }

VENV="${ROOT_DIR}/.venv-h20-infer"
PY="${VENV}/bin/python"
[[ -x "${PY}" ]] || { echo "infer venv missing; run setup_h20_env.sh"; exit 2; }

LOG_DIR="${LOG_DIR:-/raid/zkq/artifacts/CAPA/logs/vllm}"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/vllm_${ALIAS}_${STAMP}.log"
PID_FILE="${LOG_DIR}/vllm_${ALIAS}.pid"

echo "[serve] ${ALIAS}: dtype=${DTYPE} tp=${TENSOR_PARALLEL_SIZE} port=${PORT}"
echo "[serve] model_dir=${MODEL_DIR}"
echo "[serve] log=${LOG_FILE}"

# Show resolved argv for the record (executes in the actual venv).
"${PY}" -m capa.inference.h20_backend "${SERVED_NAME}" \
  --models-root "${MODELS_ROOT}" --host "${HOST}" --port "${PORT}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" --dtype "${DTYPE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" 2>/dev/null \
  || true  # informational only

exec > >(tee -a "${LOG_FILE}") 2>&1

# CUDA_VISIBLE_DEVICES governs which H20s the process sees.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[serve] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

"${PY}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_DIR}" \
  --served-model-name "${SERVED_NAME}" \
  --host "${HOST}" --port "${PORT}" \
  --dtype "${DTYPE}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --trust-remote-code &

VLLM_PID=$!
echo "${VLLM_PID}" > "${PID_FILE}"
echo "[serve] vllm pid=${VLLM_PID}"

trap 'echo "[serve] shutting down ${VLLM_PID}"; kill "${VLLM_PID}" 2>/dev/null || true; rm -f "${PID_FILE}"' INT TERM EXIT
wait "${VLLM_PID}"
