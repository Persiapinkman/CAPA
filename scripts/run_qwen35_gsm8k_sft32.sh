#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ENV_DIR="${ENV_DIR:-${ROOT_DIR}/.venv-qwen35-grpo}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_DIR}/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${ENV_DIR}/bin/torchrun}"
MODEL_PATH="${MODEL_PATH:-/raid/zkq/models/Qwen3.5-4B}"
DATA_DIR="${DATA_DIR:-training/public_sft_grpo_v1/data/gsm8k_sft32_v1}"
RUN_MODE="${RUN_MODE:-dry-run}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,3,5}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
SEED="${SEED:-42}"

case "${RUN_MODE}" in
  dry-run)
    TRAIN_MODE="dry-run"
    OUTPUT_DIR="${OUTPUT_DIR:-experiments/runs/20260715_qwen35_4b_gsm8k_sft32_dryrun_v1}"
    ;;
  train)
    TRAIN_MODE="train"
    OUTPUT_DIR="${OUTPUT_DIR:-experiments/runs/20260715_qwen35_4b_gsm8k_sft32_overfit_v1}"
    ;;
  *)
    echo "Unsupported RUN_MODE=${RUN_MODE}; use dry-run or train" >&2
    exit 2
    ;;
esac

if [[ ! -x "${PYTHON_BIN}" || ! -x "${TORCHRUN_BIN}" ]]; then
  echo "Missing frozen Qwen3.5 environment under ${ENV_DIR}" >&2
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" || ! -f "${DATA_DIR}/manifest.json" ]]; then
  echo "Missing model or prepared GSM8K data" >&2
  exit 1
fi

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#GPU_IDS[@]}" -ne "${NUM_PROCESSES}" ]]; then
  echo "CUDA_VISIBLE_DEVICES count must equal NUM_PROCESSES" >&2
  exit 1
fi
if [[ "${NUM_PROCESSES}" -ne 4 && "${NUM_PROCESSES}" -ne 6 ]]; then
  echo "Audited SFT32 topology requires 4 or 6 ranks" >&2
  exit 1
fi
if [[ "${TRAIN_MODE}" == "train" && -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to reuse training output directory ${OUTPUT_DIR}" >&2
  exit 1
fi

confirm_args=()
if [[ "${TRAIN_MODE}" == "train" ]]; then
  if [[ "${CONFIRM_TRAIN:-}" != "YES" ]]; then
    echo "Optimizer steps require CONFIRM_TRAIN=YES" >&2
    exit 1
  fi
  for gpu_id in "${GPU_IDS[@]}"; do
    busy="$(nvidia-smi -i "${gpu_id}" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | wc -l)"
    if [[ "${busy}" -ne 0 ]]; then
      echo "Refusing selected busy GPU ${gpu_id}" >&2
      exit 1
    fi
  done
  confirm_args=(--confirm-train)
fi

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

common_args=(
  --mode "${TRAIN_MODE}"
  "${confirm_args[@]}"
  --model-name-or-path "${MODEL_PATH}"
  --data-dir "${DATA_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --expected-world-size "${NUM_PROCESSES}"
  --max-length 1024
  --max-steps 100
  --learning-rate 2e-5
  --warmup-steps 5
  --per-device-train-batch-size 1
  --per-device-eval-batch-size 1
  --gradient-accumulation-steps 2
  --eval-steps 10
  --save-steps 25
  --logging-steps 1
  --seed "${SEED}"
  --attn-implementation sdpa
)

if [[ "${RUN_MODE}" == "dry-run" ]]; then
  "${PYTHON_BIN}" training/public_sft_grpo_v1/scripts/train_qwen35_gsm8k_sft.py "${common_args[@]}"
else
  "${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc-per-node="${NUM_PROCESSES}" \
    training/public_sft_grpo_v1/scripts/train_qwen35_gsm8k_sft.py "${common_args[@]}"
fi
