#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ENV_DIR="${ENV_DIR:-${ROOT_DIR}/.venv-qwen35-grpo}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_DIR}/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${ENV_DIR}/bin/torchrun}"
MODEL_PATH="${MODEL_PATH:-/raid/zkq/models/Qwen3.5-4B}"
DATA_DIR="${DATA_DIR:-training/planner_grpo_seed_v1/sft_data_planner_retry_migrate_v6_qwen35_nothinking}"
RUN_MODE="${RUN_MODE:-dry-run}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
NUM_PROCESSES="${NUM_PROCESSES:-6}"
SEED="${SEED:-42}"
REPORT_TO="${REPORT_TO:-auto}"
WANDB_PROJECT="${WANDB_PROJECT:-capa-planner-post-training}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-planner-retry-migrate-v6}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_TAGS="${WANDB_TAGS:-stage1,sft,planner-v6,seed${SEED}}"

case "${RUN_MODE}" in
  dry-run)
    TRAIN_MODE="dry-run"
    OUTPUT_DIR="${OUTPUT_DIR:-experiments/runs/qwen35_4b_planner_v6_sft_dryrun}"
    ;;
  train)
    TRAIN_MODE="train"
    RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_qwen35_4b_planner_v6_sft_seed${SEED}}"
    OUTPUT_DIR="${OUTPUT_DIR:-experiments/runs/${RUN_ID}}"
    ;;
  *)
    echo "Unsupported RUN_MODE=${RUN_MODE}; use dry-run or train" >&2
    exit 2
    ;;
esac

if [[ ! -x "${PYTHON_BIN}" || ! -x "${TORCHRUN_BIN}" ]]; then
  echo "Missing Qwen3.5 training environment under ${ENV_DIR}" >&2
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" || ! -f "${DATA_DIR}/metadata.json" ]]; then
  echo "Missing model or frozen Planner V6 SFT data" >&2
  exit 1
fi

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#GPU_IDS[@]}" -ne "${NUM_PROCESSES}" ]]; then
  echo "CUDA_VISIBLE_DEVICES count must equal NUM_PROCESSES" >&2
  exit 1
fi
if [[ "${NUM_PROCESSES}" -ne 4 && "${NUM_PROCESSES}" -ne 6 ]]; then
  echo "Audited Planner V6 SFT topology requires 4 or 6 ranks" >&2
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
  if [[ "${REPORT_TO}" == "auto" || "${REPORT_TO}" == "all" || ",${REPORT_TO}," == *",wandb,"* ]]; then
    if ! "${PYTHON_BIN}" -c 'import wandb' >/dev/null 2>&1; then
      echo "W&B reporting requested but wandb is missing in ${ENV_DIR}" >&2
      echo "Install it with: uv pip install --python ${PYTHON_BIN} wandb==0.28.0" >&2
      exit 1
    fi
  fi
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
  --max-length "${MAX_LENGTH:-4800}"
  --max-steps "${MAX_STEPS:-100}"
  --learning-rate "${LEARNING_RATE:-2e-5}"
  --warmup-steps "${WARMUP_STEPS:-5}"
  --per-device-train-batch-size 1
  --per-device-eval-batch-size 1
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-2}"
  --eval-steps "${EVAL_STEPS:-10}"
  --save-steps "${SAVE_STEPS:-25}"
  --save-total-limit "${SAVE_TOTAL_LIMIT:-4}"
  --logging-steps 1
  --seed "${SEED}"
  --attn-implementation sdpa
  --report-to "${REPORT_TO}"
  --run-name "${WANDB_RUN_NAME:-$(basename "${OUTPUT_DIR}")}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-entity "${WANDB_ENTITY}"
  --wandb-group "${WANDB_RUN_GROUP}"
  --wandb-tags "${WANDB_TAGS}"
  --wandb-mode "${WANDB_MODE}"
)

if [[ "${RUN_MODE}" == "dry-run" ]]; then
  "${PYTHON_BIN}" training/planner_grpo_seed_v1/scripts/train_qwen35_4b_planner_v6_sft.py "${common_args[@]}"
else
  "${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc-per-node="${NUM_PROCESSES}" \
    training/planner_grpo_seed_v1/scripts/train_qwen35_4b_planner_v6_sft.py "${common_args[@]}"
fi
