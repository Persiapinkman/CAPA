#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ENV_DIR="${ENV_DIR:-${ROOT_DIR}/.venv-qwen35-grpo}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_DIR}/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${ENV_DIR}/bin/torchrun}"
MODEL_PATH="${MODEL_PATH:-/raid/zkq/models/Qwen3.5-4B}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
STEP_DATA="${STEP_DATA:-training/planner_grpo_seed_v1/step_data/planner_multistep_grpo_value_v5_train_v1_qwen35_4b_nothinking_step2.jsonl}"
EXPECTED_DATASET_ID="${EXPECTED_DATASET_ID:-planner_multistep_grpo_value_v5_train_v1}"
EXPECTED_ROWS="${EXPECTED_ROWS:-480}"
ALLOWED_STEP_INDICES="${ALLOWED_STEP_INDICES:-2}"
RUN_MODE="${RUN_MODE:-dry-run}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
SEED="${SEED:-42}"
REPORT_TO="${REPORT_TO:-auto}"
WANDB_PROJECT="${WANDB_PROJECT:-capa-planner-post-training}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-planner-retry-migrate-v6}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_TAGS="${WANDB_TAGS:-stage2,grpo,planner-v6,seed${SEED}}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
SAVE_STEPS="${SAVE_STEPS:-25}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-4}"
GENERATION_TEMPERATURE="${GENERATION_TEMPERATURE:-0.7}"
GENERATION_TOP_P="${GENERATION_TOP_P:-0.9}"
TASK_REWARD_WEIGHT="${TASK_REWARD_WEIGHT:-0.95}"
FORMAT_REWARD_WEIGHT="${FORMAT_REWARD_WEIGHT:-0.05}"
NO_FORBIDDEN_ACTION_REWARD_WEIGHT="${NO_FORBIDDEN_ACTION_REWARD_WEIGHT:-0.0}"

case "${RUN_MODE}" in
  dry-run)
    TRAIN_MODE="dry-run"
    MAX_STEPS="${MAX_STEPS:-100}"
    OUTPUT_DIR="${OUTPUT_DIR:-experiments/runs/20260715_qwen35_4b_v5_train_v1_dryrun}"
    ;;
  g4)
    TRAIN_MODE="g4"
    MAX_STEPS="1"
    OUTPUT_DIR="${OUTPUT_DIR:-experiments/runs/${RUN_TIMESTAMP}_qwen35_4b_v5_train_v1_g4}"
    ;;
  canary)
    TRAIN_MODE="train"
    MAX_STEPS="${MAX_STEPS:-5}"
    OUTPUT_DIR="${OUTPUT_DIR:-experiments/runs/${RUN_TIMESTAMP}_qwen35_4b_v5_train_v1_seed${SEED}_canary5}"
    ;;
  screen)
    TRAIN_MODE="train"
    MAX_STEPS="${MAX_STEPS:-100}"
    OUTPUT_DIR="${OUTPUT_DIR:-experiments/runs/${RUN_TIMESTAMP}_qwen35_4b_v5_train_v1_seed${SEED}_screen100}"
    ;;
  *)
    echo "Unsupported RUN_MODE=${RUN_MODE}; use dry-run, g4, canary, or screen" >&2
    exit 2
    ;;
esac

if [[ ! -x "${PYTHON_BIN}" || ! -x "${TORCHRUN_BIN}" ]]; then
  echo "Missing frozen Qwen3.5 GRPO environment under ${ENV_DIR}" >&2
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" || ! -f "${STEP_DATA}" || ! -f "${STEP_DATA%.jsonl}.manifest.json" ]]; then
  echo "Missing model or frozen step-data artifact" >&2
  exit 1
fi
if [[ -n "${ADAPTER_PATH}" && ! -f "${ADAPTER_PATH}/adapter_model.safetensors" ]]; then
  echo "Missing SFT adapter weights under ${ADAPTER_PATH}" >&2
  exit 1
fi
if [[ "${RUN_MODE}" != "dry-run" && -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to reuse training output directory ${OUTPUT_DIR}" >&2
  exit 1
fi

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_PROCESSES="${NUM_PROCESSES:-${#GPU_IDS[@]}}"
if [[ "${NUM_PROCESSES}" -ne 4 && "${NUM_PROCESSES}" -ne 8 ]]; then
  echo "The audited run requires 4 or 8 ranks; got ${NUM_PROCESSES}" >&2
  exit 1
fi
if [[ "${#GPU_IDS[@]}" -ne "${NUM_PROCESSES}" ]]; then
  echo "CUDA_VISIBLE_DEVICES count must equal NUM_PROCESSES" >&2
  exit 1
fi

if [[ "${RUN_MODE}" != "dry-run" && "${ALLOW_BUSY_GPUS:-0}" != "1" ]]; then
  BUSY_COUNT=0
  for gpu_id in "${GPU_IDS[@]}"; do
    gpu_busy="$(nvidia-smi -i "${gpu_id}" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | wc -l)"
    BUSY_COUNT="$((BUSY_COUNT + gpu_busy))"
  done
  if [[ "${BUSY_COUNT}" -ne 0 ]]; then
    echo "Refusing to launch on selected busy GPUs: ${BUSY_COUNT} compute process(es) detected" >&2
    exit 1
  fi
fi

GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-${NUM_PROCESSES}}"
if [[ "${NUM_PROCESSES}" -eq 8 ]]; then
  GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
else
  GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
fi

confirm_args=()
adapter_args=()
if [[ -n "${ADAPTER_PATH}" ]]; then
  adapter_args=(--adapter-path "${ADAPTER_PATH}")
fi
if [[ "${TRAIN_MODE}" == "train" ]]; then
  if [[ "${CONFIRM_TRAIN:-}" != "YES" ]]; then
    echo "Optimizer steps require CONFIRM_TRAIN=YES" >&2
    exit 1
  fi
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
  "${adapter_args[@]}"
  --step-data "${STEP_DATA}"
  --expected-dataset-id "${EXPECTED_DATASET_ID}"
  --expected-rows "${EXPECTED_ROWS}"
  --allowed-step-indices "${ALLOWED_STEP_INDICES}"
  --output-dir "${OUTPUT_DIR}"
  --max-steps "${MAX_STEPS}"
  --seed "${SEED}"
  --max-prompt-tokens "${MAX_PROMPT_TOKENS:-4608}"
  --max-completion-length "${MAX_COMPLETION_LENGTH:-320}"
  --num-generations "${NUM_GENERATIONS:-4}"
  --generation-batch-size "${GENERATION_BATCH_SIZE}"
  --expected-world-size "${NUM_PROCESSES}"
  --per-device-train-batch-size 1
  --gradient-accumulation-steps "${GRAD_ACCUM_STEPS}"
  --learning-rate "${LEARNING_RATE}"
  --warmup-steps "${WARMUP_STEPS}"
  --save-steps "${SAVE_STEPS}"
  --save-total-limit "${SAVE_TOTAL_LIMIT}"
  --logging-steps 1
  --temperature "${GENERATION_TEMPERATURE}"
  --top-p "${GENERATION_TOP_P}"
  --task-reward-weight "${TASK_REWARD_WEIGHT}"
  --format-reward-weight "${FORMAT_REWARD_WEIGHT}"
  --no-forbidden-action-reward-weight "${NO_FORBIDDEN_ACTION_REWARD_WEIGHT}"
  --report-to "${REPORT_TO}"
  --run-name "${WANDB_RUN_NAME:-$(basename "${OUTPUT_DIR}")}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-entity "${WANDB_ENTITY}"
  --wandb-group "${WANDB_RUN_GROUP}"
  --wandb-tags "${WANDB_TAGS}"
  --wandb-mode "${WANDB_MODE}"
)

if [[ "${RUN_MODE}" == "dry-run" ]]; then
  "${PYTHON_BIN}" training/planner_grpo_seed_v1/scripts/train_qwen35_4b_grpo.py "${common_args[@]}"
else
  "${TORCHRUN_BIN}" \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="${NUM_PROCESSES}" \
    training/planner_grpo_seed_v1/scripts/train_qwen35_4b_grpo.py \
    "${common_args[@]}"
fi
