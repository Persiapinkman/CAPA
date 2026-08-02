#!/usr/bin/env bash
# Qwen3.5-4B SFT + GRPO smoke on 4x H20.
#
# The intent is to prove that the existing trainer code executes end-to-end on
# the 4-H20 topology with bf16 + SDPA. The wrapped scripts already implement
# ``dry-run`` (no optimizer step) and ``canary``/``g4`` modes for a single
# optimizer step. We use them, so we never fork the trainer logic itself.
#
# Phases:
#   sft-dry   run SFT trainer in dry-run mode  (model + data audit only)
#   sft-smoke run SFT trainer with 3 optimizer steps and 1 gradient-accumulation
#   grpo-dry  run GRPO trainer in dry-run mode
#   grpo-smoke run GRPO with MAX_STEPS=1 (matches the ``g4`` mode)
#   full-smoke sft-dry -> sft-smoke -> grpo-dry -> grpo-smoke
#
# Env overrides:
#   MODEL_PATH             default /raid/zkq/models/Qwen3.5-4B
#   CUDA_VISIBLE_DEVICES   default 0,1,2,3
#   NUM_PROCESSES          default 4  (must match GPU count)
#   ATTN_IMPL              default sdpa (H20 also supports flash_attention_2)
#   PRECISION              informational; the underlying script sets bf16 by
#                          picking bfloat16 based on device capability

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PHASE="${1:?usage: train_qwen35_4b_h20_smoke.sh <sft-dry|sft-smoke|grpo-dry|grpo-smoke|full-smoke>}"

ENV_DIR="${ROOT_DIR}/.venv-qwen35-grpo"
PY="${ENV_DIR}/bin/python"
TORCHRUN="${ENV_DIR}/bin/torchrun"
[[ -x "${PY}" && -x "${TORCHRUN}" ]] || {
  echo "train venv missing at ${ENV_DIR}; run setup_h20_env.sh TARGET=train"; exit 2; }

MODEL_PATH="${MODEL_PATH:-/raid/zkq/models/Qwen3.5-4B}"
[[ -d "${MODEL_PATH}" ]] || { echo "model missing at ${MODEL_PATH}"; exit 2; }

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a _gpus <<< "${CUDA_VISIBLE_DEVICES}"
NUM_PROCESSES="${NUM_PROCESSES:-${#_gpus[@]}}"

if [[ "${NUM_PROCESSES}" -ne 4 && "${NUM_PROCESSES}" -ne 6 && "${NUM_PROCESSES}" -ne 8 ]]; then
  # The audited SFT script accepts 4 or 6; GRPO accepts 4 or 8. On 4x H20 we use 4.
  echo "NUM_PROCESSES=${NUM_PROCESSES} not accepted by upstream launchers"; exit 2
fi

ATTN_IMPL="${ATTN_IMPL:-sdpa}"
ART_ROOT="${ART_ROOT:-/raid/zkq/artifacts/CAPA}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

STAMP="$(date +%Y%m%dT%H%M%SZ)"

# ---- SFT smoke ---------------------------------------------------------------
SFT_DATA_DIR="training/planner_grpo_seed_v1/sft_data_planner_retry_migrate_v6_qwen35_nothinking"
[[ -f "${SFT_DATA_DIR}/metadata.json" ]] || { echo "SFT data missing"; exit 2; }

phase_sft_dry() {
  echo "[smoke] SFT dry-run"
  RUN_MODE=dry-run \
    ENV_DIR="${ENV_DIR}" MODEL_PATH="${MODEL_PATH}" \
    DATA_DIR="${SFT_DATA_DIR}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    NUM_PROCESSES="${NUM_PROCESSES}" \
    REPORT_TO="none" \
    OUTPUT_DIR="${ART_ROOT}/smoke/qwen35_4b_sft_dryrun_${STAMP}" \
    bash scripts/run_qwen35_4b_planner_v6_sft.sh
}

phase_sft_smoke() {
  echo "[smoke] SFT canary: MAX_STEPS=3, no wandb"
  CONFIRM_TRAIN=YES \
    RUN_MODE=train \
    ENV_DIR="${ENV_DIR}" MODEL_PATH="${MODEL_PATH}" \
    DATA_DIR="${SFT_DATA_DIR}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    NUM_PROCESSES="${NUM_PROCESSES}" \
    REPORT_TO="none" \
    MAX_STEPS=3 \
    GRADIENT_ACCUMULATION_STEPS=1 \
    EVAL_STEPS=100 SAVE_STEPS=100 SAVE_TOTAL_LIMIT=1 \
    OUTPUT_DIR="${ART_ROOT}/smoke/qwen35_4b_sft_smoke_${STAMP}" \
    bash scripts/run_qwen35_4b_planner_v6_sft.sh
}

# ---- GRPO smoke --------------------------------------------------------------
GRPO_STEP_DATA="training/planner_grpo_seed_v1/step_data/planner_multistep_grpo_value_v5_train_v1_qwen35_4b_nothinking_step2.jsonl"
[[ -f "${GRPO_STEP_DATA}" ]] || { echo "GRPO step data missing"; exit 2; }
[[ -f "${GRPO_STEP_DATA%.jsonl}.manifest.json" ]] || { echo "GRPO manifest missing"; exit 2; }

phase_grpo_dry() {
  echo "[smoke] GRPO dry-run"
  RUN_MODE=dry-run \
    ENV_DIR="${ENV_DIR}" MODEL_PATH="${MODEL_PATH}" \
    STEP_DATA="${GRPO_STEP_DATA}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    NUM_PROCESSES="${NUM_PROCESSES}" \
    REPORT_TO="none" \
    OUTPUT_DIR="${ART_ROOT}/smoke/qwen35_4b_grpo_dryrun_${STAMP}" \
    bash scripts/run_qwen35_4b_grpo_v5_train_v1.sh
}

phase_grpo_smoke() {
  echo "[smoke] GRPO g4: single optimizer step for infra validation"
  # upstream ``g4`` mode forces MAX_STEPS=1 and does not require CONFIRM_TRAIN
  RUN_MODE=g4 \
    ENV_DIR="${ENV_DIR}" MODEL_PATH="${MODEL_PATH}" \
    STEP_DATA="${GRPO_STEP_DATA}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    NUM_PROCESSES="${NUM_PROCESSES}" \
    REPORT_TO="none" \
    OUTPUT_DIR="${ART_ROOT}/smoke/qwen35_4b_grpo_g4_${STAMP}" \
    bash scripts/run_qwen35_4b_grpo_v5_train_v1.sh
}

case "${PHASE}" in
  sft-dry)      phase_sft_dry ;;
  sft-smoke)    phase_sft_smoke ;;
  grpo-dry)     phase_grpo_dry ;;
  grpo-smoke)   phase_grpo_smoke ;;
  full-smoke)   phase_sft_dry; phase_sft_smoke; phase_grpo_dry; phase_grpo_smoke ;;
  *) echo "unknown phase '${PHASE}'"; exit 2 ;;
esac
echo "[smoke] done: ${PHASE}"
