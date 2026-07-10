#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-trl-grpo-cu124/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${ROOT_DIR}/.venv-trl-grpo-cu124/bin/accelerate}"
MODEL_PATH="${MODEL_PATH:-/raid/zkq/models/Qwen2.5-7B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
TRAIN_FILE="${TRAIN_FILE:-training/planner_grpo_seed_v1/sft_data/train.jsonl}"
EVAL_FILE="${EVAL_FILE:-training/planner_grpo_seed_v1/sft_data/val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/planner-sft-qwen25-7b-trl-lora-warmup-v1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"

MAX_STEPS="${MAX_STEPS:--1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
SAVE_STEPS="${SAVE_STEPS:-25}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
EVAL_STEPS="${EVAL_STEPS:-25}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
MAX_LENGTH="${MAX_LENGTH:-5120}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

if [[ ! -x "${PYTHON_BIN}" || ! -x "${ACCELERATE_BIN}" ]]; then
  echo "Missing TRL env. Expected ${PYTHON_BIN} and ${ACCELERATE_BIN}" >&2
  exit 1
fi

if [[ ! -f "${TRAIN_FILE}" || ! -f "${EVAL_FILE}" ]]; then
  "${PYTHON_BIN}" training/planner_grpo_seed_v1/scripts/build_planner_sft_data.py
fi

export CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/demo${PYTHONPATH:+:${PYTHONPATH}}"

adapter_args=()
if [[ -n "${ADAPTER_PATH}" ]]; then
  adapter_args=(--adapter-path "${ADAPTER_PATH}")
fi

"${ACCELERATE_BIN}" launch \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision fp16 \
  training/planner_grpo_seed_v1/scripts/train_planner_sft_trl.py \
  --model-name-or-path "${MODEL_PATH}" \
  "${adapter_args[@]}" \
  --train-file "${TRAIN_FILE}" \
  --eval-file "${EVAL_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-steps "${MAX_STEPS}" \
  --num-train-epochs "${NUM_TRAIN_EPOCHS}" \
  --save-steps "${SAVE_STEPS}" \
  --save-total-limit "${SAVE_TOTAL_LIMIT}" \
  --eval-steps "${EVAL_STEPS}" \
  --gradient-accumulation-steps "${GRAD_ACCUM_STEPS}" \
  --max-length "${MAX_LENGTH}" \
  --learning-rate "${LEARNING_RATE}" \
  --lora-r "${LORA_R}" \
  --lora-alpha "${LORA_ALPHA}" \
  --lora-dropout "${LORA_DROPOUT}" \
  --attn-implementation sdpa \
  --logging-steps 1 \
  --report-to none
