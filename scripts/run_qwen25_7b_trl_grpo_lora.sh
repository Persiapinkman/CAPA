#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-trl-grpo-cu124/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${ROOT_DIR}/.venv-trl-grpo-cu124/bin/accelerate}"
MODEL_PATH="${MODEL_PATH:-/raid/zkq/models/Qwen2.5-7B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
CASES="${CASES:-training/planner_grpo_seed_v1/cases/planner_grpo_focused_4b_cases.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/planner-grpo-qwen25-7b-trl-lora-v1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"

MAX_STEPS="${MAX_STEPS:-20}"
SAVE_STEPS="${SAVE_STEPS:-10}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-8}"
NUM_GENERATIONS="${NUM_GENERATIONS:-2}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-128}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
TEMPERATURE="${TEMPERATURE:-0.4}"
TOP_P="${TOP_P:-0.9}"
SCORE_FIRST_JSON_ONLY="${SCORE_FIRST_JSON_ONLY:-true}"

if [[ ! -x "${PYTHON_BIN}" || ! -x "${ACCELERATE_BIN}" ]]; then
  echo "Missing TRL GRPO env. Expected ${PYTHON_BIN} and ${ACCELERATE_BIN}" >&2
  echo "Create it with the commands recorded in experiments/QWEN25_7B_TRL_GRPO_V1_RUN_REPORT.md" >&2
  exit 1
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
  training/planner_grpo_seed_v1/scripts/train_planner_grpo_trl.py \
  --model-name-or-path "${MODEL_PATH}" \
  "${adapter_args[@]}" \
  --cases "${CASES}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-steps "${MAX_STEPS}" \
  --save-steps "${SAVE_STEPS}" \
  --save-total-limit "${SAVE_TOTAL_LIMIT}" \
  --gradient-accumulation-steps "${GRAD_ACCUM_STEPS}" \
  --generation-batch-size "${GENERATION_BATCH_SIZE}" \
  --num-generations "${NUM_GENERATIONS}" \
  --max-completion-length "${MAX_COMPLETION_LENGTH}" \
  --learning-rate "${LEARNING_RATE}" \
  --lora-r "${LORA_R}" \
  --lora-alpha "${LORA_ALPHA}" \
  --lora-dropout "${LORA_DROPOUT}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --attn-implementation sdpa \
  --remove-invalid-values true \
  --renormalize-logits true \
  --score-first-json-only "${SCORE_FIRST_JSON_ONLY}" \
  --mask-truncated-completions false \
  --logging-steps 1 \
  --report-to none
