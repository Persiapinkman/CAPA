#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/path_utils.sh"

MODEL_DIR="$(resolve_model_dir "${MODEL_DIR:-/raid/zkq/models/Qwen3.5-4B}")"
CASES="${CASES:-$ROOT_DIR/training/planner_grpo_seed_v1/cases/planner_grpo_focused_4b_cases.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/planner-grpo-qwen35-4b-focused-lora}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv-train/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

if [[ ! -f "$CASES" ]]; then
  "$PYTHON_BIN" training/planner_grpo_seed_v1/scripts/build_planner_grpo_focused_cases.py
fi

echo "[$(date '+%F %T')] starting Qwen3.5-4B focused Planner GRPO"
echo "base_model=$MODEL_DIR"
echo "cases=$CASES"
echo "output_dir=$OUTPUT_DIR"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  echo "resume_from_checkpoint=$RESUME_FROM_CHECKPOINT"
fi

EXTRA_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
fi

exec "$PYTHON_BIN" training/planner_grpo_seed_v1/scripts/train_planner_grpo.py \
  --model-name-or-path "$MODEL_DIR" \
  --cases "$CASES" \
  --output-dir "$OUTPUT_DIR" \
  --num-train-epochs 1 \
  --max-steps "${MAX_STEPS:--1}" \
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
  --learning-rate "${LEARNING_RATE:-5e-6}" \
  --num-generations "${NUM_GENERATIONS:-4}" \
  --max-prompt-length "${MAX_PROMPT_LENGTH:-3072}" \
  --max-completion-length "${MAX_COMPLETION_LENGTH:-512}" \
  --temperature "${TEMPERATURE:-0.7}" \
  --top-p "${TOP_P:-0.95}" \
  --save-steps "${SAVE_STEPS:-25}" \
  --save-total-limit "${SAVE_TOTAL_LIMIT:-2}" \
  --logging-steps "${LOGGING_STEPS:-1}" \
  --fp16 "${FP16:-true}" \
  --bf16 "${BF16:-false}" \
  --use-lora "${USE_LORA:-true}" \
  --lora-r "${LORA_R:-16}" \
  --lora-alpha "${LORA_ALPHA:-32}" \
  --lora-dropout "${LORA_DROPOUT:-0.05}" \
  --lora-target-modules "${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}" \
  --report-to "${REPORT_TO:-tensorboard}" \
  "${EXTRA_ARGS[@]}"
