#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_DIR="${MODEL_DIR:-/mnt/zkq/models/Qwen3.5-4B}"
CASES="${CASES:-$ROOT_DIR/training/planner_grpo_seed_v1/cases/planner_grpo_focused_4b_cases.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/planner-grpo-qwen35-4b-focused-full-fsdp}"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/experiments/runs/2026-07-08_4b_fullparam_grpo_vs_ppo_fsdp}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29641}"

export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv-train-cu124/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  if [[ -x "$ROOT_DIR/.venv-train/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv-train/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

mkdir -p "$RUN_DIR/logs"

if [[ ! -f "$CASES" ]]; then
  "$PYTHON_BIN" training/planner_grpo_seed_v1/scripts/build_planner_grpo_focused_cases.py
fi

echo "[$(date '+%F %T')] starting full-parameter Qwen3.5-4B Planner GRPO with FSDP"
echo "base_model=$MODEL_DIR"
echo "cases=$CASES"
echo "output_dir=$OUTPUT_DIR"
echo "run_dir=$RUN_DIR"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "num_processes=$NUM_PROCESSES"

EXTRA_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
fi

"$PYTHON_BIN" -m accelerate.commands.launch \
  --num_processes "$NUM_PROCESSES" \
  --mixed_precision fp16 \
  --use_fsdp \
  --fsdp_sharding_strategy FULL_SHARD \
  --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
  --fsdp_transformer_layer_cls_to_wrap Qwen3_5DecoderLayer \
  --fsdp_backward_prefetch BACKWARD_PRE \
  --fsdp_state_dict_type "${FSDP_STATE_DICT_TYPE:-SHARDED_STATE_DICT}" \
  --fsdp_use_orig_params true \
  --fsdp_activation_checkpointing true \
  --main_process_port "$MAIN_PROCESS_PORT" \
  training/planner_grpo_seed_v1/scripts/train_planner_grpo.py \
  --model-name-or-path "$MODEL_DIR" \
  --cases "$CASES" \
  --output-dir "$OUTPUT_DIR" \
  --num-train-epochs "${NUM_TRAIN_EPOCHS:-1}" \
  --max-steps "${MAX_STEPS:--1}" \
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
  --learning-rate "${LEARNING_RATE:-2e-6}" \
  --num-generations "${NUM_GENERATIONS:-4}" \
  --max-prompt-length "${MAX_PROMPT_LENGTH:-3072}" \
  --max-completion-length "${MAX_COMPLETION_LENGTH:-384}" \
  --temperature "${TEMPERATURE:-0.7}" \
  --top-p "${TOP_P:-0.95}" \
  --save-steps "${SAVE_STEPS:-25}" \
  --save-total-limit "${SAVE_TOTAL_LIMIT:-2}" \
  --logging-steps "${LOGGING_STEPS:-1}" \
  --fp16 true \
  --bf16 false \
  --gradient-checkpointing false \
  --use-lora false \
  --optim "${OPTIM:-adamw_torch}" \
  --fsdp "full_shard auto_wrap" \
  --fsdp-config '{"transformer_layer_cls_to_wrap":["Qwen3_5DecoderLayer"],"activation_checkpointing":true,"use_orig_params":true}' \
  --ddp-find-unused-parameters false \
  --report-to "${REPORT_TO:-none}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$RUN_DIR/logs/grpo_fullparam_fsdp.log"
