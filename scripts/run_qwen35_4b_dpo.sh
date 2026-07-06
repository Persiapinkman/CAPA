#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_DIR="${MODEL_DIR:-/mnt/zkq/models/Qwen3.5-4B}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/planner-dpo-qwen35-4b-lora}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[$(date '+%F %T')] starting Qwen3.5-4B Planner DPO"
echo "base_model=$MODEL_DIR"
echo "output_dir=$OUTPUT_DIR"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"

exec .venv-train/bin/python demo/eval/train_planner_dpo.py \
  --model-name-or-path "$MODEL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --num-train-epochs 1 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 5e-6 \
  --beta 0.1 \
  --max-length 1024 \
  --max-prompt-length 768 \
  --max-completion-length 256 \
  --save-steps 25 \
  --save-total-limit 2 \
  --eval-steps 10 \
  --logging-steps 1 \
  --fp16 false \
  --bf16 false \
  --use-lora true \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --lora-target-modules q_proj,k_proj,v_proj,o_proj \
  --report-to tensorboard
