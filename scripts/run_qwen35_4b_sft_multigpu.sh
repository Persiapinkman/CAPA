#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_DIR="${MODEL_DIR:-/mnt/zkq/models/Qwen3.5-4B}"
TRAIN_FILE="${TRAIN_FILE:-$ROOT_DIR/training/planner_dpo_train_seed_v1/training_data/planner_dpo_text_train.jsonl}"
VAL_FILE="${VAL_FILE:-$ROOT_DIR/training/planner_dpo_train_seed_v1/training_data/planner_dpo_text_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/planner-sft-qwen35-4b-chosen-lora}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-5}"
MASTER_PORT="${MASTER_PORT:-29531}"
DDP_BACKEND="${DDP_BACKEND:-gloo}"

export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "[$(date '+%F %T')] starting Qwen3.5-4B Planner SFT"
echo "base_model=$MODEL_DIR"
echo "train_file=$TRAIN_FILE"
echo "val_file=$VAL_FILE"
echo "output_dir=$OUTPUT_DIR"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "nproc_per_node=$NPROC_PER_NODE"
echo "ddp_backend=$DDP_BACKEND"

exec .venv-train/bin/torchrun \
  --standalone \
  --nnodes 1 \
  --nproc_per_node "$NPROC_PER_NODE" \
  --master_port "$MASTER_PORT" \
  demo/eval/train_planner_sft.py \
    --model-name-or-path "$MODEL_DIR" \
    --train-file "$TRAIN_FILE" \
    --validation-file "$VAL_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --num-train-epochs 1 \
    --per-device-train-batch-size 1 \
    --per-device-eval-batch-size 1 \
    --gradient-accumulation-steps 1 \
    --learning-rate 1e-5 \
    --max-length 1024 \
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
    --report-to tensorboard \
    --ddp-backend "$DDP_BACKEND"
