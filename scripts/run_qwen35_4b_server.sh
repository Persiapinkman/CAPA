#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/path_utils.sh"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-4B}"
MODEL_DIR="$(resolve_model_dir "${MODEL_DIR:-/raid/zkq/models/Qwen3.5-4B}")"
MODEL_SOURCE="${MODEL_SOURCE:-modelscope}"
MAX_WORKERS="${MAX_WORKERS:-8}"
HF_HOME="${HF_HOME:-$ROOT_DIR/.hf-cache}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export HF_HOME
export CUDA_VISIBLE_DEVICES
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"

mkdir -p "$MODEL_DIR"

if [[ ! -f "$MODEL_DIR/model.safetensors-00001-of-00002.safetensors" || ! -f "$MODEL_DIR/model.safetensors-00002-of-00002.safetensors" ]]; then
  echo "[$(date '+%F %T')] downloading $MODEL_ID from $MODEL_SOURCE -> $MODEL_DIR"
  if [[ "$MODEL_SOURCE" == "modelscope" ]]; then
    .venv-train/bin/python - <<PY
from modelscope import snapshot_download

snapshot_download(
    model_id="${MODEL_ID}",
    local_dir="${MODEL_DIR}",
    max_workers=int("${MAX_WORKERS}"),
)
PY
  else
    .venv-train/bin/python - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="${MODEL_ID}",
    local_dir="${MODEL_DIR}",
    local_dir_use_symlinks=False,
    resume_download=True,
)
PY
  fi
fi

echo "[$(date '+%F %T')] starting Qwen server"
echo "model_dir=$MODEL_DIR"
echo "host=$HOST"
echo "port=$PORT"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"

exec .venv-train/bin/python demo/deploy_qwen_server.py \
  --model-path "$MODEL_DIR" \
  --model-name qwen3.5-4b \
  --host "$HOST" \
  --port "$PORT" \
  --dtype float16 \
  --device-map auto \
  --trust-remote-code
