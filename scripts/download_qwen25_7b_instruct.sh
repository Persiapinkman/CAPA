#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
LOCAL_DIR="${LOCAL_DIR:-/raid/zkq/models/Qwen2.5-7B-Instruct}"

mkdir -p "$LOCAL_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv-train-cu124/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

"$PYTHON_BIN" - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="${MODEL_ID}",
    local_dir="${LOCAL_DIR}",
)
PY

echo "model_dir=$LOCAL_DIR"
