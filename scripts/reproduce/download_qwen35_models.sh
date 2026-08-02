#!/usr/bin/env bash
# Download Qwen3.5-4B and Qwen3.5-35B-A3B weights.
#
# The training / eval scripts hard-code /raid/zkq/models/Qwen3.5-<variant> as
# the on-disk path. The HuggingFace repo id is different from that internal
# alias; override with QWEN35_4B_REPO / QWEN35_35B_REPO.
#
# Default repos are the closest public equivalents shipped by Qwen:
#   QWEN35_4B_REPO  = Qwen/Qwen3-4B
#   QWEN35_35B_REPO = Qwen/Qwen3-30B-A3B  (public MoE stand-in for 35B-A3B)
#
# Env:
#   MODELS_ROOT           default /raid/zkq/models
#   TARGET                4b | 35b | both  (default both)
#   HF_HUB_ENABLE_HF_TRANSFER=1 recommended for large downloads
#   HUGGINGFACE_HUB_TOKEN needed for gated repos
#   VENV_BIN              python bin (default .venv-h20-infer/bin/python)

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

MODELS_ROOT="${MODELS_ROOT:-/raid/zkq/models}"
TARGET="${TARGET:-both}"
QWEN35_4B_REPO="${QWEN35_4B_REPO:-Qwen/Qwen3-4B}"
QWEN35_35B_REPO="${QWEN35_35B_REPO:-Qwen/Qwen3-30B-A3B}"
VENV_BIN="${VENV_BIN:-${ROOT_DIR}/.venv-h20-infer/bin/python}"

if [[ ! -x "${VENV_BIN}" ]]; then
  echo "python venv missing at ${VENV_BIN}; run setup_h20_env.sh first" >&2
  exit 2
fi

mkdir -p "${MODELS_ROOT}"

download_one() {
  local alias="$1" repo_id="$2"
  local local_dir="${MODELS_ROOT}/${alias}"

  if [[ -f "${local_dir}/config.json" ]]; then
    echo "[download] ${alias} already present at ${local_dir}, skipping"
    return 0
  fi
  mkdir -p "${local_dir}"

  echo "[download] ${alias} <- ${repo_id} -> ${local_dir}"
  HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}" \
  "${VENV_BIN}" - <<PY
import os, sys
from huggingface_hub import snapshot_download
try:
    path = snapshot_download(
        repo_id=os.environ["REPO_ID"],
        local_dir=os.environ["LOCAL_DIR"],
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "tokenizer*",
            "*.jinja",
            "*.txt",
            "*.safetensors",
            "*.safetensors.index.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
        ],
    )
    print(f"snapshot_dir={path}")
except Exception as exc:
    print(f"snapshot_download failed: {exc}", file=sys.stderr)
    sys.exit(1)
PY
}

case "${TARGET}" in
  4b)    REPO_ID="${QWEN35_4B_REPO}"  LOCAL_DIR="${MODELS_ROOT}/Qwen3.5-4B"       download_one "Qwen3.5-4B" "${QWEN35_4B_REPO}" ;;
  35b)   REPO_ID="${QWEN35_35B_REPO}" LOCAL_DIR="${MODELS_ROOT}/Qwen3.5-35B-A3B"  download_one "Qwen3.5-35B-A3B" "${QWEN35_35B_REPO}" ;;
  both)
    REPO_ID="${QWEN35_4B_REPO}"  LOCAL_DIR="${MODELS_ROOT}/Qwen3.5-4B"       download_one "Qwen3.5-4B" "${QWEN35_4B_REPO}"
    REPO_ID="${QWEN35_35B_REPO}" LOCAL_DIR="${MODELS_ROOT}/Qwen3.5-35B-A3B"  download_one "Qwen3.5-35B-A3B" "${QWEN35_35B_REPO}"
    ;;
  *) echo "unknown TARGET=${TARGET}"; exit 2 ;;
esac

echo "[download] done"
