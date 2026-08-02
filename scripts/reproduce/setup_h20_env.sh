#!/usr/bin/env bash
# Provision two virtual environments for H20 execution.
#
#   .venv-h20-infer   vLLM 0.7.x + HF hub + OpenAI client. Used for serving
#                     Qwen3.5-4B and Qwen3.5-35B-A3B and for driving evaluations.
#   .venv-qwen35-grpo trainer stack pinned by the existing scripts. Rebuild only
#                     when missing; contents match configs/environments/trl-cu124.
#
# Both envs are Python 3.10 + CUDA 12.4 build of PyTorch, which the H20 driver
# stack (>=535 with CUDA 12.4 runtime) exposes as sm_90 (Hopper).
#
# Usage:
#   bash scripts/reproduce/setup_h20_env.sh          # provision both
#   TARGET=infer bash scripts/reproduce/setup_h20_env.sh
#   TARGET=train bash scripts/reproduce/setup_h20_env.sh
#
# Env overrides:
#   PYTHON_BIN     python3.10 binary to bootstrap the venvs (auto-discovered)
#   VLLM_VERSION   default 0.7.3   (works with transformers>=4.57 and Qwen3-MoE)
#   FLASH_ATTN     default 2.7.4.post1
#   TORCH_INDEX    default https://download.pytorch.org/whl/cu124

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

TARGET="${TARGET:-both}"                # infer | train | both
VLLM_VERSION="${VLLM_VERSION:-0.7.3}"
FLASH_ATTN="${FLASH_ATTN:-2.7.4.post1}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3.10 || command -v python3)}"
[[ -x "${PYTHON_BIN}" ]] || { echo "python3.10 not found; set PYTHON_BIN"; exit 2; }

log() { printf '[setup_h20_env %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

ensure_infer_env() {
  local env_dir="${ROOT_DIR}/.venv-h20-infer"
  log "provisioning ${env_dir}"
  if [[ ! -x "${env_dir}/bin/python" ]]; then
    "${PYTHON_BIN}" -m venv "${env_dir}"
  fi
  local pip="${env_dir}/bin/pip"
  "${pip}" install -U pip wheel setuptools

  # Torch first, then vLLM. vLLM's own metadata will pull compatible triton.
  "${pip}" install --index-url "${TORCH_INDEX}" "torch==2.6.0+cu124"
  "${pip}" install \
    "vllm==${VLLM_VERSION}" \
    "transformers==4.57.6" \
    "huggingface_hub>=0.28,<1.0" \
    "openai==2.45.0" \
    "httpx==0.28.1" \
    "jsonschema==4.25.1" \
    "pillow==12.3.0" \
    "requests==2.34.2" \
    "fastapi==0.136.3" \
    "uvicorn==0.49.0"
  # flash-attn is optional; vLLM already ships FA2 kernels but installing the
  # wheel enables transformers ``attn_implementation=flash_attention_2`` for the
  # local eval fallback in run_repeated_planner_grpo_eval.py.
  "${pip}" install --no-build-isolation "flash-attn==${FLASH_ATTN}" || \
    log "flash-attn wheel not available; skipping (vLLM still uses FA2 internally)"
  log "infer env ready"
}

ensure_train_env() {
  local env_dir="${ROOT_DIR}/.venv-qwen35-grpo"
  log "provisioning ${env_dir}"
  if [[ ! -x "${env_dir}/bin/python" ]]; then
    "${PYTHON_BIN}" -m venv "${env_dir}"
  fi
  local pip="${env_dir}/bin/pip"
  "${pip}" install -U pip wheel setuptools

  "${pip}" install --index-url "${TORCH_INDEX}" "torch==2.6.0+cu124"
  "${pip}" install \
    "transformers==4.57.6" \
    "trl==1.8.0" \
    "peft==0.19.1" \
    "accelerate==1.14.0" \
    "datasets==5.0.0" \
    "wandb==0.28.0" \
    "openai==2.45.0" \
    "httpx==0.28.1" \
    "pillow==12.3.0" \
    "requests==2.34.2"
  "${pip}" install --no-build-isolation "flash-attn==${FLASH_ATTN}" || \
    log "flash-attn wheel not available; keeping SDPA (H20 still fine)"
  log "train env ready"
}

case "${TARGET}" in
  infer) ensure_infer_env ;;
  train) ensure_train_env ;;
  both)  ensure_infer_env; ensure_train_env ;;
  *) echo "unknown TARGET=${TARGET}"; exit 2 ;;
esac

log "done"
