#!/usr/bin/env bash
set -euo pipefail

STUDY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${MODEL_DIR:-/raid/zkq/models/Qwen3-32B-vllm}"
PORT="${PORT:-18080}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="/raid/zkq/projects/CAPA/demo/vllm_compat${PYTHONPATH:+:${PYTHONPATH}}"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=1

exec /raid/zkq/models/.venv-vllm-cu124/bin/python \
  -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name qwen3-32b-bench \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype float16 \
  --tensor-parallel-size 4 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --model-impl auto \
  --trust-remote-code \
  --enforce-eager \
  --disable-custom-all-reduce \
  --distributed-executor-backend mp \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 16 \
  "$@" 2>&1 | tee "$STUDY_DIR/server_32b.log"
