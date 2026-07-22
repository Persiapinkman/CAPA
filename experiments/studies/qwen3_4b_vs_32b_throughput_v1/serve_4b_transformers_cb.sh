#!/usr/bin/env bash
set -euo pipefail

STUDY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${MODEL_DIR:-/raid/zkq/artifacts/CAPA/bench_models/Qwen3.5-4B-text-only-fp16}"
PORT="${PORT:-18080}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="/raid/zkq/projects/CAPA/demo/vllm_compat${PYTHONPATH:+:${PYTHONPATH}}"

exec /raid/zkq/projects/CAPA/.venv-train/bin/python \
  -m transformers.cli.transformers serve "$MODEL_DIR" \
  --continuous-batching \
  --attn-implementation sdpa \
  --device cuda:0 \
  --dtype float16 \
  --reasoning off \
  --no-compile \
  --cb-block-size 16 \
  --cb-max-batch-tokens 8192 \
  --cb-max-memory-percent 0.85 \
  --no-cb-use-cuda-graph \
  --host 127.0.0.1 \
  --port "$PORT" \
  --log-level info 2>&1 | tee "$STUDY_DIR/server_4b_transformers_cb.log"
