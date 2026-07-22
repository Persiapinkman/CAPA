#!/usr/bin/env bash
set -euo pipefail

STUDY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${MODEL_DIR:-/raid/zkq/artifacts/CAPA/bench_models/Qwen3.5-4B-vllm-fp16}"
PORT="${PORT:-18080}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="/raid/zkq/projects/CAPA/demo/vllm_compat${PYTHONPATH:+:${PYTHONPATH}}"
export VLLM_USE_FLASHINFER_SAMPLER=0
# V1 forces chunked prefill even when the CLI explicitly disables it.  Keep
# both deployments on the V0 scheduler so the frozen scheduling controls hold.
export VLLM_USE_V1=0

exec /raid/zkq/projects/CAPA/.venv-train/bin/python \
  -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name qwen35-4b-bench \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype float16 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --model-impl transformers \
  --trust-remote-code \
  --enforce-eager \
  --reasoning-parser qwen3 \
  --hf-overrides '{"architectures":["TransformersForCausalLM"]}' \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 16 \
  "$@" 2>&1 | tee "$STUDY_DIR/server_4b.log"
