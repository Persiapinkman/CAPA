#!/bin/bash
# vLLM serve for RexOmni. Called inside Docker with: PORT MODEL_PATH TOKENIZER_PATH CUDA_VISIBLE_DEVICES
REXOMNI_PORT=$1
REXOMNI_MODEL_PATH=$2
REXOMNI_TOKENIZER_PATH=$3
REXOMNI_CUDA_VISIBLE_DEVICES=$4

CUDA_VISIBLE_DEVICES=${REXOMNI_CUDA_VISIBLE_DEVICES} vllm serve ${REXOMNI_MODEL_PATH} --tokenizer ${REXOMNI_TOKENIZER_PATH} --tokenizer-mode slow --max-model-len 4096 --gpu-memory-utilization 0.6 --tensor-parallel-size 1 --trust-remote-code --limit-mm-per-prompt image=10 --port ${REXOMNI_PORT} --host 0.0.0.0 --served-model-name RexOmni --enforce-eager
