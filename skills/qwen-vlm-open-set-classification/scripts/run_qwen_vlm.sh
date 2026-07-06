#!/bin/bash
# Start Qwen VLM vLLM service. Used inside Docker by start_qwen_vlm_service.sh.
# Env: VLM_DEVICES, VLM_MODEL_PATH, VLM_PORT (default 9012)

VLM_DEVICES=${VLM_DEVICES:-0,1}
VLM_MODEL_PATH=${VLM_MODEL_PATH:-/workspace/models/Qwen2.5-VL-7B-Instruct}
VLM_PORT=${VLM_PORT:-9012}

CUDA_VISIBLE_DEVICES=${VLM_DEVICES} vllm serve ${VLM_MODEL_PATH} --max-model-len 15000 --served-model-name Qwen2.5-VL-7B-Instruct --dtype half --port ${VLM_PORT} --tensor-parallel-size 2 --pipeline-parallel-size 1 --limit_mm_per_prompt image=20 --gpu_memory_utilization=0.8
