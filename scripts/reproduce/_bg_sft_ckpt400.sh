#!/usr/bin/env bash
# Compare SFT ckpt-100 vs ckpt-400 on softbnd_dev (1 run).
set -euo pipefail
cd /apdcephfs_hzlf/share_1227201/zkq/projects/CAPA

LOG_DIR=/apdcephfs_hzlf/share_1227201/zkq/capa_h20/artifacts/CAPA/logs/train
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/sft_ckpt400_eval_${STAMP}.log"
exec > "${LOG_FILE}" 2>&1

echo "[merge] ckpt-400"
.venv-qwen35-grpo/bin/python scripts/merge_lora_adapter.py \
  --base-model /apdcephfs_hzlf/share_1227201/zkq/capa_h20/models/Qwen3.5-4B \
  --adapter /apdcephfs_hzlf/share_1227201/zkq/capa_h20/artifacts/CAPA/repro_h20/sft/20260802_155804_qwen35_4b_planner_v6_sft/checkpoint-400 \
  --output-dir /apdcephfs_hzlf/share_1227201/zkq/capa_h20/artifacts/CAPA/repro_h20/sft/20260802_155804_qwen35_4b_planner_v6_sft/checkpoint-400_merged

echo "[serve] vLLM ckpt-400"
export CUDA_VISIBLE_DEVICES=0
.venv-h20-infer/bin/python -m vllm.entrypoints.openai.api_server \
  --model /apdcephfs_hzlf/share_1227201/zkq/capa_h20/artifacts/CAPA/repro_h20/sft/20260802_155804_qwen35_4b_planner_v6_sft/checkpoint-400_merged \
  --served-model-name qwen35_4b_sft_ckpt400 \
  --host 127.0.0.1 --port 8001 \
  --dtype bfloat16 --tensor-parallel-size 1 \
  --max-model-len 12288 --gpu-memory-utilization 0.90 \
  --trust-remote-code &
VLLM_PID=$!
echo $VLLM_PID > /tmp/vllm_ckpt400.pid

echo "[wait] serve ready"
for i in $(seq 1 60); do
  sleep 5
  if curl -sf http://127.0.0.1:8001/v1/models >/dev/null 2>&1; then echo "ready after ${i}*5s"; break; fi
done

echo "[eval] softbnd_dev 3 run"
CAPA_OMIT_MODEL_IMAGE_PAYLOAD=1 \
.venv-h20-infer/bin/python training/planner_grpo_seed_v1/scripts/run_repeated_planner_grpo_eval.py \
  --cases training/planner_grpo_seed_v1/cases/planner_retry_migrate_v7_longobs_grpo_dev_cases.jsonl \
  --out-dir /apdcephfs_hzlf/share_1227201/zkq/capa_h20/artifacts/CAPA/repro_h20/eval/sft_ckpt400_v7_3run \
  --report-prefix sft_ckpt400 \
  --model qwen35_4b_sft_ckpt400 \
  --api-base http://127.0.0.1:8001/v1 \
  --runs 3 --max-steps 3 --max-tokens 512 \
  --temperature 0 --top-p 1 --seed 42 --do-sample false \
  --timeout-seconds 600 --openai-timeout-seconds 600

echo "[stop] vllm"
kill -9 $VLLM_PID 2>&1 || true
