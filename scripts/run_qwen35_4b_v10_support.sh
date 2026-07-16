#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-qwen35-grpo/bin/python}"
RUN_DIR="${RUN_DIR:-experiments/runs/20260716_qwen35_4b_v10_support4x_sft100}"
STUDY_DIR="experiments/studies/planner_retry_anti_forgetting_v10_qwen35_4b_v1"
DATASET_DIR="data/datasets/planner_retry_anti_forgetting_v10"
STEP_DATA="training/planner_grpo_seed_v1/step_data/planner_retry_anti_forgetting_v10_support_dev_qwen35_4b_nothinking_mixed_steps.jsonl"
ADAPTER="experiments/runs/20260716_qwen35_4b_planner_v6_sft_seed42_v1/checkpoint-100"
GPUS=(3 4 5 7)

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
[[ -f "${ADAPTER}/adapter_model.safetensors" ]] || { echo "Missing ${ADAPTER}" >&2; exit 1; }

pids=()
for shard in 0 1 2 3; do
  shard_dir="${RUN_DIR}/shard${shard}"
  mkdir -p "${shard_dir}"
  CUDA_VISIBLE_DEVICES="${GPUS[$shard]}" "${PYTHON_BIN}" \
    training/planner_grpo_seed_v1/scripts/eval_qwen35_planner_v6_step_support.py \
    --model-name-or-path /raid/zkq/models/Qwen3.5-4B \
    --adapter-path "${ADAPTER}" \
    --step-data "${STEP_DATA}" \
    --samples-out "${shard_dir}/samples.jsonl" \
    --summary-out "${shard_dir}/summary.json" \
    --shard-index "${shard}" --num-shards 4 \
    --samples-per-prompt 4 --max-new-tokens 320 \
    --temperature 0.9 --top-p 0.9 --seed 42 \
    > "${shard_dir}/stdout.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
[[ "${status}" -eq 0 ]] || { echo "V10 support sampling failed" >&2; exit 1; }

samples=()
for shard in 0 1 2 3; do
  samples+=("${RUN_DIR}/shard${shard}/samples.jsonl")
done

"${PYTHON_BIN}" training/planner_grpo_seed_v1/scripts/combine_qwen35_planner_v6_step_support.py \
  --samples "${samples[@]}" \
  --samples-out "${RUN_DIR}/samples.jsonl" \
  --summary-out "${RUN_DIR}/summary.json"

"${PYTHON_BIN}" training/planner_grpo_seed_v1/scripts/gate_planner_retry_anti_forgetting_v10_support.py \
  --preregistration "${STUDY_DIR}/preregistration.json" \
  --step-data "${STEP_DATA}" \
  --samples "${RUN_DIR}/samples.jsonl" \
  --output "${STUDY_DIR}/support_decision.json" \
  --accepted-scenarios-out "${DATASET_DIR}/accepted_optimizer_scenarios.txt"

jq -e '.status == "pass" and .optimizer_authorized == true' \
  "${STUDY_DIR}/support_decision.json" >/dev/null
