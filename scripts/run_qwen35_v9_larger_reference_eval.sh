#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CASES="${CASES:?set CASES to the frozen selection or sealed case JSONL}"
OUT_DIR="${OUT_DIR:?set OUT_DIR for the larger-reference artifacts}"
REPORT_PREFIX="${REPORT_PREFIX:-qwen35_35b_v9_reference_t0}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
MODEL="${MODEL:-Qwen3.5-35B-A3B}"
API_BASE="${API_BASE:-http://10.111.32.253:8000/v1}"
EXPECTED_ROWS="${EXPECTED_ROWS:?set EXPECTED_ROWS to the frozen case count}"
NUM_SHARDS=4

if (( EXPECTED_ROWS % NUM_SHARDS != 0 )); then
  echo "EXPECTED_ROWS must be divisible by ${NUM_SHARDS}" >&2
  exit 1
fi
LIMIT=$((EXPECTED_ROWS / NUM_SHARDS))
mkdir -p "${OUT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export CAPA_OMIT_MODEL_IMAGE_PAYLOAD=1
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost,10.111.32.253}"
export no_proxy="${no_proxy:-${NO_PROXY}}"

pids=()
for shard in 0 1 2 3; do
  shard_dir="${OUT_DIR}/shard${shard}"
  mkdir -p "${shard_dir}"
  offset=$((shard * LIMIT))
  "${PYTHON_BIN}" training/planner_grpo_seed_v1/scripts/run_repeated_planner_grpo_eval.py \
    --cases "${CASES}" --out-dir "${shard_dir}" \
    --report-prefix "${REPORT_PREFIX}_shard${shard}" \
    --model "${MODEL}" --api-base "${API_BASE}" --runs 1 \
    --offset "${offset}" --limit "${LIMIT}" \
    --max-steps 3 --max-tokens 2048 --temperature 0 --top-p 1 --seed 42 \
    --do-sample false --timeout-seconds 300 --openai-timeout-seconds 300 \
    > "${shard_dir}/stdout.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
[[ "${status}" -eq 0 ]] || { echo "Larger-reference shard failed" >&2; exit 1; }

predictions=()
for shard in 0 1 2 3; do
  predictions+=("${OUT_DIR}/shard${shard}/${REPORT_PREFIX}_shard${shard}_run1_predictions.jsonl")
done
"${PYTHON_BIN}" training/planner_grpo_seed_v1/scripts/combine_planner_rollout_prediction_shards.py \
  --cases "${CASES}" --predictions "${predictions[@]}" \
  --output "${OUT_DIR}/predictions.jsonl"
"${PYTHON_BIN}" training/planner_grpo_seed_v1/scripts/reward_planner_grpo.py \
  --cases "${CASES}" --predictions "${OUT_DIR}/predictions.jsonl" \
  --out "${OUT_DIR}/reward.json"
