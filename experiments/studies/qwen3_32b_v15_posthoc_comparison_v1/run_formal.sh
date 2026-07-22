#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="/raid/zkq/projects/CAPA/.venv-qwen35-grpo/bin/python"
STUDY="experiments/studies/qwen3_32b_v15_posthoc_comparison_v1"
CASES="experiments/studies/planner_retry_ladder_v15_confirmation_v1/sealed_data/v15_confirmation_cases.jsonl"
RUN_ROOT="${RUN_ROOT:-/raid/zkq/artifacts/CAPA/evals/qwen3_32b_v15_posthoc_comparison_v1/formal_20260722T072244Z}"
API_BASE="http://127.0.0.1:18081/v1"
MODEL="qwen3-32b-v15-posthoc"
PREFIX="qwen3_32b"
REPEATED="training/planner_grpo_seed_v1/scripts/run_repeated_planner_grpo_eval.py"
COMBINER="training/planner_grpo_seed_v1/scripts/combine_planner_rollout_prediction_shards.py"
REWARD="training/planner_grpo_seed_v1/scripts/reward_planner_grpo.py"

check_sha() {
  local expected="$1" path="$2" actual
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || { echo "SHA mismatch: ${path}"; exit 1; }
}

check_sha 37d9a739585c012041397fba77b995d5842359ba5e23a8e8f8f604578e4ade78 "${CASES}"
check_sha 4bc4a819cf6f2291ac125c33b0a42133c679d1dbbb21d1dc54bb0337db8dcfd7 /raid/zkq/artifacts/CAPA/final/planner_retry_ladder_v15_n67/final_open_once/final_report.json
check_sha 97e295b63283935788fac5e4f8860862a56d4089538cafc93f0431f2ebe483bb /raid/zkq/models/Qwen3-32B-vllm/config.json

[[ ! -e "${RUN_ROOT}" ]] || { echo "Refusing to overwrite ${RUN_ROOT}"; exit 1; }
mkdir -p "${RUN_ROOT}/shards"
cp "${STUDY}/config.json" "${STUDY}/PROTOCOL.md" "${RUN_ROOT}/"
curl --silent --show-error --fail --max-time 20 "${API_BASE}/models" > "${RUN_ROOT}/models.json"
"${PYTHON_BIN}" - "${RUN_ROOT}/models.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
assert any(x.get("id") == "qwen3-32b-v15-posthoc" for x in d.get("data", []))
PY

pids=()
for shard in 0 1 2 3; do
  shard_dir="${RUN_ROOT}/shards/shard${shard}"
  mkdir -p "${shard_dir}"
  offset=$((shard * 6))
  CAPA_OMIT_MODEL_IMAGE_PAYLOAD=1 DEMO_OPENAI_STREAM=0 \
    "${PYTHON_BIN}" "${REPEATED}" \
    --cases "${CASES}" \
    --out-dir "${shard_dir}" \
    --report-prefix "${PREFIX}_shard${shard}" \
    --model "${MODEL}" \
    --api-base "${API_BASE}" \
    --api-key local-dummy \
    --runs 3 \
    --offset "${offset}" \
    --limit 6 \
    --max-steps 3 \
    --max-tokens 4096 \
    --temperature 0 \
    --top-p 1 \
    --seed 42 \
    --do-sample false \
    --timeout-seconds 600 \
    --openai-timeout-seconds 600 \
    > "${shard_dir}/runner_stdout.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
[[ "${status}" -eq 0 ]] || { echo "Formal 32B batch incomplete; selective rerun forbidden"; exit 1; }

for run in 1 2 3; do
  "${PYTHON_BIN}" "${COMBINER}" \
    --cases "${CASES}" \
    --predictions \
      "${RUN_ROOT}/shards/shard0/${PREFIX}_shard0_run${run}_predictions.jsonl" \
      "${RUN_ROOT}/shards/shard1/${PREFIX}_shard1_run${run}_predictions.jsonl" \
      "${RUN_ROOT}/shards/shard2/${PREFIX}_shard2_run${run}_predictions.jsonl" \
      "${RUN_ROOT}/shards/shard3/${PREFIX}_shard3_run${run}_predictions.jsonl" \
    --output "${RUN_ROOT}/${PREFIX}_run${run}_predictions.jsonl"
  "${PYTHON_BIN}" "${REWARD}" \
    --cases "${CASES}" \
    --predictions "${RUN_ROOT}/${PREFIX}_run${run}_predictions.jsonl" \
    --out "${RUN_ROOT}/${PREFIX}_run${run}_reward.json"
done

"${PYTHON_BIN}" "${STUDY}/analyze_comparison.py" \
  --config "${STUDY}/config.json" \
  --output-root "${RUN_ROOT}" \
  --json-output "${RUN_ROOT}/comparison_report.json" \
  --markdown-output "${RUN_ROOT}/comparison_table.md"
