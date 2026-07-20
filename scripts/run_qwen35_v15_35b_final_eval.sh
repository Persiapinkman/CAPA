#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT_DIR}"; export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${PYTHON_BIN:-/raid/zkq/projects/CAPA/.venv-qwen35-grpo/bin/python}"
CONFIG="${CONFIG:-configs/eval/qwen35_v15_final_ladder.json}"
AUDITOR="training/planner_grpo_seed_v1/scripts/audit_qwen35_v15_final_ladder.py"
REPEATED="training/planner_grpo_seed_v1/scripts/run_repeated_planner_grpo_eval.py"
COMBINER="training/planner_grpo_seed_v1/scripts/combine_planner_rollout_prediction_shards.py"
RUN_ROOT="${RUN_ROOT:?set RUN_ROOT}"; OPENING_RECEIPT="${OPENING_RECEIPT:?set OPENING_RECEIPT}"
"${PYTHON_BIN}" "${AUDITOR}" verify-opening --config "${CONFIG}" --opening-receipt "${OPENING_RECEIPT}" >/dev/null
CASES="$(jq -r '.sealed_confirmation.cases_path' "${CONFIG}")"; [[ "${CASES}" = /* ]] || CASES="${ROOT_DIR}/${CASES}"
MODEL="$(jq -r '.arms.qwen35_35b_a3b.model_id' "${CONFIG}")"; API_BASE="$(jq -r '.arms.qwen35_35b_a3b.api_base' "${CONFIG}")"; PREFIX="$(jq -r '.arms.qwen35_35b_a3b.report_prefix' "${CONFIG}")"
ARM_DIR="${RUN_ROOT}/raw/qwen35_35b_a3b"; [[ ! -e "${ARM_DIR}" ]] || exit 1; mkdir -p "${ARM_DIR}/shards"
curl --silent --show-error --fail --max-time 20 "${API_BASE}/models" | "${PYTHON_BIN}" -c 'import json,sys;d=json.load(sys.stdin);expected=sys.argv[1];assert any(x.get("id")==expected for x in d.get("data",[]))' "${MODEL}"
pids=()
for shard in 0 1 2 3; do
  shard_dir="${ARM_DIR}/shards/shard${shard}"; mkdir -p "${shard_dir}"; offset=$((shard * 6))
  CAPA_OMIT_MODEL_IMAGE_PAYLOAD=1 DEMO_OPENAI_STREAM=0 "${PYTHON_BIN}" "${REPEATED}" --cases "${CASES}" --out-dir "${shard_dir}" --report-prefix "${PREFIX}_shard${shard}" --model "${MODEL}" --api-base "${API_BASE}" --runs 3 --offset "${offset}" --limit 6 --max-steps 3 --max-tokens 4096 --temperature 0 --top-p 1 --seed 42 --do-sample false --timeout-seconds 600 --openai-timeout-seconds 600 > "${shard_dir}/runner_stdout.log" 2>&1 & pids+=("$!")
done
status=0; for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
[[ "${status}" -eq 0 ]] || { echo "V15 35B execution incomplete; selective rerun forbidden"; exit 1; }
for run in 1 2 3; do
  "${PYTHON_BIN}" "${COMBINER}" --cases "${CASES}" --predictions "${ARM_DIR}/shards/shard0/${PREFIX}_shard0_run${run}_predictions.jsonl" "${ARM_DIR}/shards/shard1/${PREFIX}_shard1_run${run}_predictions.jsonl" "${ARM_DIR}/shards/shard2/${PREFIX}_shard2_run${run}_predictions.jsonl" "${ARM_DIR}/shards/shard3/${PREFIX}_shard3_run${run}_predictions.jsonl" --output "${ARM_DIR}/${PREFIX}_run${run}_predictions.jsonl"
done
echo "V15 35B execution complete"
