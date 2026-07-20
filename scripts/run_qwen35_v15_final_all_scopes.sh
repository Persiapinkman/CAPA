#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT_DIR}"; export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${PYTHON_BIN:-/raid/zkq/projects/CAPA/.venv-qwen35-grpo/bin/python}"
CONFIG="${CONFIG:-configs/eval/qwen35_v15_final_ladder.json}"; EXPECTED_CONFIG_SHA="063932854510fafe7b7952ed8a3f2d937bbdbf98f78710b3dec849b8ebfacae0"
AUDITOR="training/planner_grpo_seed_v1/scripts/audit_qwen35_v15_final_ladder.py"; LOCAL="scripts/run_qwen35_v15_local_4b_final_eval.sh"; LARGER="scripts/run_qwen35_v15_35b_final_eval.sh"
[[ "$(sha256sum "${CONFIG}" | awk '{print $1}')" == "${EXPECTED_CONFIG_SHA}" ]] || exit 1
RUN_ROOT="$(jq -r '.execution.output_root' "${CONFIG}")"; OPENING_RECEIPT="$(jq -r '.execution.opening_receipt' "${CONFIG}")"; [[ ! -e "${RUN_ROOT}" ]] || { echo "Refusing to reuse V15 output root"; exit 1; }
for arm in qwen35_4b_base qwen35_4b_sft qwen35_4b_grpo_n64; do
  gpu="$(jq -r --arg arm "${arm}" '.execution.local_arm_gpu[$arm]' "${CONFIG}")"; [[ "$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | wc -l)" -eq 0 ]] || { echo "GPU ${gpu} busy before V15 opening"; exit 1; }
done
API_BASE="$(jq -r '.arms.qwen35_35b_a3b.api_base' "${CONFIG}")"; MODEL="$(jq -r '.arms.qwen35_35b_a3b.model_id' "${CONFIG}")"
curl --silent --show-error --fail --max-time 20 "${API_BASE}/models" | "${PYTHON_BIN}" -c 'import json,sys;d=json.load(sys.stdin);expected=sys.argv[1];assert any(x.get("id")==expected for x in d.get("data",[]))' "${MODEL}"
"${PYTHON_BIN}" "${AUDITOR}" open --config "${CONFIG}" --output "${OPENING_RECEIPT}" >/dev/null
CONFIG="${CONFIG}" RUN_ROOT="${RUN_ROOT}" OPENING_RECEIPT="${OPENING_RECEIPT}" "${LOCAL}" > "${RUN_ROOT}/local_scope_stdout.log" 2>&1 & local_pid="$!"
CONFIG="${CONFIG}" RUN_ROOT="${RUN_ROOT}" OPENING_RECEIPT="${OPENING_RECEIPT}" "${LARGER}" > "${RUN_ROOT}/larger_scope_stdout.log" 2>&1 & larger_pid="$!"
status=0; wait "${local_pid}" || status=1; wait "${larger_pid}" || status=1
[[ "${status}" -eq 0 ]] || { echo "V15 execution incomplete; outputs quarantined and no selective rerun permitted"; exit 1; }
"${PYTHON_BIN}" "${AUDITOR}" aggregate --config "${CONFIG}" --opening-receipt "${OPENING_RECEIPT}" --output-root "${RUN_ROOT}" --report-output "${RUN_ROOT}/final_report.json" --table-output "${RUN_ROOT}/final_table.md" > "${RUN_ROOT}/aggregate_stdout.json"
jq -e '.status=="pass" and ([.hard_gates[]] | all)' "${RUN_ROOT}/final_report.json" >/dev/null || { echo "V15 final ladder failed"; jq '{status,hard_gates,table}' "${RUN_ROOT}/final_report.json"; exit 2; }
echo "V15 final ladder passed"; jq '{status,table,hard_gates,grpo_minus_larger_mean_pp}' "${RUN_ROOT}/final_report.json"
