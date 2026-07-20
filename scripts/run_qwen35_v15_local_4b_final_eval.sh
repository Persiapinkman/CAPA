#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT_DIR}"; export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${PYTHON_BIN:-/raid/zkq/projects/CAPA/.venv-qwen35-grpo/bin/python}"
CONFIG="${CONFIG:-configs/eval/qwen35_v15_final_ladder.json}"
AUDITOR="training/planner_grpo_seed_v1/scripts/audit_qwen35_v15_final_ladder.py"
REPEATED="training/planner_grpo_seed_v1/scripts/run_repeated_planner_grpo_eval.py"
RUN_ROOT="${RUN_ROOT:?set RUN_ROOT}"; OPENING_RECEIPT="${OPENING_RECEIPT:?set OPENING_RECEIPT}"
"${PYTHON_BIN}" "${AUDITOR}" verify-opening --config "${CONFIG}" --opening-receipt "${OPENING_RECEIPT}" >/dev/null
CASES="$(jq -r '.sealed_confirmation.cases_path' "${CONFIG}")"; [[ "${CASES}" = /* ]] || CASES="${ROOT_DIR}/${CASES}"
run_arm() {
  local arm="$1" gpu="$2" arm_dir="${RUN_ROOT}/raw/$1" model_path adapter_path prefix
  model_path="$(jq -r --arg arm "${arm}" '.arms[$arm].model_path' "${CONFIG}")"
  adapter_path="$(jq -r --arg arm "${arm}" '.arms[$arm].adapter_path // ""' "${CONFIG}")"
  prefix="$(jq -r --arg arm "${arm}" '.arms[$arm].report_prefix' "${CONFIG}")"
  [[ ! -e "${arm_dir}" ]] || { echo "Refusing to overwrite ${arm_dir}"; return 1; }
  mkdir -p "${arm_dir}"
  cmd=("${PYTHON_BIN}" "${REPEATED}" --cases "${CASES}" --out-dir "${arm_dir}" --report-prefix "${prefix}" --model "${arm}" --api-base http://127.0.0.1:9/v1 --runs 3 --max-steps 3 --max-tokens 4096 --temperature 0 --top-p 1 --seed 42 --do-sample false --timeout-seconds 600 --openai-timeout-seconds 600 --local-model-path "${model_path}" --local-device cuda --local-attn-implementation sdpa)
  [[ -z "${adapter_path}" ]] || cmd+=(--local-adapter-path "${adapter_path}")
  CUDA_VISIBLE_DEVICES="${gpu}" CAPA_OMIT_MODEL_IMAGE_PAYLOAD=1 DEMO_OPENAI_STREAM=0 "${cmd[@]}" > "${arm_dir}/runner_stdout.log" 2>&1
}
pids=()
for arm in qwen35_4b_base qwen35_4b_sft qwen35_4b_grpo_n64; do
  run_arm "${arm}" "$(jq -r --arg arm "${arm}" '.execution.local_arm_gpu[$arm]' "${CONFIG}")" & pids+=("$!")
done
status=0; for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
[[ "${status}" -eq 0 ]] || { echo "V15 local execution incomplete; selective rerun forbidden"; exit 1; }
echo "V15 local three-arm execution complete"
