#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODE="${MODE:-primary}"
CASES="${CASES:-/raid/zkq/artifacts/CAPA/arbor/ladder_n8/v12_open_dev/base_ladder_t0/prepared/selected_cases.jsonl}"
EXPECTED_CASES_SHA256="bb895cb2cc6bbe1ef0b402750eb6abd120174f9bf637e2d4511c116dd43ce985"
EXPECTED_ROWS=96
SHARDS=4
SHARD_SIZE=24
RUNS=3
OUT_DIR="${OUT_DIR:-/raid/zkq/artifacts/CAPA/arbor/ladder_n10/v12_open_dev/35b_t0_max320_3x}"
CONFIG="${CONFIG:-${ROOT_DIR}/configs/eval/qwen35_v12_35b_stability_n10.json}"
PYTHON_BIN="${PYTHON_BIN:-/raid/zkq/projects/CAPA/.venv-qwen35-grpo/bin/python}"
MODEL="Qwen3.5-35B-A3B"
API_BASE="http://10.111.32.253:8000/v1"
PREFIX="n10_qwen35_35b"

[[ -f "${CASES}" ]] || { echo "Missing frozen cases: ${CASES}" >&2; exit 1; }
[[ -f "${CONFIG}" ]] || { echo "Missing n10 config: ${CONFIG}" >&2; exit 1; }
[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python executable: ${PYTHON_BIN}" >&2; exit 1; }
[[ "$(wc -l < "${CASES}")" -eq "${EXPECTED_ROWS}" ]] || {
  echo "Frozen case row-count mismatch" >&2
  exit 1
}
[[ "$(sha256sum "${CASES}" | awk '{print $1}')" == "${EXPECTED_CASES_SHA256}" ]] || {
  echo "Frozen case SHA-256 mismatch" >&2
  exit 1
}
[[ "${MODE}" == "primary" || "${MODE}" == "retry_once" ]] || {
  echo "MODE must be primary or retry_once" >&2
  exit 1
}

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export CAPA_OMIT_MODEL_IMAGE_PAYLOAD=1
export DEMO_OPENAI_STREAM=0

# Fail fast if the predeclared model is no longer advertised by the fixed gateway.
curl --silent --show-error --fail --max-time 20 "${API_BASE}/models" \
  | "${PYTHON_BIN}" -c \
    'import json,sys; d=json.load(sys.stdin); assert any(x.get("id")=="Qwen3.5-35B-A3B" for x in d.get("data",[]))'

run_eval() {
  local cases_path="$1"
  local out_dir="$2"
  local report_prefix="$3"
  local runs="$4"
  local offset="${5:-0}"
  local limit="${6:-0}"
  mkdir -p "${out_dir}"
  "${PYTHON_BIN}" training/planner_grpo_seed_v1/scripts/run_repeated_planner_grpo_eval.py \
    --cases "${cases_path}" --out-dir "${out_dir}" \
    --report-prefix "${report_prefix}" \
    --model "${MODEL}" --api-base "${API_BASE}" --runs "${runs}" \
    --offset "${offset}" --limit "${limit}" \
    --max-steps 3 --max-tokens 320 --temperature 0 --top-p 1 --seed 42 \
    --do-sample false --timeout-seconds 300 --openai-timeout-seconds 300
}

if [[ "${MODE}" == "primary" ]]; then
  [[ ! -e "${OUT_DIR}" ]] || {
    echo "Refusing to overwrite n10 output directory: ${OUT_DIR}" >&2
    exit 1
  }
  mkdir -p "${OUT_DIR}/raw"
  cp "${CONFIG}" "${OUT_DIR}/frozen_config.json"
  sha256sum "${CASES}" "${OUT_DIR}/frozen_config.json" > "${OUT_DIR}/input_sha256.txt"

  pids=()
  for shard in 0 1 2 3; do
    shard_dir="${OUT_DIR}/raw/shard${shard}"
    mkdir -p "${shard_dir}"
    offset=$((shard * SHARD_SIZE))
    run_eval "${CASES}" "${shard_dir}" "${PREFIX}_shard${shard}" \
      "${RUNS}" "${offset}" "${SHARD_SIZE}" \
      > "${shard_dir}/stdout.log" 2>&1 &
    pids+=("$!")
  done

  status=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
  done
  [[ "${status}" -eq 0 ]] || {
    echo "At least one primary shard process failed; raw files are preserved under ${OUT_DIR}/raw" >&2
    exit 1
  }

  "${PYTHON_BIN}" \
    training/planner_grpo_seed_v1/scripts/evaluate_qwen35_v12_35b_stability_n10.py \
    --config "${CONFIG}" --raw-root "${OUT_DIR}/raw" \
    --analysis-dir "${OUT_DIR}/analysis_primary"
  echo "n10 primary experiment complete: ${OUT_DIR}/analysis_primary/stability_report.json"
  exit 0
fi

PLAN="${OUT_DIR}/analysis_primary/transport_retry_plan.json"
RETRY_ROOT="${OUT_DIR}/retry_once"
[[ -f "${PLAN}" ]] || {
  echo "Missing primary transport retry plan: ${PLAN}" >&2
  exit 1
}
[[ ! -e "${RETRY_ROOT}" ]] || {
  echo "Refusing a second or overwriting transport retry: ${RETRY_ROOT}" >&2
  exit 1
}

total_retry="$(${PYTHON_BIN} - "${PLAN}" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
print(sum(len(p["runs"][str(run)]["transport_failure_case_ids"]) for run in (1, 2, 3)))
PY
)"
[[ "${total_retry}" -gt 0 ]] || {
  echo "Primary artifacts contain no predeclared transport failures; retry is forbidden" >&2
  exit 1
}

mkdir -p "${RETRY_ROOT}"
pids=()
for run in 1 2 3; do
  retry_cases="$(${PYTHON_BIN} - "${PLAN}" "${run}" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
print(p["runs"][sys.argv[2]]["retry_cases_path"])
PY
)"
  [[ -n "${retry_cases}" ]] || continue
  retry_dir="${RETRY_ROOT}/run${run}"
  mkdir -p "${retry_dir}"
  run_eval "${retry_cases}" "${retry_dir}" "${PREFIX}_retry_run${run}" 1 \
    0 0 > "${retry_dir}/stdout.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
[[ "${status}" -eq 0 ]] || {
  echo "The one permitted transport-only retry failed; raw retry files are preserved" >&2
  exit 1
}

"${PYTHON_BIN}" \
  training/planner_grpo_seed_v1/scripts/evaluate_qwen35_v12_35b_stability_n10.py \
  --config "${CONFIG}" --raw-root "${OUT_DIR}/raw" \
  --retry-root "${RETRY_ROOT}" --retry-plan "${PLAN}" \
  --analysis-dir "${OUT_DIR}/analysis_retry_once"
echo "n10 transport-retried analysis complete: ${OUT_DIR}/analysis_retry_once/stability_report.json"
