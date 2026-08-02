#!/usr/bin/env bash
# Evaluate Qwen3.5-4B and/or Qwen3.5-35B-A3B on planner routing cases via a
# running vLLM OpenAI endpoint.
#
# The evaluator itself is unmodified:
#   training/planner_grpo_seed_v1/scripts/run_repeated_planner_grpo_eval.py
# It only needs an OpenAI-compatible ``--api-base`` and ``--model`` id.
#
# Usage:
#   API_BASE=http://127.0.0.1:8001/v1 MODEL_ID=Qwen3.5-4B \
#     bash scripts/reproduce/eval_qwen35_h20.sh 4b
#
# Env:
#   CASES               default: planner_grpo_focused_val_v3_cases.jsonl
#   OUT_ROOT            default: /raid/zkq/artifacts/CAPA/outputs/eval_h20
#   RUNS                default: 3
#   MAX_STEPS           default: 3
#   MAX_TOKENS          default: 4096
#   TEMPERATURE         default: 0
#   TOP_P               default: 1
#   SEED                default: 42
#   TIMEOUT_SECONDS     default: 600
#   OPENAI_TIMEOUT_SECONDS default: 600

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

ALIAS="${1:?usage: eval_qwen35_h20.sh <4b|35b|both>}"

CASES_DEFAULT="training/planner_grpo_seed_v1/cases/planner_grpo_focused_val_v3_cases.jsonl"
CASES="${CASES:-${CASES_DEFAULT}}"
[[ -f "${CASES}" ]] || { echo "cases file missing: ${CASES}"; exit 2; }

OUT_ROOT="${OUT_ROOT:-/raid/zkq/artifacts/CAPA/outputs/eval_h20}"
RUNS="${RUNS:-3}"
MAX_STEPS="${MAX_STEPS:-3}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1}"
SEED="${SEED:-42}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-600}"
OPENAI_TIMEOUT_SECONDS="${OPENAI_TIMEOUT_SECONDS:-600}"

INFER_PY="${INFER_PY:-${ROOT_DIR}/.venv-h20-infer/bin/python}"
[[ -x "${INFER_PY}" ]] || { echo "infer venv missing at ${INFER_PY}"; exit 2; }

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy || true

STAMP="$(date +%Y%m%d_%H%M%S)"

eval_arm() {
  local arm="$1" model_id="$2" api_base="$3"
  local out_dir="${OUT_ROOT}/${STAMP}_${arm}"
  mkdir -p "${out_dir}"

  echo "[eval] arm=${arm} api_base=${api_base} model=${model_id} cases=${CASES}"

  # Sanity: does the endpoint publish this model id?
  "${INFER_PY}" - <<PY
import json, urllib.request, sys
url = "${api_base}/models"
with urllib.request.urlopen(url, timeout=10) as r:
    data = json.loads(r.read().decode("utf-8"))
served = {m.get("id") for m in data.get("data", [])}
expected = "${model_id}"
assert expected in served, f"model '{expected}' not in served {sorted(served)}"
print(f"[eval] endpoint ok, served={sorted(served)}")
PY

  "${INFER_PY}" training/planner_grpo_seed_v1/scripts/run_repeated_planner_grpo_eval.py \
    --cases "${CASES}" \
    --out-dir "${out_dir}" \
    --report-prefix "${arm}" \
    --model "${model_id}" \
    --api-base "${api_base}" \
    --runs "${RUNS}" \
    --max-steps "${MAX_STEPS}" \
    --max-tokens "${MAX_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --seed "${SEED}" \
    --do-sample false \
    --timeout-seconds "${TIMEOUT_SECONDS}" \
    --openai-timeout-seconds "${OPENAI_TIMEOUT_SECONDS}"

  echo "[eval] arm ${arm} written to ${out_dir}"
}

case "${ALIAS}" in
  4b)
    API_BASE="${API_BASE:-http://127.0.0.1:8001/v1}"
    MODEL_ID="${MODEL_ID:-Qwen3.5-4B}"
    eval_arm "qwen35_4b_h20" "${MODEL_ID}" "${API_BASE}"
    ;;
  35b)
    API_BASE="${API_BASE:-http://127.0.0.1:8002/v1}"
    MODEL_ID="${MODEL_ID:-Qwen3.5-35B-A3B}"
    eval_arm "qwen35_35b_a3b_h20" "${MODEL_ID}" "${API_BASE}"
    ;;
  both)
    eval_arm "qwen35_4b_h20"        "${MODEL_ID_4B:-Qwen3.5-4B}"        "${API_BASE_4B:-http://127.0.0.1:8001/v1}"
    eval_arm "qwen35_35b_a3b_h20"   "${MODEL_ID_35B:-Qwen3.5-35B-A3B}"  "${API_BASE_35B:-http://127.0.0.1:8002/v1}"
    ;;
  *) echo "unknown alias '${ALIAS}'"; exit 2 ;;
esac

# Cross-arm summary table: parse *_aggregate.json under each arm dir.
"${INFER_PY}" - <<PY
import json, glob, os, sys
root = "${OUT_ROOT}/${STAMP}"
rows = []
for path in sorted(glob.glob(root + "_*/*_aggregate.json")):
    try:
        payload = json.loads(open(path, encoding="utf-8").read())
    except Exception as exc:
        rows.append({"file": path, "error": str(exc)})
        continue
    rows.append({
        "arm": os.path.basename(os.path.dirname(path)),
        "runs": payload.get("runs"),
        "case_macro_mean": payload.get("case_macro_mean"),
        "step_mean_verifier_score": payload.get("step_mean_verifier_score"),
        "case_pass_rate": payload.get("case_pass_rate"),
        "file": path,
    })
summary_path = "${OUT_ROOT}/${STAMP}_summary.json"
with open(summary_path, "w", encoding="utf-8") as fh:
    json.dump({"stamp": "${STAMP}", "arms": rows}, fh, ensure_ascii=False, indent=2)
print(f"[eval] summary -> {summary_path}")
for r in rows:
    print(f"  - {r}")
PY
