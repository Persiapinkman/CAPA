#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SCREEN_DIR="${SCREEN_DIR:?set SCREEN_DIR to the completed V11 screen directory}"
OUT_DIR="${OUT_DIR:-experiments/runs/20260717_qwen35_4b_v11_selection_dev}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-qwen35-grpo/bin/python}"
CASES="training/planner_grpo_seed_v1/cases/planner_retry_safety_balanced_v11_selection_dev_cases.jsonl"
STUDY_DIR="experiments/studies/planner_retry_safety_balanced_v11_qwen35_4b_v1"
SCREEN_AUDIT="${SCREEN_AUDIT:-${STUDY_DIR}/screen_health.json}"
BASE_MODEL="/raid/zkq/models/Qwen3.5-4B"
SFT_ADAPTER="experiments/runs/20260716_qwen35_4b_planner_v6_sft_seed42_v1/checkpoint-100"
GPUS=(3 4 5 7)
LABELS=(sft checkpoint-2 checkpoint-5 checkpoint-8)
ADAPTERS=("${SFT_ADAPTER}" "${SCREEN_DIR}/checkpoint-2" "${SCREEN_DIR}/checkpoint-5" "${SCREEN_DIR}/checkpoint-8")

[[ -f "${SCREEN_AUDIT}" ]] || { echo "Missing screen audit: ${SCREEN_AUDIT}" >&2; exit 1; }
[[ "$(jq -r '.status' "${SCREEN_AUDIT}")" == "pass" ]] || { echo "V11 screen audit did not pass" >&2; exit 1; }
[[ ! -e "${OUT_DIR}" ]] || { echo "Refusing to overwrite ${OUT_DIR}" >&2; exit 1; }

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export CAPA_OMIT_MODEL_IMAGE_PAYLOAD=1

pids=()
for index in 0 1 2 3; do
  label="${LABELS[$index]}"
  adapter="${ADAPTERS[$index]}"
  [[ -f "${adapter}/adapter_model.safetensors" ]] || { echo "Missing ${adapter}" >&2; exit 1; }
  run_dir="${OUT_DIR}/${label}"
  mkdir -p "${run_dir}"
  CUDA_VISIBLE_DEVICES="${GPUS[$index]}" "${PYTHON_BIN}" \
    training/planner_grpo_seed_v1/scripts/run_repeated_planner_grpo_eval.py \
    --cases "${CASES}" --out-dir "${run_dir}" --report-prefix "v11_${label}" \
    --model "qwen35-4b-${label}" --api-base local --runs 1 \
    --max-steps 3 --max-tokens 320 --temperature 0 --top-p 1 --seed 42 \
    --do-sample false --timeout-seconds 180 --openai-timeout-seconds 180 \
    --local-model-path "${BASE_MODEL}" --local-adapter-path "${adapter}" \
    > "${run_dir}/stdout.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
[[ "${status}" -eq 0 ]] || { echo "V11 selection rollout failed" >&2; exit 1; }

"${PYTHON_BIN}" training/planner_grpo_seed_v1/scripts/select_planner_grpo_checkpoint.py \
  --preregistration "${STUDY_DIR}/preregistration.json" \
  --cases "${CASES}" \
  --sft-predictions "${OUT_DIR}/sft/v11_sft_run1_predictions.jsonl" \
  --candidate "checkpoint-2=${OUT_DIR}/checkpoint-2/v11_checkpoint-2_run1_predictions.jsonl" \
  --candidate "checkpoint-5=${OUT_DIR}/checkpoint-5/v11_checkpoint-5_run1_predictions.jsonl" \
  --candidate "checkpoint-8=${OUT_DIR}/checkpoint-8/v11_checkpoint-8_run1_predictions.jsonl" \
  --output "${STUDY_DIR}/selection_decision.json"
