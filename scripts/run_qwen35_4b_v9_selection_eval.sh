#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SCREEN_DIR="${SCREEN_DIR:?set SCREEN_DIR to the completed V9 screen directory}"
OUT_DIR="${OUT_DIR:-experiments/runs/20260716_qwen35_4b_v9_selection_dev}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-train/bin/python}"
CASES="training/planner_grpo_seed_v1/cases/planner_retry_safe_end_hard_residual_v9_selection_dev_cases.jsonl"
STUDY_DIR="experiments/studies/planner_retry_safe_end_hard_residual_v9_qwen35_4b_v1"
BASE_MODEL="/raid/zkq/models/Qwen3.5-4B"
SFT_ADAPTER="experiments/runs/20260716_qwen35_4b_planner_v6_sft_seed42_v1/checkpoint-100"
GPUS=(3 4 5 7)
LABELS=(sft checkpoint-10 checkpoint-20 checkpoint-40)
ADAPTERS=("${SFT_ADAPTER}" "${SCREEN_DIR}/checkpoint-10" "${SCREEN_DIR}/checkpoint-20" "${SCREEN_DIR}/checkpoint-40")

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
    --cases "${CASES}" --out-dir "${run_dir}" --report-prefix "v9_${label}" \
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
[[ "${status}" -eq 0 ]] || { echo "Selection rollout failed" >&2; exit 1; }

"${PYTHON_BIN}" training/planner_grpo_seed_v1/scripts/select_planner_retry_safe_end_hard_v9_checkpoint.py \
  --preregistration "${STUDY_DIR}/preregistration.json" \
  --cases "${CASES}" \
  --sft-predictions "${OUT_DIR}/sft/v9_sft_run1_predictions.jsonl" \
  --candidate "checkpoint-10=${OUT_DIR}/checkpoint-10/v9_checkpoint-10_run1_predictions.jsonl" \
  --candidate "checkpoint-20=${OUT_DIR}/checkpoint-20/v9_checkpoint-20_run1_predictions.jsonl" \
  --candidate "checkpoint-40=${OUT_DIR}/checkpoint-40/v9_checkpoint-40_run1_predictions.jsonl" \
  --output "${STUDY_DIR}/selection_decision.json"
