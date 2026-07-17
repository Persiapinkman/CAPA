#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STUDY_DIR="experiments/studies/planner_retry_anti_forgetting_v10_qwen35_4b_v1"
OPENING="${OPENING:-${STUDY_DIR}/sealed_test_opening.json}"
OUT_DIR="${OUT_DIR:-experiments/runs/20260717_qwen35_v10_sealed}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-qwen35-grpo/bin/python}"
REMOTE_PYTHON_BIN="${REMOTE_PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
BASE_MODEL="/raid/zkq/models/Qwen3.5-4B"
SFT_ADAPTER="experiments/runs/20260716_qwen35_4b_planner_v6_sft_seed42_v1/checkpoint-100"
EXPECTED_ROWS=432
NUM_LOCAL_SHARDS=3
LOCAL_LIMIT=$((EXPECTED_ROWS / NUM_LOCAL_SHARDS))
SFT_GPUS=(2 3 4)
GRPO_GPUS=(5 6 7)

[[ -f "${OPENING}" ]] || { echo "Missing sealed-test opening: ${OPENING}" >&2; exit 1; }
[[ ! -e "${STUDY_DIR}/sealed_objective.json" ]] || { echo "Sealed objective already exists" >&2; exit 1; }
[[ ! -e "${OUT_DIR}" ]] || { echo "Refusing to overwrite sealed rollout directory: ${OUT_DIR}" >&2; exit 1; }
[[ "$(jq -r '.status' "${OPENING}")" == "sealed_test_materialized_once" ]] || {
  echo "V10 sealed test has not been validly opened" >&2
  exit 1
}
CASES="$(jq -r '.combined_test_cases' "${OPENING}")"
GRPO_ADAPTER="$(jq -r '.selected.adapter' "${OPENING}")"
EXPECTED_ADAPTER_SHA="$(jq -r '.selected.adapter_sha256' "${OPENING}")"
[[ -f "${CASES}" ]] || { echo "Missing sealed cases: ${CASES}" >&2; exit 1; }
[[ "$(wc -l < "${CASES}")" -eq "${EXPECTED_ROWS}" ]] || { echo "Sealed case count changed" >&2; exit 1; }
[[ -f "${SFT_ADAPTER}/adapter_model.safetensors" ]] || { echo "Missing SFT adapter" >&2; exit 1; }
[[ -f "${GRPO_ADAPTER}/adapter_model.safetensors" ]] || { echo "Missing selected GRPO adapter" >&2; exit 1; }
[[ "$(sha256sum "${GRPO_ADAPTER}/adapter_model.safetensors" | awk '{print $1}')" == "${EXPECTED_ADAPTER_SHA}" ]] || {
  echo "Selected GRPO adapter SHA-256 changed" >&2
  exit 1
}

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export CAPA_OMIT_MODEL_IMAGE_PAYLOAD=1
mkdir -p "${OUT_DIR}"

pids=()
for shard in 0 1 2; do
  offset=$((shard * LOCAL_LIMIT))
  for family in sft grpo; do
    if [[ "${family}" == "sft" ]]; then
      gpu="${SFT_GPUS[$shard]}"
      adapter="${SFT_ADAPTER}"
    else
      gpu="${GRPO_GPUS[$shard]}"
      adapter="${GRPO_ADAPTER}"
    fi
    shard_dir="${OUT_DIR}/${family}/shard${shard}"
    mkdir -p "${shard_dir}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" \
      training/planner_grpo_seed_v1/scripts/run_repeated_planner_grpo_eval.py \
      --cases "${CASES}" --out-dir "${shard_dir}" \
      --report-prefix "v10_sealed_${family}_shard${shard}" \
      --model "qwen35-4b-${family}" --api-base local --runs 1 \
      --offset "${offset}" --limit "${LOCAL_LIMIT}" \
      --max-steps 3 --max-tokens 320 --temperature 0 --top-p 1 --seed 42 \
      --do-sample false --timeout-seconds 180 --openai-timeout-seconds 180 \
      --local-model-path "${BASE_MODEL}" --local-adapter-path "${adapter}" \
      > "${shard_dir}/stdout.log" 2>&1 &
    pids+=("$!")
  done
done

CASES="${CASES}" \
OUT_DIR="${OUT_DIR}/larger" \
REPORT_PREFIX="v10_sealed_qwen35_35b_reference_t0" \
EXPECTED_ROWS="${EXPECTED_ROWS}" \
PYTHON_BIN="${REMOTE_PYTHON_BIN}" \
scripts/run_qwen35_larger_reference_eval.sh \
  > "${OUT_DIR}/larger_stdout.log" 2>&1 &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
[[ "${status}" -eq 0 ]] || { echo "Sealed rollout failed" >&2; exit 1; }

for family in sft grpo; do
  predictions=()
  for shard in 0 1 2; do
    predictions+=("${OUT_DIR}/${family}/shard${shard}/v10_sealed_${family}_shard${shard}_run1_predictions.jsonl")
  done
  "${PYTHON_BIN}" training/planner_grpo_seed_v1/scripts/combine_planner_rollout_prediction_shards.py \
    --cases "${CASES}" --predictions "${predictions[@]}" \
    --output "${OUT_DIR}/${family}/predictions.jsonl"
  "${PYTHON_BIN}" training/planner_grpo_seed_v1/scripts/reward_planner_grpo.py \
    --cases "${CASES}" --predictions "${OUT_DIR}/${family}/predictions.jsonl" \
    --out "${OUT_DIR}/${family}/reward.json"
done

"${PYTHON_BIN}" training/planner_grpo_seed_v1/scripts/finalize_planner_grpo_objective.py \
  --contract "${STUDY_DIR}/comparison_contract.json" \
  --cases "${CASES}" \
  --sft-predictions "${OUT_DIR}/sft/predictions.jsonl" \
  --grpo-predictions "${OUT_DIR}/grpo/predictions.jsonl" \
  --larger-predictions "${OUT_DIR}/larger/predictions.jsonl" \
  --output "${STUDY_DIR}/sealed_objective.json"
