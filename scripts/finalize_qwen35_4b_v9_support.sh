#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-train/bin/python}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
RUN_DIR="${RUN_DIR:-experiments/runs/20260716_qwen35_4b_v9_hard_residual_support4x_sft100}"
STUDY_DIR="experiments/studies/planner_retry_safe_end_hard_residual_v9_qwen35_4b_v1"
STEP_DATA="training/planner_grpo_seed_v1/step_data/planner_retry_safe_end_hard_residual_v9_support_dev_qwen35_4b_nothinking_mixed_steps.jsonl"

samples=()
for shard in 0 1 2 3; do
  path="${RUN_DIR}/shard${shard}/samples.jsonl"
  [[ -f "${path}" ]] || { echo "Missing ${path}" >&2; exit 1; }
  samples+=("${path}")
done

"${PYTHON_BIN}" training/planner_grpo_seed_v1/scripts/combine_qwen35_planner_v6_step_support.py \
  --samples "${samples[@]}" \
  --samples-out "${RUN_DIR}/samples.jsonl" \
  --summary-out "${RUN_DIR}/summary.json"

"${PYTHON_BIN}" training/planner_grpo_seed_v1/scripts/gate_planner_retry_safe_end_hard_residual_v9_support.py \
  --preregistration "${STUDY_DIR}/preregistration.json" \
  --step-data "${STEP_DATA}" \
  --samples "${RUN_DIR}/samples.jsonl" \
  --output "${STUDY_DIR}/support_decision.json" \
  --accepted-scenarios-out data/datasets/planner_retry_safe_end_hard_residual_v9/accepted_optimizer_scenarios.txt
