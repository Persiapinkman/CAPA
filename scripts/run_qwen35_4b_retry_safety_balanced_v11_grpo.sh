#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_MODE="${RUN_MODE:-canary}"
SUPPORT_DECISION="${SUPPORT_DECISION:-experiments/studies/planner_retry_safety_balanced_v11_qwen35_4b_v1/support_decision.json}"
STEP_DATA="${STEP_DATA:-training/planner_grpo_seed_v1/step_data/planner_retry_safety_balanced_v11_optimizer_qwen35_4b_nothinking_mixed_steps.jsonl}"
STEP_MANIFEST="${STEP_DATA%.jsonl}.manifest.json"
ADAPTER_PATH="${ADAPTER_PATH:-experiments/runs/20260717T001316Z_qwen35_4b_anti_forgetting_v10_screen10_seed42/checkpoint-10}"
EXPECTED_ADAPTER_SHA="8ef6b4e609497b3837050cce0bf23498e7382f77c28d5b2712ecedf2600b7348"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,4,5,7}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

case "${RUN_MODE}" in
  canary)
    MAX_STEPS="${MAX_STEPS:-2}"
    SAVE_STEPS="${SAVE_STEPS:-2}"
    SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
    OUTPUT_DIR="${OUTPUT_DIR:-experiments/runs/${RUN_TIMESTAMP}_qwen35_4b_safety_balanced_v11_canary2_seed42}"
    ;;
  screen)
    MAX_STEPS="${MAX_STEPS:-8}"
    SAVE_STEPS="${SAVE_STEPS:-1}"
    SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
    OUTPUT_DIR="${OUTPUT_DIR:-experiments/runs/${RUN_TIMESTAMP}_qwen35_4b_safety_balanced_v11_screen8_seed42}"
    ;;
  *)
    echo "RUN_MODE must be canary or screen" >&2
    exit 2
    ;;
esac

if [[ ! -f "${SUPPORT_DECISION}" || ! -f "${STEP_DATA}" || ! -f "${STEP_MANIFEST}" ]]; then
  echo "Missing V11 support decision or frozen optimizer data" >&2
  exit 1
fi
[[ -f "${ADAPTER_PATH}/adapter_model.safetensors" ]] || { echo "Missing V11 initializer" >&2; exit 1; }
[[ "$(sha256sum "${ADAPTER_PATH}/adapter_model.safetensors" | awk '{print $1}')" == "${EXPECTED_ADAPTER_SHA}" ]] || {
  echo "V11 continuation initializer SHA-256 changed" >&2
  exit 1
}

.venv-qwen35-grpo/bin/python - "${SUPPORT_DECISION}" "${STEP_MANIFEST}" <<'PY'
import json
import sys

decision = json.load(open(sys.argv[1], encoding="utf-8"))
manifest = json.load(open(sys.argv[2], encoding="utf-8"))
primary = {"current_success_step2", "fresh_retry_step2", "post_retry_success_step3"}
controls = {
    "post_retry_error_step3", "post_retry_metric_veto_step3",
    "conflicting_state_step2", "nonretryable_step2",
    "budget_exhausted_step2", "missing_required_state_step2",
}
expected = primary | controls
if decision.get("status") != "pass" or decision.get("optimizer_authorized") is not True:
    raise SystemExit("V11 support gate does not authorize optimizer steps")
if set(decision.get("optimizer_scenarios") or []) != expected:
    raise SystemExit("V11 support decision does not authorize all nine scenarios")
if any(not item.get("passed") for item in decision.get("hard_checks") or []):
    raise SystemExit("V11 support decision contains a failed hard check")
if manifest.get("rows") != 576 or set(manifest.get("accepted_scenarios") or []) != expected:
    raise SystemExit("V11 optimizer manifest does not match preregistration")
if manifest.get("scenario_multipliers") != {
    **{scenario: 2 for scenario in primary},
    **{scenario: 1 for scenario in controls},
}:
    raise SystemExit("V11 optimizer primary replay multipliers changed")
if manifest.get("distribution", {}).get("target_actions") != {
    "end": 192, "migrate": 288, "retry": 96
}:
    raise SystemExit("V11 optimizer action balance changed")
PY

export RUN_MODE MAX_STEPS SAVE_STEPS SAVE_TOTAL_LIMIT OUTPUT_DIR STEP_DATA ADAPTER_PATH CUDA_VISIBLE_DEVICES
export EXPECTED_DATASET_ID="planner_retry_safety_balanced_v11"
export EXPECTED_ROWS="576"
export ALLOWED_STEP_INDICES="2,3"
export NUM_PROCESSES="4"
export GENERATION_BATCH_SIZE="4"
export GRAD_ACCUM_STEPS="8"
export GENERATION_TEMPERATURE="0.9"
export GENERATION_TOP_P="0.9"
export LEARNING_RATE="1e-6"
export WARMUP_STEPS="1"
export TASK_REWARD_WEIGHT="0.75"
export FORMAT_REWARD_WEIGHT="0.05"
export NO_FORBIDDEN_ACTION_REWARD_WEIGHT="0.20"
export SEED="42"
export REPORT_TO="auto"
export WANDB_ENTITY="1139090915-tsinghua-university"
export WANDB_PROJECT="capa-planner-post-training"
export WANDB_RUN_GROUP="planner-retry-safety-balanced-v11"
export WANDB_TAGS="stage2,grpo,planner-v11,safety-weighted,action-balanced,continuation,${RUN_MODE},seed42"
export WANDB_RUN_NAME="$(basename "${OUTPUT_DIR}")"
export CONFIRM_TRAIN="YES"

exec scripts/run_qwen35_4b_grpo_v5_train_v1.sh
