#!/usr/bin/env bash
# CAPA reproduction orchestrator.
#
# Phases: preflight | env | models | services | data | unittest | smoke |
#         sft | merge | grpo | eval | compare | gate | registry | sealed | all
#
# Every phase is a thin wrapper around the existing pipelines / scripts.
# The point is to (1) guarantee ordering, (2) enforce the preregistered gate
# before opening the sealed test, and (3) refuse side effects unless the user
# explicitly acknowledges cost.
#
# Environment overrides:
#   DRY_RUN=1                  print commands but do not execute
#   ALLOW_SIDE_EFFECTS=1       permit Flux + full pipeline live calls
#   SEEDS="42 43 44"           GRPO seeds
#   BASE_MODEL_DIR=/raid/zkq/models/Qwen2.5-7B-Instruct
#   ARTIFACT_ROOT=/raid/zkq/artifacts/CAPA
#   STUDY_ID=planner_runtime_routing_grpo_v1
#   TRAIN_CASES=training/planner_grpo_seed_v1/cases/planner_runtime_probe_curriculum_v1_train_cases.jsonl
#   DEV_CASES=training/planner_grpo_seed_v1/cases/planner_runtime_probe_curriculum_v1_dev_cases.jsonl
#   TEST_CASES=training/planner_grpo_seed_v1/cases/planner_runtime_probe_curriculum_v1_test_cases.jsonl

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

# --- defaults -----------------------------------------------------------------
DRY_RUN="${DRY_RUN:-0}"
ALLOW_SIDE_EFFECTS="${ALLOW_SIDE_EFFECTS:-0}"
SEEDS="${SEEDS:-42 43 44}"

BASE_MODEL_DIR="${BASE_MODEL_DIR:-/raid/zkq/models/Qwen2.5-7B-Instruct}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/raid/zkq/artifacts/CAPA}"
OUTPUTS_DIR="${OUTPUTS_DIR:-${ARTIFACT_ROOT}/outputs}"
EVAL_DIR="${EVAL_DIR:-${OUTPUTS_DIR}/eval}"
SFT_OUT_DIR="${SFT_OUT_DIR:-${OUTPUTS_DIR}/planner-sft-qwen25-7b-v3}"
SFT_MERGED_DIR="${SFT_MERGED_DIR:-${OUTPUTS_DIR}/merged-qwen25-7b-sft-v3-chatml}"

STUDY_ID="${STUDY_ID:-planner_runtime_routing_grpo_v1}"
TRAIN_CASES="${TRAIN_CASES:-training/planner_grpo_seed_v1/cases/planner_runtime_probe_curriculum_v1_train_cases.jsonl}"
DEV_CASES="${DEV_CASES:-training/planner_grpo_seed_v1/cases/planner_runtime_probe_curriculum_v1_dev_cases.jsonl}"
TEST_CASES="${TEST_CASES:-training/planner_grpo_seed_v1/cases/planner_runtime_probe_curriculum_v1_test_cases.jsonl}"
STUDY_JSON="${STUDY_JSON:-experiments/studies/${STUDY_ID}/study.json}"

DEMO_VENV_PY="${DEMO_VENV_PY:-${ROOT_DIR}/.venv/bin/python}"
TRAIN_VENV_PY="${TRAIN_VENV_PY:-${ROOT_DIR}/.venv-trl-grpo-cu124/bin/python}"

REPORTS_DIR="${REPORTS_DIR:-${ROOT_DIR}/reports}"
mkdir -p "${REPORTS_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-${ARTIFACT_ROOT}/logs/repro/${TS}}"

# --- helpers ------------------------------------------------------------------
log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die()  { log "FATAL: $*"; exit 2; }

run() {
  log "$ $*"
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  mkdir -p "${LOG_DIR}"
  ( set -o pipefail; "$@" ) 2>&1 | tee -a "${LOG_DIR}/orchestrator.log"
}

require_train_venv() {
  [[ -x "${TRAIN_VENV_PY}" ]] || die "train venv missing: ${TRAIN_VENV_PY} (run phase 'env' first)"
}

require_demo_venv() {
  [[ -x "${DEMO_VENV_PY}" ]] || die "demo venv missing: ${DEMO_VENV_PY} (run phase 'env' first)"
}

require_side_effects() {
  [[ "${ALLOW_SIDE_EFFECTS}" == "1" ]] \
    || die "phase '$1' produces external side effects (cost); rerun with ALLOW_SIDE_EFFECTS=1"
}

# --- phases -------------------------------------------------------------------

phase_preflight() {
  log "== P0 preflight =="
  local py="${TRAIN_VENV_PY}"
  [[ -x "${py}" ]] || py="$(command -v python3)"
  run "${py}" scripts/reproduce_preflight.py --out "${REPORTS_DIR}/preflight_${TS}.json"
}

phase_env() {
  log "== P1 build venvs =="
  local py310
  py310="$(command -v python3.10 || command -v python3)"
  [[ -n "${py310}" ]] || die "python3.10 not found"

  if [[ ! -x "${DEMO_VENV_PY}" ]]; then
    run "${py310}" -m venv "${ROOT_DIR}/.venv"
    run "${ROOT_DIR}/.venv/bin/pip" install -U pip
    run "${ROOT_DIR}/.venv/bin/pip" install -e '.[demo]'
  else
    log "demo venv already exists at ${DEMO_VENV_PY}"
  fi

  if [[ ! -x "${TRAIN_VENV_PY}" ]]; then
    run "${py310}" -m venv "${ROOT_DIR}/.venv-trl-grpo-cu124"
    run "${ROOT_DIR}/.venv-trl-grpo-cu124/bin/pip" install -U pip
    run "${ROOT_DIR}/.venv-trl-grpo-cu124/bin/pip" install \
      -r configs/environments/trl-cu124.lock.txt
    run "${ROOT_DIR}/.venv-trl-grpo-cu124/bin/pip" install -e '.[train-cu124]'
  else
    log "train venv already exists at ${TRAIN_VENV_PY}"
  fi
}

phase_models() {
  log "== P2 fetch base model =="
  if [[ -d "${BASE_MODEL_DIR}" ]] && [[ -n "$(ls -A "${BASE_MODEL_DIR}" 2>/dev/null)" ]]; then
    log "model already present: ${BASE_MODEL_DIR}"
    return 0
  fi
  MODEL_ID="Qwen/Qwen2.5-7B-Instruct" LOCAL_DIR="${BASE_MODEL_DIR}" \
    PYTHON_BIN="${TRAIN_VENV_PY}" \
    run bash scripts/download_qwen25_7b_instruct.sh
}

phase_services() {
  log "== P3 external service reachability =="
  # Just probe; RAG tunnel & demo server must be started by operator.
  require_train_venv
  run "${TRAIN_VENV_PY}" scripts/reproduce_preflight.py \
    --check services --out "${REPORTS_DIR}/preflight_services_${TS}.json"
  log "note: bring up RAG tunnel manually if needed:"
  log "      bash pipelines/demo/open_rag_tunnel.sh"
  log "      source init_env.sh && ${DEMO_VENV_PY} demo/demo_server.py --port 18080"
}

phase_data() {
  log "== P4 register + validate datasets =="
  require_train_venv
  run "${TRAIN_VENV_PY}" pipelines/data/register_planner_dataset.py
  run "${TRAIN_VENV_PY}" pipelines/data/register_runtime_routing_dataset.py
  run "${TRAIN_VENV_PY}" pipelines/data/register_stateful_retrieval_dataset.py
  run "${TRAIN_VENV_PY}" pipelines/experiments/registry_cli.py validate
}

phase_unittest() {
  log "== P5 unit tests =="
  require_train_venv
  PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}" \
    run "${TRAIN_VENV_PY}" -m unittest discover -s tests -v
}

phase_smoke() {
  log "== P6 demo smoke (no side effects unless explicitly allowed) =="
  require_demo_venv
  [[ -f "${ROOT_DIR}/init_env.sh" ]] || die "init_env.sh missing"
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/init_env.sh"

  local args=(--include-migration)
  if [[ "${ALLOW_SIDE_EFFECTS}" == "1" ]]; then
    args+=(--include-flux --include-pipeline --allow-side-effects)
  fi
  run "${DEMO_VENV_PY}" pipelines/demo/run_full_demo_smoke.py "${args[@]}"
}

phase_sft() {
  log "== P7 SFT (single run, 8-GPU) =="
  require_train_venv
  [[ -d "${BASE_MODEL_DIR}" ]] || die "base model missing at ${BASE_MODEL_DIR}"
  MODEL_PATH="${BASE_MODEL_DIR}" \
    OUTPUT_DIR="${SFT_OUT_DIR}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
    NUM_PROCESSES="${NUM_PROCESSES:-8}" \
    PYTHON_BIN="${TRAIN_VENV_PY}" \
    ACCELERATE_BIN="${ROOT_DIR}/.venv-trl-grpo-cu124/bin/accelerate" \
    run bash scripts/run_qwen25_7b_trl_sft_lora.sh
}

phase_merge() {
  log "== P8 merge SFT LoRA -> ${SFT_MERGED_DIR} =="
  require_train_venv
  [[ -d "${SFT_OUT_DIR}" ]] || die "SFT output missing at ${SFT_OUT_DIR}"
  run "${TRAIN_VENV_PY}" scripts/merge_lora_adapter.py \
    --base-model "${BASE_MODEL_DIR}" \
    --adapter "${SFT_OUT_DIR}" \
    --output-dir "${SFT_MERGED_DIR}"
}

phase_grpo() {
  log "== P9 GRPO x ${SEEDS} =="
  require_train_venv
  [[ -d "${SFT_MERGED_DIR}" ]] || die "merged SFT missing at ${SFT_MERGED_DIR}"
  [[ -f "${TRAIN_CASES}" ]]    || die "train cases missing: ${TRAIN_CASES}"

  for seed in ${SEEDS}; do
    local out="${OUTPUTS_DIR}/runtime_probe_grpo${seed}"
    log "-- GRPO seed=${seed} out=${out}"
    MODEL_PATH="${SFT_MERGED_DIR}" \
      CASES="${TRAIN_CASES}" \
      PROMPT_FORMAT="qwen_chatml" \
      OUTPUT_DIR="${out}" \
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
      NUM_PROCESSES="${NUM_PROCESSES:-8}" \
      GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-8}" \
      NUM_GENERATIONS="${NUM_GENERATIONS:-8}" \
      LEARNING_RATE="${LEARNING_RATE:-2e-6}" \
      TEMPERATURE="${TEMPERATURE:-0.7}" \
      TOP_P="${TOP_P:-0.9}" \
      MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-128}" \
      GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}" \
      MAX_STEPS="${MAX_STEPS:-80}" \
      SAVE_STEPS="${SAVE_STEPS:-40}" \
      TASK_REWARD_WEIGHT="${TASK_REWARD_WEIGHT:-0.85}" \
      FORMAT_REWARD_WEIGHT="${FORMAT_REWARD_WEIGHT:-0.15}" \
      SEED="${seed}" \
      PYTHON_BIN="${TRAIN_VENV_PY}" \
      ACCELERATE_BIN="${ROOT_DIR}/.venv-trl-grpo-cu124/bin/accelerate" \
      run bash scripts/run_qwen25_7b_trl_grpo_lora.sh
  done
}

_eval_one() {
  local tag="$1" model_path="$2" cases="$3"
  local out="${EVAL_DIR}/${tag}"
  local run_id="${TS}_${STUDY_ID}_${tag}"
  log "-- eval tag=${tag} model=${model_path} cases=${cases}"
  run "${TRAIN_VENV_PY}" pipelines/eval/run_generation_eval.py \
    --run-id "${run_id}" \
    --study-id "${STUDY_ID}" \
    --model-path "${model_path}" \
    --cases "${cases}" \
    --temperature 0 --top-p 1 --do-sample false \
    --repeats 3 --seed 42 \
    --out-dir "${out}"
}

phase_eval() {
  log "== P10 development 3x deterministic eval =="
  require_train_venv
  [[ -f "${DEV_CASES}" ]] || die "dev cases missing: ${DEV_CASES}"
  [[ -d "${SFT_MERGED_DIR}" ]] || die "baseline (merged SFT) missing"

  _eval_one "sft_v3_dev3x" "${SFT_MERGED_DIR}" "${DEV_CASES}"
  for seed in ${SEEDS}; do
    local model="${OUTPUTS_DIR}/runtime_probe_grpo${seed}"
    [[ -d "${model}" ]] || { log "skip seed ${seed}: ${model} missing"; continue; }
    _eval_one "runtime_probe_grpo${seed}_dev3x" "${model}" "${DEV_CASES}"
  done
}

phase_compare() {
  log "== P11 case-macro paired compare =="
  require_train_venv
  local baseline="${EVAL_DIR}/sft_v3_dev3x"
  local candidates=()
  for seed in ${SEEDS}; do
    local c="${EVAL_DIR}/runtime_probe_grpo${seed}_dev3x"
    [[ -d "${c}" ]] && candidates+=("${c}")
  done
  [[ ${#candidates[@]} -gt 0 ]] || die "no candidate eval outputs found; run 'eval' first"

  run "${TRAIN_VENV_PY}" pipelines/eval/compare_generation_runs.py \
    --baseline "${baseline}" \
    --candidates "${candidates[@]}" \
    --out "${REPORTS_DIR}/compare_runtime_probe_${TS}.json"
}

phase_gate() {
  log "== P12 preregistered multi-seed development gate =="
  require_train_venv
  [[ -f "${STUDY_JSON}" ]] || die "study spec missing: ${STUDY_JSON}"
  local baseline="${EVAL_DIR}/sft_v3_dev3x"
  local candidates=()
  for seed in ${SEEDS}; do
    local c="${EVAL_DIR}/runtime_probe_grpo${seed}_dev3x"
    [[ -d "${c}" ]] && candidates+=("${c}")
  done
  [[ ${#candidates[@]} -gt 0 ]] || die "no candidate eval outputs found"

  local gate_out="${REPORTS_DIR}/gate_runtime_probe_${TS}.json"
  run "${TRAIN_VENV_PY}" pipelines/eval/check_runtime_routing_multiseed_gate.py \
    --study "${STUDY_JSON}" \
    --baseline "${baseline}" \
    --candidates "${candidates[@]}" \
    --out "${gate_out}"
  log "gate report written to ${gate_out}"
  log "if gate PASSED, phase 'sealed' becomes eligible (still one-shot)"
}

phase_registry() {
  log "== P13 append registry + render CURRENT =="
  require_train_venv
  shopt -s nullglob
  local added=0
  for rec in "${EVAL_DIR}"/*/run_record.json; do
    run "${TRAIN_VENV_PY}" pipelines/experiments/registry_cli.py add "${rec}"
    added=$((added + 1))
  done
  shopt -u nullglob
  log "added ${added} run record(s)"
  run "${TRAIN_VENV_PY}" pipelines/experiments/registry_cli.py render
}

phase_sealed() {
  log "== P14 sealed test evaluation (one-shot) =="
  require_train_venv
  [[ -f "${TEST_CASES}" ]] || die "sealed test cases missing: ${TEST_CASES}"

  local gate_file
  gate_file="$(ls -1t "${REPORTS_DIR}"/gate_runtime_probe_*.json 2>/dev/null | head -n1 || true)"
  [[ -n "${gate_file}" ]] || die "no gate report found; run phase 'gate' first"

  local gate_pass
  gate_pass="$("${TRAIN_VENV_PY}" -c "import json,sys;d=json.load(open('${gate_file}'));print(d.get('passed', False))")"
  if [[ "${gate_pass}" != "True" ]]; then
    die "development gate did NOT pass (${gate_file}); sealed test remains closed"
  fi
  log "development gate passed; proceeding with sealed test on preselected model"

  local model="${OUTPUTS_DIR}/runtime_probe_grpo42"
  [[ -d "${model}" ]] || die "preselected model missing: ${model}"
  _eval_one "runtime_probe_grpo42_sealed" "${model}" "${TEST_CASES}"
}

phase_all() {
  phase_preflight
  phase_env
  phase_models
  phase_services
  phase_data
  phase_unittest
  phase_smoke
  phase_sft
  phase_merge
  phase_grpo
  phase_eval
  phase_compare
  phase_gate
  phase_registry
  log "== all default phases finished; sealed test is manual =="
}

# ---------------------------------------------------------------------------
# H20 (4x Hopper) phases: separate venvs, vLLM serving, Qwen3.5 evaluation and
# training smoke. These do NOT touch the V100 pipeline above.
# ---------------------------------------------------------------------------

H20_MODELS_ROOT="${H20_MODELS_ROOT:-/raid/zkq/models}"
QWEN35_4B_REPO="${QWEN35_4B_REPO:-Qwen/Qwen3-4B}"
QWEN35_35B_REPO="${QWEN35_35B_REPO:-Qwen/Qwen3-30B-A3B}"
INFER_VENV_PY="${INFER_VENV_PY:-${ROOT_DIR}/.venv-h20-infer/bin/python}"
QWEN35_TRAIN_VENV_PY="${QWEN35_TRAIN_VENV_PY:-${ROOT_DIR}/.venv-qwen35-grpo/bin/python}"

phase_h20_env() {
  log "== H0 build .venv-h20-infer and .venv-qwen35-grpo =="
  run bash scripts/reproduce/setup_h20_env.sh
}

phase_h20_models() {
  log "== H1 download Qwen3.5-4B and Qwen3.5-35B-A3B weights =="
  MODELS_ROOT="${H20_MODELS_ROOT}" \
    QWEN35_4B_REPO="${QWEN35_4B_REPO}" \
    QWEN35_35B_REPO="${QWEN35_35B_REPO}" \
    VENV_BIN="${INFER_VENV_PY}" \
    run bash scripts/reproduce/download_qwen35_models.sh
}

phase_h20_serve_4b() {
  log "== H2 launch vLLM serve for Qwen3.5-4B (background) =="
  local pid_file="${ARTIFACT_ROOT}/logs/vllm/vllm_4b.pid"
  if [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
    log "vLLM 4B already running pid=$(cat "${pid_file}")"
    return 0
  fi
  MODELS_ROOT="${H20_MODELS_ROOT}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    PORT="${VLLM_4B_PORT:-8001}" \
    nohup bash scripts/reproduce/serve_qwen35_vllm.sh 4b \
      >>"${ARTIFACT_ROOT}/logs/vllm/vllm_4b.out" 2>&1 &
  disown || true
  log "spawned vLLM 4B; probe endpoint..."
  "${INFER_VENV_PY}" -c "
import sys
sys.path.insert(0, 'src')
from capa.inference.h20_backend import ServeSpec, probe_endpoint
from pathlib import Path
spec = ServeSpec(model_alias='Qwen3.5-4B',
                 model_path=Path('${H20_MODELS_ROOT}/Qwen3.5-4B'),
                 served_model_name='Qwen3.5-4B',
                 port=int('${VLLM_4B_PORT:-8001}'))
print(probe_endpoint(spec, timeout_seconds=900))
"
}

phase_h20_serve_35b() {
  log "== H3 launch vLLM serve for Qwen3.5-35B-A3B (TP=4, background) =="
  local pid_file="${ARTIFACT_ROOT}/logs/vllm/vllm_35b.pid"
  if [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
    log "vLLM 35B already running pid=$(cat "${pid_file}")"
    return 0
  fi
  MODELS_ROOT="${H20_MODELS_ROOT}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
    PORT="${VLLM_35B_PORT:-8002}" \
    TENSOR_PARALLEL_SIZE=4 \
    nohup bash scripts/reproduce/serve_qwen35_vllm.sh 35b \
      >>"${ARTIFACT_ROOT}/logs/vllm/vllm_35b.out" 2>&1 &
  disown || true
  log "spawned vLLM 35B; probe endpoint..."
  "${INFER_VENV_PY}" -c "
import sys
sys.path.insert(0, 'src')
from capa.inference.h20_backend import ServeSpec, probe_endpoint
from pathlib import Path
spec = ServeSpec(model_alias='Qwen3.5-35B-A3B',
                 model_path=Path('${H20_MODELS_ROOT}/Qwen3.5-35B-A3B'),
                 served_model_name='Qwen3.5-35B-A3B',
                 port=int('${VLLM_35B_PORT:-8002}'), tensor_parallel_size=4)
print(probe_endpoint(spec, timeout_seconds=1800))
"
}

phase_h20_eval_4b() {
  log "== H4 evaluate Qwen3.5-4B on planner routing =="
  API_BASE="http://127.0.0.1:${VLLM_4B_PORT:-8001}/v1" \
    MODEL_ID="Qwen3.5-4B" \
    run bash scripts/reproduce/eval_qwen35_h20.sh 4b
}

phase_h20_eval_35b() {
  log "== H5 evaluate Qwen3.5-35B-A3B on planner routing =="
  API_BASE="http://127.0.0.1:${VLLM_35B_PORT:-8002}/v1" \
    MODEL_ID="Qwen3.5-35B-A3B" \
    run bash scripts/reproduce/eval_qwen35_h20.sh 35b
}

phase_h20_smoke_sft() {
  log "== H6 Qwen3.5-4B SFT smoke (dry-run + 3 optimizer steps) =="
  CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
    NUM_PROCESSES=4 \
    run bash scripts/reproduce/train_qwen35_4b_h20_smoke.sh sft-dry
  CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
    NUM_PROCESSES=4 \
    run bash scripts/reproduce/train_qwen35_4b_h20_smoke.sh sft-smoke
}

phase_h20_smoke_grpo() {
  log "== H7 Qwen3.5-4B GRPO smoke (dry-run + g4 single optimizer step) =="
  CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
    NUM_PROCESSES=4 \
    run bash scripts/reproduce/train_qwen35_4b_h20_smoke.sh grpo-dry
  CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
    NUM_PROCESSES=4 \
    run bash scripts/reproduce/train_qwen35_4b_h20_smoke.sh grpo-smoke
}

phase_h20_stop() {
  log "== H* stop vLLM background servers =="
  for pid_file in "${ARTIFACT_ROOT}/logs/vllm/"vllm_*.pid; do
    [[ -f "${pid_file}" ]] || continue
    local pid
    pid="$(cat "${pid_file}")"
    if kill -0 "${pid}" 2>/dev/null; then
      log "stopping ${pid_file} pid=${pid}"
      kill "${pid}" || true
    fi
    rm -f "${pid_file}"
  done
}

phase_h20_all() {
  phase_h20_env
  phase_h20_models
  phase_h20_serve_4b
  phase_h20_eval_4b
  phase_h20_smoke_sft
  phase_h20_smoke_grpo
  phase_h20_stop
  phase_h20_serve_35b
  phase_h20_eval_35b
  phase_h20_stop
  log "== H20 all done: 4B eval + 4B training smoke + 35B eval =="
}

# --- dispatch -----------------------------------------------------------------

usage() {
  cat <<'EOF'
Usage: scripts/reproduce/reproduce_all.sh <phase> [<phase> ...]

Phases:
  preflight  hardware/repo/env/data/models/services checks
  env        build .venv (demo) and .venv-trl-grpo-cu124 (train)
  models     download Qwen2.5-7B-Instruct into /raid/zkq/models
  services   probe SOCKS / RAG tunnels / demo server (no start)
  data       run register_*_dataset.py + registry validate
  unittest   unittest discover -s tests
  smoke      demo end-to-end smoke (needs services + init_env.sh)
  sft        LoRA SFT on Qwen2.5-7B-Instruct
  merge      merge SFT adapter -> merged-qwen25-7b-sft-v3-chatml
  grpo       GRPO for each seed in $SEEDS (default 42 43 44)
  eval       3x deterministic dev eval for baseline + all seeds
  compare    case-macro paired CI report
  gate       preregistered multi-seed development gate
  registry   append run records + render reports/CURRENT.md
  sealed     open sealed test ONCE if gate passed (one-shot)
  all        preflight..registry (skips sealed)

H20 phases (4x Hopper, bf16, vLLM):
  h20-env         build .venv-h20-infer (vLLM) and .venv-qwen35-grpo (trainer)
  h20-models      download Qwen3.5-4B and Qwen3.5-35B-A3B into $MODELS_ROOT
                  (defaults: Qwen/Qwen3-4B, Qwen/Qwen3-30B-A3B; override via
                   QWEN35_4B_REPO / QWEN35_35B_REPO)
  h20-serve-4b    vLLM serve Qwen3.5-4B on 1 GPU, port 8001
  h20-serve-35b   vLLM serve Qwen3.5-35B-A3B on 4 GPUs (TP=4), port 8002
  h20-eval-4b     planner routing 3x eval on the 4B endpoint
  h20-eval-35b    planner routing 3x eval on the 35B endpoint
  h20-smoke-sft   Qwen3.5-4B SFT dry-run + 3-step canary
  h20-smoke-grpo  Qwen3.5-4B GRPO dry-run + g4 single optimizer step
  h20-stop        stop any background vLLM server started here
  h20-all         h20-env + h20-models + serve/eval/smoke, alternating GPUs

Env flags:
  DRY_RUN=1              print commands but do not execute
  ALLOW_SIDE_EFFECTS=1   allow Flux + full pipeline live calls
  SEEDS="42 43 44"       GRPO seeds
EOF
}

main() {
  [[ $# -ge 1 ]] || { usage; exit 1; }
  for phase in "$@"; do
    case "${phase}" in
      preflight) phase_preflight ;;
      env)       phase_env ;;
      models)    phase_models ;;
      services)  phase_services ;;
      data)      phase_data ;;
      unittest)  phase_unittest ;;
      smoke)     phase_smoke ;;
      sft)       phase_sft ;;
      merge)     phase_merge ;;
      grpo)      phase_grpo ;;
      eval)      phase_eval ;;
      compare)   phase_compare ;;
      gate)      phase_gate ;;
      registry)  phase_registry ;;
      sealed)    phase_sealed ;;
      all)       phase_all ;;
      # ---- H20 phases ----
      h20-env)         phase_h20_env ;;
      h20-models)      phase_h20_models ;;
      h20-serve-4b)    phase_h20_serve_4b ;;
      h20-serve-35b)   phase_h20_serve_35b ;;
      h20-eval-4b)     phase_h20_eval_4b ;;
      h20-eval-35b)    phase_h20_eval_35b ;;
      h20-smoke-sft)   phase_h20_smoke_sft ;;
      h20-smoke-grpo)  phase_h20_smoke_grpo ;;
      h20-stop)        phase_h20_stop ;;
      h20-all)         phase_h20_all ;;
      -h|--help) usage; exit 0 ;;
      *) die "unknown phase: ${phase}" ;;
    esac
  done
  log "done."
}

main "$@"
