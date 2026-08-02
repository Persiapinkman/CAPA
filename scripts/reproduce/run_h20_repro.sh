#!/usr/bin/env bash
# H20 CAPA reproduction driver (three scenarios + SFT + GRPO).
#
# Scenarios reused across every model arm:
#   - single-step routing:      training/planner_dpo_train_seed_v1/eval/planner_routing_eval_90cases.json
#   - multi-step routing:       training/planner_grpo_seed_v1/cases/planner_grpo_focused_val_v3_cases.jsonl
#   - soft-boundary (V6 dev):   training/planner_grpo_seed_v1/cases/planner_retry_migrate_v6_grpo_dev_cases.jsonl
#     (test split is sealed and only opened by phase 'sealed')
#
# Model arms:
#   base-4b, base-35b, sft, grpo42, grpo43, grpo44
#
# All heavy artifacts live under ${ART_ROOT} (default follows this repo's
# data location, NOT /raid/zkq — that legacy prefix is not available on this
# host). Every phase is idempotent: re-running only fills what is missing.
#
# Layout under ART_ROOT/repro_h20/:
#   preflight/preflight_<STAMP>.json
#   eval/<STAMP>_<arm>/<scenario>/               # 3x eval + aggregate + summary
#   sft/<RUN_ID>/checkpoint-*
#   sft/<RUN_ID>_merged                          # merged model dir for vLLM serve
#   grpo/<RUN_ID>_seed<SEED>/checkpoint-*
#   grpo/<RUN_ID>_seed<SEED>_merged
#   compare/compare_<STAMP>.json
#   gate/gate_<STAMP>.json
#
# Usage:
#   scripts/reproduce/run_h20_repro.sh <phase> [<phase> ...]
#   scripts/reproduce/run_h20_repro.sh all-base       # everything base needs
#   scripts/reproduce/run_h20_repro.sh all-train      # SFT + GRPOx3 + eval + gate
#
# Env:
#   ART_ROOT               default: <repo>/../capa_h20/artifacts/CAPA
#   H20_MODELS_ROOT        default: <repo>/../capa_h20/models
#   INFER_VENV             default: <repo>/.venv-h20-infer
#   TRAIN_VENV             default: <repo>/.venv-qwen35-grpo
#   DRY_RUN=1              print commands but do not execute
#   FORCE=1                overwrite existing phase output directories
#   VLLM_4B_PORT           default 8001
#   VLLM_35B_PORT          default 8002
#   SEEDS                  default "42 43 44"

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

# ---------- defaults ---------------------------------------------------------

# Locate capa_h20 sibling tree. Standard layout is
# <workspace>/projects/CAPA + <workspace>/capa_h20; on some hosts the repo sits
# directly under <workspace>. Try both parents.
_h20_root=""
for _cand in \
    "$(cd "${ROOT_DIR}/../.." 2>/dev/null && pwd)/capa_h20" \
    "$(cd "${ROOT_DIR}/.." 2>/dev/null && pwd)/capa_h20"; do
  if [[ -d "${_cand}" ]]; then _h20_root="${_cand}"; break; fi
done
[[ -n "${_h20_root}" ]] || _h20_root="$(cd "${ROOT_DIR}/../.." && pwd)/capa_h20"
DEFAULT_ART_ROOT="${_h20_root}/artifacts/CAPA"
DEFAULT_MODELS_ROOT="${_h20_root}/models"

ART_ROOT="${ART_ROOT:-${DEFAULT_ART_ROOT}}"
H20_MODELS_ROOT="${H20_MODELS_ROOT:-${DEFAULT_MODELS_ROOT}}"
MODEL_4B="${MODEL_4B:-${H20_MODELS_ROOT}/Qwen3.5-4B}"
MODEL_35B="${MODEL_35B:-${H20_MODELS_ROOT}/Qwen3.5-35B-A3B}"

INFER_VENV="${INFER_VENV:-${ROOT_DIR}/.venv-h20-infer}"
TRAIN_VENV="${TRAIN_VENV:-${ROOT_DIR}/.venv-qwen35-grpo}"
INFER_PY="${INFER_VENV}/bin/python"
TRAIN_PY="${TRAIN_VENV}/bin/python"

VLLM_4B_PORT="${VLLM_4B_PORT:-8001}"
VLLM_35B_PORT="${VLLM_35B_PORT:-8002}"
VLLM_LOG_DIR="${ART_ROOT}/logs/vllm"

REPRO_ROOT="${ART_ROOT}/repro_h20"
PREFLIGHT_DIR="${REPRO_ROOT}/preflight"
EVAL_ROOT="${REPRO_ROOT}/eval"
SFT_ROOT="${REPRO_ROOT}/sft"
GRPO_ROOT="${REPRO_ROOT}/grpo"
COMPARE_ROOT="${REPRO_ROOT}/compare"
GATE_ROOT="${REPRO_ROOT}/gate"
STATUS_DIR="${REPRO_ROOT}/status"

SEEDS="${SEEDS:-42 43 44}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"

STAMP="$(date +%Y%m%d_%H%M%S)"

# The Qwen3.5-{4B,35B-A3B} arms served by our local vLLM are text-only
# Planners. Uploading images to a non-multimodal endpoint returns HTTP 400
# and every downstream step falls back to 'answerer'. The historical 0.79-
# accuracy focused_val_v3 run recorded ``omit_model_image_payload=true``;
# reproduce that contract here.
export CAPA_OMIT_MODEL_IMAGE_PAYLOAD="${CAPA_OMIT_MODEL_IMAGE_PAYLOAD:-1}"
# Some Planner code paths look for the same flag under this name.
export OMIT_MODEL_IMAGE_PAYLOAD="${OMIT_MODEL_IMAGE_PAYLOAD:-1}"
# The vendored trl.chat_template_utils shim in .venv-qwen35-grpo hard-codes
# /raid/zkq/models/Qwen3.5-4B for its tokenizer_config lookup; redirect it to
# the actual on-disk model dir so SFT/GRPO can import trl without file-not-found.
export CAPA_QWEN35_TOKENIZER_DIR="${CAPA_QWEN35_TOKENIZER_DIR:-${MODEL_4B}}"
# The frozen contract expected Qwen3.5-internal eos=248046 / pad=248044. The
# public Qwen3-4B tokenizer we ship here uses 151645 / 151643 (same behaviour,
# different id table). SFT/GRPO data is stored as raw text, so the gate is a
# provenance check; override it to match the public tokenizer.
export CAPA_EXPECTED_EOS_ID="${CAPA_EXPECTED_EOS_ID:-151645}"
export CAPA_EXPECTED_PAD_ID="${CAPA_EXPECTED_PAD_ID:-151643}"
export CAPA_EXPECTED_MODEL_CLASS="${CAPA_EXPECTED_MODEL_CLASS:-Qwen3ForCausalLM}"
# The frozen prompt_token_count / completion_token_count fields were computed
# with the internal Qwen3.5-4B tokenizer; the public Qwen3-4B tokenizer produces
# equivalent text but different id lengths. Allow SFT audit to bypass drift.
export CAPA_SKIP_TOKEN_COUNT_DRIFT="${CAPA_SKIP_TOKEN_COUNT_DRIFT:-1}"
# Make sure our local vLLM is not blocked by outer proxies.
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy 2>/dev/null || true

# Datasets (scenario -> cases file; sealed test is separate).
# NOTE: softbnd_dev/softbnd_test default to v7_longobs (regenerated 2026-08-02).
# v6 remains available by overriding CASES_SOFTBND_DEV / CASES_SOFTBND_TEST.
CASES_ROUTING90="${CASES_ROUTING90:-training/planner_dpo_train_seed_v1/eval/planner_routing_eval_90cases.json}"
CASES_MULTISTEP="${CASES_MULTISTEP:-training/planner_grpo_seed_v1/cases/planner_grpo_focused_val_v3_cases.jsonl}"
CASES_SOFTBND_DEV="${CASES_SOFTBND_DEV:-training/planner_grpo_seed_v1/cases/planner_retry_migrate_v7_longobs_grpo_dev_cases.jsonl}"
CASES_SOFTBND_TEST="${CASES_SOFTBND_TEST:-training/planner_grpo_seed_v1/cases/planner_retry_migrate_v7_longobs_test_cases.jsonl}"

SFT_DATA_DIR="${SFT_DATA_DIR:-training/planner_grpo_seed_v1/sft_data_planner_retry_migrate_v7_longobs_qwen35_nothinking}"
GRPO_STEP_TRAIN="${GRPO_STEP_TRAIN:-training/planner_grpo_seed_v1/step_data/planner_retry_migrate_v7_longobs_grpo_train_qwen35_4b_nothinking_step2.jsonl}"

mkdir -p "${PREFLIGHT_DIR}" "${EVAL_ROOT}" "${SFT_ROOT}" "${GRPO_ROOT}" \
         "${COMPARE_ROOT}" "${GATE_ROOT}" "${STATUS_DIR}" "${VLLM_LOG_DIR}"

# ---------- helpers ----------------------------------------------------------

log()  { printf '[%s %s] %s\n' "$(date +%H:%M:%S)" "${SELF_PHASE:-driver}" "$*" >&2; }
die()  { log "FATAL: $*"; exit 2; }

run() {
  log "\$ $*"
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  "$@"
}

require_file() { [[ -f "$1" ]] || die "missing file: $1${2:+  ($2)}"; }
require_dir()  { [[ -d "$1" ]] || die "missing dir: $1${2:+  ($2)}"; }

require_infer() {
  [[ -x "${INFER_PY}" ]] || die "infer venv broken: ${INFER_PY} not executable.  Rebuild with: bash scripts/reproduce/setup_h20_env.sh TARGET=infer"
}
require_train() {
  [[ -x "${TRAIN_PY}" ]] || die "train venv broken: ${TRAIN_PY} not executable.  Rebuild with: bash scripts/reproduce/setup_h20_env.sh TARGET=train"
}

gpus_free_or_die() {
  local wanted="$1"    # comma-separated GPU indices
  IFS=',' read -r -a ids <<< "${wanted}"
  local busy=0
  for id in "${ids[@]}"; do
    local n
    n="$(nvidia-smi -i "${id}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' | wc -l)"
    if [[ "${n}" -ne 0 ]]; then
      log "GPU ${id} busy: ${n} compute proc(s)"
      busy=$((busy + n))
    fi
  done
  [[ "${busy}" -eq 0 ]] || die "GPUs ${wanted} not free; stop other jobs or run phase 'stop' first"
}

mark_done()  { [[ "${DRY_RUN}" == "1" ]] && return 0; date -Iseconds > "${STATUS_DIR}/$1.done"; }
is_done()    { [[ -f "${STATUS_DIR}/$1.done" ]]; }
skip_if_done() {
  if is_done "$1"; then
    if [[ "${FORCE}" == "1" ]]; then
      log "phase $1 already done, FORCE=1 -> rerun"
      rm -f "${STATUS_DIR}/$1.done"
    else
      log "phase $1 already done (see ${STATUS_DIR}/$1.done); skipping.  Set FORCE=1 to rerun."
      return 1
    fi
  fi
  return 0
}

# ---------- vLLM serve/stop --------------------------------------------------

_vllm_up() {
  local pid_file="$1"
  [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null
}

serve_4b() {
  SELF_PHASE="serve-4b"
  require_infer
  require_dir "${MODEL_4B}" "expected Qwen3.5-4B at ${MODEL_4B}"
  local pid_file="${VLLM_LOG_DIR}/vllm_4b.pid"
  if _vllm_up "${pid_file}"; then
    log "vLLM 4B already up (pid $(cat "${pid_file}"))"
    return 0
  fi
  gpus_free_or_die "0"
  log "spawning vLLM 4B on GPU 0, port ${VLLM_4B_PORT}"
  if [[ "${DRY_RUN}" == "1" ]]; then log "(dry) skip spawn"; return 0; fi
  MODELS_ROOT="${H20_MODELS_ROOT}" \
    CUDA_VISIBLE_DEVICES=0 PORT="${VLLM_4B_PORT}" \
    LOG_DIR="${VLLM_LOG_DIR}" \
    nohup bash scripts/reproduce/serve_qwen35_vllm.sh 4b \
    >>"${VLLM_LOG_DIR}/vllm_4b.out" 2>&1 &
  disown || true
  log "waiting for endpoint http://127.0.0.1:${VLLM_4B_PORT}/v1/models ..."
  for _ in $(seq 1 180); do
    if "${INFER_PY}" -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:${VLLM_4B_PORT}/v1/models', timeout=3)" 2>/dev/null; then
      log "vLLM 4B ready"; return 0
    fi
    sleep 5
  done
  die "vLLM 4B did not come up in 15 min"
}

serve_35b() {
  SELF_PHASE="serve-35b"
  require_infer
  require_dir "${MODEL_35B}" "expected Qwen3.5-35B-A3B at ${MODEL_35B}"
  local pid_file="${VLLM_LOG_DIR}/vllm_35b.pid"
  if _vllm_up "${pid_file}"; then
    log "vLLM 35B already up (pid $(cat "${pid_file}"))"
    return 0
  fi
  gpus_free_or_die "0,1,2,3"
  log "spawning vLLM 35B on GPU 0-3 (TP=4), port ${VLLM_35B_PORT}"
  if [[ "${DRY_RUN}" == "1" ]]; then log "(dry) skip spawn"; return 0; fi
  MODELS_ROOT="${H20_MODELS_ROOT}" \
    CUDA_VISIBLE_DEVICES=0,1,2,3 PORT="${VLLM_35B_PORT}" \
    TENSOR_PARALLEL_SIZE=4 LOG_DIR="${VLLM_LOG_DIR}" \
    nohup bash scripts/reproduce/serve_qwen35_vllm.sh 35b \
    >>"${VLLM_LOG_DIR}/vllm_35b.out" 2>&1 &
  disown || true
  log "waiting for endpoint http://127.0.0.1:${VLLM_35B_PORT}/v1/models ..."
  for _ in $(seq 1 360); do
    if "${INFER_PY}" -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:${VLLM_35B_PORT}/v1/models', timeout=3)" 2>/dev/null; then
      log "vLLM 35B ready"; return 0
    fi
    sleep 5
  done
  die "vLLM 35B did not come up in 30 min"
}

serve_local() {
  # Serve an arbitrary local model dir on port 8001 (single GPU).
  # Used for SFT- and GRPO-merged checkpoints.
  SELF_PHASE="serve-local"
  require_infer
  local model_dir="${1:?model_dir required}"
  local served_name="${2:?served_name required}"
  require_dir "${model_dir}"
  local pid_file="${VLLM_LOG_DIR}/vllm_4b.pid"
  if _vllm_up "${pid_file}"; then
    log "an existing vLLM is running on port ${VLLM_4B_PORT}; stopping to re-serve ${served_name}"
    kill "$(cat "${pid_file}")" 2>/dev/null || true
    rm -f "${pid_file}"
    sleep 5
  fi
  gpus_free_or_die "0"
  log "spawning vLLM local model=${model_dir} name=${served_name} GPU=0 port=${VLLM_4B_PORT}"
  if [[ "${DRY_RUN}" == "1" ]]; then log "(dry) skip spawn"; return 0; fi
  CUDA_VISIBLE_DEVICES=0 \
    nohup "${INFER_PY}" -m vllm.entrypoints.openai.api_server \
      --model "${model_dir}" \
      --served-model-name "${served_name}" \
      --host 127.0.0.1 --port "${VLLM_4B_PORT}" \
      --dtype bfloat16 --tensor-parallel-size 1 \
      --max-model-len "${VLLM_MAX_MODEL_LEN:-12288}" --gpu-memory-utilization 0.90 \
      --trust-remote-code \
    >>"${VLLM_LOG_DIR}/vllm_local_${served_name}.out" 2>&1 &
  local pid=$!
  echo "${pid}" > "${pid_file}"
  disown || true
  for _ in $(seq 1 180); do
    if "${INFER_PY}" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${VLLM_4B_PORT}/v1/models', timeout=3)" 2>/dev/null; then
      log "vLLM local ready (${served_name})"
      return 0
    fi
    sleep 5
  done
  die "vLLM local (${served_name}) did not come up in 15 min"
}

stop_vllm() {
  SELF_PHASE="stop"
  local any=0
  for pid_file in "${VLLM_LOG_DIR}/"vllm_*.pid; do
    [[ -f "${pid_file}" ]] || continue
    local pid; pid="$(cat "${pid_file}")"
    if kill -0 "${pid}" 2>/dev/null; then
      log "stopping vllm pid=${pid} (${pid_file})"
      kill "${pid}" 2>/dev/null || true
      any=1
    fi
    rm -f "${pid_file}"
  done
  if [[ "${any}" -eq 0 ]]; then log "no vllm running"; fi
  # Best-effort wait for GPU release.
  sleep 8
}

# ---------- prep -------------------------------------------------------------

phase_prep() {
  SELF_PHASE="prep"
  skip_if_done "prep" || return 0
  require_infer
  require_file "${GRPO_STEP_TRAIN}" "run builder first: build_planner_retry_migrate_v6.py"
  log "generating V6 GRPO step-data manifest sidecars"
  run "${INFER_PY}" scripts/reproduce/write_v6_grpo_step_manifest.py \
      --model-dir "${MODEL_4B}"
  require_file "${GRPO_STEP_TRAIN%.jsonl}.manifest.json"
  mark_done "prep"
}

# ---------- preflight --------------------------------------------------------

phase_preflight() {
  SELF_PHASE="preflight"
  require_infer
  local out="${PREFLIGHT_DIR}/preflight_${STAMP}.json"
  run "${INFER_PY}" scripts/reproduce_preflight.py --out "${out}" || true
  log "preflight -> ${out}"
}

# ---------- eval (single arm across three scenarios) ------------------------

# Args: arm_tag  model_id  api_base
_eval_arm_all_scenarios() {
  SELF_PHASE="eval-${1}"
  local arm="$1" model_id="$2" api_base="$3"
  local scen_stamp="${SCEN_STAMP:-${STAMP}}"
  local base="${EVAL_ROOT}/${scen_stamp}_${arm}"
  mkdir -p "${base}"

  # Sanity: endpoint must publish this model id.
  run "${INFER_PY}" - <<PY
import json, sys, urllib.request
url = "${api_base}/models"
data = json.loads(urllib.request.urlopen(url, timeout=15).read())
served = {m.get("id") for m in data.get("data", [])}
assert "${model_id}" in served, f"'${model_id}' not in {sorted(served)}"
print("endpoint ok:", sorted(served))
PY

  # 1) single-step routing 90 case  (JSON schema, uses demo/eval evaluator)
  if [[ ! -f "${base}/routing90/summary_aggregate.json" ]]; then
    mkdir -p "${base}/routing90"
    run "${INFER_PY}" demo/eval/run_repeated_planner_routing_eval.py \
      --cases "${CASES_ROUTING90}" \
      --out-dir "${base}/routing90" \
      --report-prefix "${arm}" \
      --model "${model_id}" --api-base "${api_base}" \
      --runs 3 --timeout-seconds 600 \
      --temperature 0 --top-p 1 --seed 42 --do-sample false
    # Aggregate the 3 per-run reports into a single file for the summary step.
    run "${INFER_PY}" - <<PY
import glob, json, statistics as st
runs = sorted(glob.glob("${base}/routing90/planner_routing_report_${arm}_*.json")) or \
       sorted(glob.glob("${base}/routing90/${arm}_run*.json"))
accs = []
for path in runs:
    payload = json.load(open(path, encoding="utf-8"))
    summ = payload.get("summary") or {}
    if "accuracy" in summ:
        accs.append(float(summ["accuracy"]))
agg = {
    "scenario": "routing90",
    "runs": len(accs),
    "accuracy_mean": (sum(accs) / len(accs)) if accs else None,
    "accuracy_stdev": (st.pstdev(accs) if len(accs) > 1 else 0.0),
    "per_run_paths": runs,
}
open("${base}/routing90/summary_aggregate.json", "w", encoding="utf-8").write(
    json.dumps(agg, ensure_ascii=False, indent=2))
print(agg)
PY
  else
    log "routing90 already present for ${arm}; skip"
  fi

  # 2) multi-step routing (focused_val_v3, 31 case)
  # NOTE: --max-tokens caps Planner *per-step* completion (default 384). We
  # previously set 4096 here, which combined with vLLM --max-model-len 8192 blew
  # past the context budget on step 2/3 and forced every soft-boundary case to
  # fall back to 'answerer'. Keep this well below (max_model_len - prompt).
  if [[ ! -f "${base}/multistep/${arm}_aggregate.json" ]]; then
    mkdir -p "${base}/multistep"
    run "${INFER_PY}" training/planner_grpo_seed_v1/scripts/run_repeated_planner_grpo_eval.py \
      --cases "${CASES_MULTISTEP}" \
      --out-dir "${base}/multistep" \
      --report-prefix "${arm}" \
      --model "${model_id}" --api-base "${api_base}" \
      --runs 3 --max-steps 3 --max-tokens "${PLANNER_MAX_TOKENS:-512}" \
      --temperature 0 --top-p 1 --seed 42 --do-sample false \
      --timeout-seconds 600 --openai-timeout-seconds 600
  else
    log "multistep already present for ${arm}; skip"
  fi

  # 3) soft-boundary retry_migrate_v6 grpo_dev (225 case)
  if [[ ! -f "${base}/softbnd_dev/${arm}_aggregate.json" ]]; then
    mkdir -p "${base}/softbnd_dev"
    run "${INFER_PY}" training/planner_grpo_seed_v1/scripts/run_repeated_planner_grpo_eval.py \
      --cases "${CASES_SOFTBND_DEV}" \
      --out-dir "${base}/softbnd_dev" \
      --report-prefix "${arm}" \
      --model "${model_id}" --api-base "${api_base}" \
      --runs 3 --max-steps 3 --max-tokens "${PLANNER_MAX_TOKENS:-512}" \
      --temperature 0 --top-p 1 --seed 42 --do-sample false \
      --timeout-seconds 600 --openai-timeout-seconds 600
  else
    log "softbnd_dev already present for ${arm}; skip"
  fi

  # Cross-scenario summary.
  run "${INFER_PY}" - <<PY
import glob, json, os
base = "${base}"
rows = []
patterns = [
    ("routing90", base + "/routing90/summary_aggregate.json"),
]
for scen, path in patterns:
    if os.path.isfile(path):
        payload = json.load(open(path, encoding="utf-8"))
        rows.append({
            "scenario": scen,
            "case_macro_mean": None,
            "case_pass_rate": payload.get("accuracy_mean"),
            "case_pass_rate_stdev": payload.get("accuracy_stdev"),
            "runs": payload.get("runs"),
            "file": path,
        })
# For evaluator-produced aggregates (multistep / softbnd_dev).
for path in sorted(glob.glob(base + "/*/*_aggregate.json")):
    scen = os.path.basename(os.path.dirname(path))
    if scen == "routing90":
        continue
    try:
        payload = json.load(open(path, encoding="utf-8"))
    except Exception as exc:
        rows.append({"scenario": scen, "error": str(exc), "file": path}); continue
    rows.append({
        "scenario": scen,
        "case_macro_mean": payload.get("case_macro_mean")
            or payload.get("aggregate", {}).get("mean_score_mean"),
        "case_pass_rate": payload.get("case_pass_rate")
            or payload.get("aggregate", {}).get("pass_rate_mean")
            or payload.get("summary", {}).get("accuracy"),
        "runs": payload.get("runs"),
        "file": path,
    })
summary = {"arm": "${arm}", "stamp": "${scen_stamp}", "results": rows}
out = os.path.join(base, "summary.json")
open(out, "w", encoding="utf-8").write(json.dumps(summary, ensure_ascii=False, indent=2))
print("summary ->", out)
for r in rows:
    print(" -", r)
PY

  mark_done "eval-${arm}"
}

phase_base_eval_4b() {
  skip_if_done "eval-base_4b" || return 0
  serve_4b
  _eval_arm_all_scenarios "base_4b" "Qwen3.5-4B" "http://127.0.0.1:${VLLM_4B_PORT}/v1"
}

phase_base_eval_35b() {
  skip_if_done "eval-base_35b" || return 0
  stop_vllm
  serve_35b
  _eval_arm_all_scenarios "base_35b" "Qwen3.5-35B-A3B" "http://127.0.0.1:${VLLM_35B_PORT}/v1"
}

# ---------- SFT + eval -------------------------------------------------------

phase_sft() {
  SELF_PHASE="sft"
  skip_if_done "sft" || return 0
  require_train
  require_dir "${MODEL_4B}"
  require_file "${SFT_DATA_DIR}/metadata.json"
  gpus_free_or_die "0,1,2,3"

  local run_id="${STAMP}_qwen35_4b_planner_v6_sft"
  local out_dir="${SFT_ROOT}/${run_id}"
  [[ -e "${out_dir}" ]] && die "output dir exists: ${out_dir}"

  log "SFT full run -> ${out_dir}"
  CONFIRM_TRAIN=YES \
    ENV_DIR="${TRAIN_VENV}" MODEL_PATH="${MODEL_4B}" \
    DATA_DIR="${SFT_DATA_DIR}" \
    CAPA_EXPECTED_DATASET_ID="${SFT_EXPECTED_DATASET_ID:-planner_retry_migrate_v7_longobs}" \
    CAPA_EXPECTED_SFT_TRAIN_ROWS="${SFT_EXPECTED_TRAIN_ROWS:-1280}" \
    CAPA_EXPECTED_SFT_DEV_ROWS="${SFT_EXPECTED_DEV_ROWS:-320}" \
    CAPA_EXPECTED_LORA_MODULES="${CAPA_EXPECTED_LORA_MODULES:-144}" \
    CAPA_EXPECTED_TRAINABLE_PARAMS="${CAPA_EXPECTED_TRAINABLE_PARAMS:-11796480}" \
    CAPA_ALLOW_MAX_LENGTH="${CAPA_ALLOW_MAX_LENGTH:-1}" \
    MAX_LENGTH="${MAX_LENGTH:-10240}" \
    CUDA_VISIBLE_DEVICES="0,1,2,3" NUM_PROCESSES=4 \
    RUN_MODE=train \
    REPORT_TO="${REPORT_TO:-none}" \
    MAX_STEPS="${SFT_MAX_STEPS:-400}" \
    LEARNING_RATE="${SFT_LR:-2e-5}" \
    GRADIENT_ACCUMULATION_STEPS="${SFT_GRAD_ACCUM:-2}" \
    SAVE_STEPS="${SFT_SAVE_STEPS:-100}" \
    SAVE_TOTAL_LIMIT="${SFT_SAVE_TOTAL_LIMIT:-4}" \
    RUN_ID="${run_id}" OUTPUT_DIR="${out_dir}" \
    run bash scripts/run_qwen35_4b_planner_v6_sft.sh
  echo "${out_dir}" > "${STATUS_DIR}/sft.output_dir"
  mark_done "sft"
}

_latest_sft_dir() {
  if [[ -f "${STATUS_DIR}/sft.output_dir" ]]; then
    cat "${STATUS_DIR}/sft.output_dir"
    return 0
  fi
  ls -1td "${SFT_ROOT}"/*qwen35_4b_planner_v6_sft* 2>/dev/null | head -n1
}

_latest_sft_checkpoint() {
  local sft_dir; sft_dir="$(_latest_sft_dir)"
  [[ -n "${sft_dir}" ]] || { echo ""; return 0; }
  # SFT saturates very early on v7-longobs (eval_loss < 0.005 by step 100).
  # Later checkpoints overfit to the hint literal and hurt generalisation.
  # Allow explicit selection via SFT_CHECKPOINT_STEP; default to the earliest
  # available checkpoint (usually step 100).
  if [[ -n "${SFT_CHECKPOINT_STEP:-}" ]]; then
    local pinned="${sft_dir}/checkpoint-${SFT_CHECKPOINT_STEP}"
    if [[ -d "${pinned}" ]]; then echo "${pinned}"; return 0; fi
    log "SFT_CHECKPOINT_STEP=${SFT_CHECKPOINT_STEP} not found under ${sft_dir}; falling back to earliest"
  fi
  ls -1d "${sft_dir}"/checkpoint-* 2>/dev/null | sort -V | head -n1
}

phase_sft_merge() {
  SELF_PHASE="sft-merge"
  skip_if_done "sft-merge" || return 0
  require_train
  local ckpt; ckpt="$(_latest_sft_checkpoint)"
  [[ -n "${ckpt}" ]] || die "no SFT checkpoint found under ${SFT_ROOT}; run phase 'sft' first"
  local merged="${ckpt%/}_merged"
  if [[ -d "${merged}" ]]; then
    log "merged model already at ${merged}"
  else
    log "merging LoRA adapter -> ${merged}"
    run "${TRAIN_PY}" scripts/merge_lora_adapter.py \
      --base-model "${MODEL_4B}" \
      --adapter "${ckpt}" \
      --output-dir "${merged}"
  fi
  echo "${merged}" > "${STATUS_DIR}/sft.merged_dir"
  mark_done "sft-merge"
}

phase_sft_eval() {
  skip_if_done "eval-sft" || return 0
  local merged
  merged="$(cat "${STATUS_DIR}/sft.merged_dir" 2>/dev/null || true)"
  [[ -n "${merged}" && -d "${merged}" ]] || die "SFT merged model not found; run phase 'sft-merge' first"
  stop_vllm
  serve_local "${merged}" "qwen35_4b_sft"
  _eval_arm_all_scenarios "sft" "qwen35_4b_sft" "http://127.0.0.1:${VLLM_4B_PORT}/v1"
}

# ---------- GRPO x seeds + eval ---------------------------------------------

phase_grpo() {
  SELF_PHASE="grpo"
  skip_if_done "grpo" || return 0
  require_train
  require_file "${GRPO_STEP_TRAIN}"
  require_file "${GRPO_STEP_TRAIN%.jsonl}.manifest.json" "run phase 'prep' first"
  local ckpt; ckpt="$(_latest_sft_checkpoint)"
  [[ -n "${ckpt}" ]] || die "no SFT checkpoint to seed GRPO; run phase 'sft' first"

  for seed in ${SEEDS}; do
    if is_done "grpo-${seed}" && [[ "${FORCE}" != "1" ]]; then
      log "grpo seed=${seed} already done; skip"
      continue
    fi
    rm -f "${STATUS_DIR}/grpo-${seed}.done"
    gpus_free_or_die "0,1,2,3"
    local run_id="${STAMP}_qwen35_4b_v6_grpo_seed${seed}"
    local out_dir="${GRPO_ROOT}/${run_id}"
    [[ -e "${out_dir}" ]] && die "output dir exists: ${out_dir}"
    log "GRPO seed=${seed} -> ${out_dir}"

    CONFIRM_TRAIN=YES \
      ENV_DIR="${TRAIN_VENV}" MODEL_PATH="${MODEL_4B}" \
      ADAPTER_PATH="${ckpt}" \
      STEP_DATA="${GRPO_STEP_TRAIN}" \
      EXPECTED_DATASET_ID="${GRPO_EXPECTED_DATASET_ID:-planner_retry_migrate_v7_longobs}" \
      EXPECTED_ROWS="${GRPO_EXPECTED_ROWS:-480}" \
      MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-8192}" \
      MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-320}" \
      CAPA_EXPECTED_LORA_MODULES="${CAPA_EXPECTED_LORA_MODULES:-144}" \
      CAPA_EXPECTED_TRAINABLE_PARAMS="${CAPA_EXPECTED_TRAINABLE_PARAMS:-11796480}" \
      CUDA_VISIBLE_DEVICES="0,1,2,3" \
      SEED="${seed}" \
      RUN_MODE="${GRPO_RUN_MODE:-screen}" \
      REPORT_TO="${REPORT_TO:-none}" \
      MAX_STEPS="${GRPO_MAX_STEPS:-100}" \
      LEARNING_RATE="${GRPO_LR:-5e-6}" \
      SAVE_STEPS="${GRPO_SAVE_STEPS:-25}" SAVE_TOTAL_LIMIT="${GRPO_SAVE_TOTAL_LIMIT:-4}" \
      OUTPUT_DIR="${out_dir}" \
      run bash scripts/run_qwen35_4b_grpo_v5_train_v1.sh
    echo "${out_dir}" > "${STATUS_DIR}/grpo-${seed}.output_dir"
    mark_done "grpo-${seed}"
  done
  mark_done "grpo"
}

_grpo_ckpt_for() {
  local seed="$1"
  local out_dir
  out_dir="$(cat "${STATUS_DIR}/grpo-${seed}.output_dir" 2>/dev/null || true)"
  [[ -n "${out_dir}" && -d "${out_dir}" ]] || {
    out_dir="$(ls -1td "${GRPO_ROOT}"/*_seed${seed}* 2>/dev/null | head -n1)"
  }
  [[ -n "${out_dir}" ]] || { echo ""; return 0; }
  ls -1d "${out_dir}"/checkpoint-* 2>/dev/null | sort -V | tail -n1
}

phase_grpo_eval() {
  for seed in ${SEEDS}; do
    if is_done "eval-grpo${seed}" && [[ "${FORCE}" != "1" ]]; then continue; fi
    local ckpt; ckpt="$(_grpo_ckpt_for "${seed}")"
    [[ -n "${ckpt}" ]] || die "no GRPO checkpoint for seed=${seed}"
    local merged="${ckpt%/}_merged"
    if [[ ! -d "${merged}" ]]; then
      require_train
      log "merging GRPO adapter seed=${seed} -> ${merged}"
      run "${TRAIN_PY}" scripts/merge_lora_adapter.py \
        --base-model "${MODEL_4B}" --adapter "${ckpt}" --output-dir "${merged}"
    fi
    stop_vllm
    serve_local "${merged}" "qwen35_4b_grpo${seed}"
    _eval_arm_all_scenarios "grpo${seed}" "qwen35_4b_grpo${seed}" "http://127.0.0.1:${VLLM_4B_PORT}/v1"
  done
  mark_done "eval-grpo"
}

# ---------- compare + gate + registry ---------------------------------------

phase_compare() {
  SELF_PHASE="compare"
  require_train
  local out="${COMPARE_ROOT}/compare_${STAMP}.json"
  local baseline; baseline="$(ls -1td "${EVAL_ROOT}"/*_sft 2>/dev/null | head -n1)"
  [[ -n "${baseline}" ]] || die "no SFT eval baseline"
  local candidates=()
  for seed in ${SEEDS}; do
    local c; c="$(ls -1td "${EVAL_ROOT}"/*_grpo${seed} 2>/dev/null | head -n1)"
    [[ -n "${c}" ]] && candidates+=("${c}")
  done
  [[ ${#candidates[@]} -gt 0 ]] || die "no GRPO eval candidates"
  log "compare baseline=${baseline} candidates=${candidates[*]}"
  # Reuse the existing paired comparator; scenario dirs are handled by the caller.
  run "${TRAIN_PY}" pipelines/eval/compare_generation_runs.py \
    --baseline "${baseline}/softbnd_dev" \
    --candidates $(printf '%s/softbnd_dev ' "${candidates[@]}") \
    --out "${out}"
  log "compare -> ${out}"
}

phase_gate() {
  SELF_PHASE="gate"
  require_train
  local baseline; baseline="$(ls -1td "${EVAL_ROOT}"/*_sft 2>/dev/null | head -n1)"
  [[ -n "${baseline}" ]] || die "no SFT eval baseline"
  local candidates=()
  for seed in ${SEEDS}; do
    local c; c="$(ls -1td "${EVAL_ROOT}"/*_grpo${seed} 2>/dev/null | head -n1)"
    [[ -n "${c}" ]] && candidates+=("${c}")
  done
  local study_json="experiments/studies/planner_runtime_routing_grpo_v1/study.json"
  local out="${GATE_ROOT}/gate_${STAMP}.json"
  run "${TRAIN_PY}" pipelines/eval/check_runtime_routing_multiseed_gate.py \
    --study "${study_json}" \
    --baseline "${baseline}/softbnd_dev" \
    --candidates $(printf '%s/softbnd_dev ' "${candidates[@]}") \
    --out "${out}" || true
  log "gate -> ${out}"
  log "if gate passes, phase 'sealed' becomes eligible (one-shot)"
}

phase_sealed() {
  SELF_PHASE="sealed"
  local gate_file
  gate_file="$(ls -1t "${GATE_ROOT}"/gate_*.json 2>/dev/null | head -n1)"
  [[ -n "${gate_file}" ]] || die "no gate report; run phase 'gate' first"
  local passed
  passed="$("${INFER_PY}" -c "import json,sys;print(json.load(open('${gate_file}')).get('passed',False))")"
  if [[ "${passed}" != "True" ]]; then
    die "gate did NOT pass; sealed test remains closed (${gate_file})"
  fi
  # Preselected candidate: seed 42.
  local ckpt; ckpt="$(_grpo_ckpt_for 42)"
  [[ -n "${ckpt}" ]] || die "no GRPO seed-42 checkpoint"
  local merged="${ckpt%/}_merged"
  [[ -d "${merged}" ]] || die "merged model missing at ${merged}; run phase 'grpo-eval' first"
  stop_vllm
  serve_local "${merged}" "qwen35_4b_grpo42_sealed"
  local out_dir="${EVAL_ROOT}/${STAMP}_grpo42_sealed"
  mkdir -p "${out_dir}"
  run "${INFER_PY}" training/planner_grpo_seed_v1/scripts/run_repeated_planner_grpo_eval.py \
    --cases "${CASES_SOFTBND_TEST}" \
    --out-dir "${out_dir}" \
    --report-prefix "grpo42_sealed" \
    --model "qwen35_4b_grpo42_sealed" \
    --api-base "http://127.0.0.1:${VLLM_4B_PORT}/v1" \
    --runs 3 --max-steps 3 --max-tokens "${PLANNER_MAX_TOKENS:-512}" \
    --temperature 0 --top-p 1 --seed 42 --do-sample false \
    --timeout-seconds 600 --openai-timeout-seconds 600
}

# ---------- aggregate presets -----------------------------------------------

phase_all_base() {
  phase_prep
  phase_preflight
  phase_base_eval_4b
  phase_base_eval_35b
  stop_vllm
}

phase_all_train() {
  phase_prep
  stop_vllm
  phase_sft
  phase_sft_merge
  phase_sft_eval
  stop_vllm
  phase_grpo
  phase_grpo_eval
  stop_vllm
  phase_compare
  phase_gate
  log "train pipeline finished; phase 'sealed' is a manual one-shot"
}

# ---------- dispatch --------------------------------------------------------

usage() {
  cat <<'EOF'
Usage: scripts/reproduce/run_h20_repro.sh <phase> [<phase> ...]

Prep and base eval:
  prep                  build V6 GRPO step-data manifest sidecar
  preflight             run scripts/reproduce_preflight.py
  serve-4b              start vLLM Qwen3.5-4B on GPU 0 (port 8001)
  serve-35b             start vLLM Qwen3.5-35B-A3B on GPU 0-3 TP=4 (port 8002)
  stop                  stop all vllm servers spawned here
  base-eval-4b          3x eval base 4B across 3 scenarios
  base-eval-35b         3x eval base 35B-A3B across 3 scenarios
  all-base              prep + preflight + base-eval-4b + base-eval-35b + stop

Training and downstream eval:
  sft                   SFT on planner_retry_migrate_v6 (4x H20)
  sft-merge             merge SFT LoRA to <sft>/checkpoint-*_merged
  sft-eval              serve merged SFT model, 3x eval across 3 scenarios
  grpo                  GRPO for each seed in $SEEDS on top of SFT checkpoint
  grpo-eval             merge and eval every seed
  compare               case-macro paired compare on softbnd_dev
  gate                  preregistered multi-seed dev gate on softbnd_dev
  sealed                open sealed V6 test once, only if gate passed
  all-train             prep + sft + sft-merge + sft-eval + grpo + grpo-eval + compare + gate

Env:
  ART_ROOT=<path>          override artifact root
  H20_MODELS_ROOT=<path>   override models root
  SEEDS="42 43 44"         GRPO seeds
  DRY_RUN=1                print commands only
  FORCE=1                  overwrite existing phase status
EOF
}

main() {
  [[ $# -ge 1 ]] || { usage; exit 1; }
  log "ROOT_DIR=${ROOT_DIR}"
  log "ART_ROOT=${ART_ROOT}"
  log "H20_MODELS_ROOT=${H20_MODELS_ROOT}"
  for phase in "$@"; do
    case "${phase}" in
      prep)              phase_prep ;;
      preflight)         phase_preflight ;;
      serve-4b)          serve_4b ;;
      serve-35b)         serve_35b ;;
      stop)              stop_vllm ;;
      base-eval-4b)      phase_base_eval_4b ;;
      base-eval-35b)     phase_base_eval_35b ;;
      all-base)          phase_all_base ;;
      sft)               phase_sft ;;
      sft-merge)         phase_sft_merge ;;
      sft-eval)          phase_sft_eval ;;
      grpo)              phase_grpo ;;
      grpo-eval)         phase_grpo_eval ;;
      compare)           phase_compare ;;
      gate)              phase_gate ;;
      sealed)            phase_sealed ;;
      all-train)         phase_all_train ;;
      -h|--help)         usage; exit 0 ;;
      *)                 die "unknown phase: ${phase}" ;;
    esac
  done
  log "done."
}

main "$@"
