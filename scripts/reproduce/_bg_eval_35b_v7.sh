#!/usr/bin/env bash
# Blocking evaluator. Meant to be launched itself via nohup by the caller,
# or wrapped by `screen`/`tmux` — do NOT rely on internal detachment magic.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

OUT_DIR="${OUT_DIR:-/apdcephfs_hzlf/share_1227201/zkq/capa_h20/artifacts/CAPA/repro_h20/eval/base_35b_v7_1run}"
CASES="${CASES:-training/planner_grpo_seed_v1/cases/planner_retry_migrate_v7_longobs_grpo_dev_cases.jsonl}"
PREFIX="${PREFIX:-base35b_v7}"
MODEL="${MODEL:-Qwen3.5-35B-A3B}"
API_BASE="${API_BASE:-http://127.0.0.1:8002/v1}"
RUNS="${RUNS:-1}"

mkdir -p "${OUT_DIR}"

export CAPA_OMIT_MODEL_IMAGE_PAYLOAD=1 OMIT_MODEL_IMAGE_PAYLOAD=1
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy 2>/dev/null || true

exec .venv-h20-infer/bin/python \
  training/planner_grpo_seed_v1/scripts/run_repeated_planner_grpo_eval.py \
    --cases "${CASES}" \
    --out-dir "${OUT_DIR}" \
    --report-prefix "${PREFIX}" \
    --model "${MODEL}" \
    --api-base "${API_BASE}" \
    --runs "${RUNS}" --max-steps 3 --max-tokens 512 \
    --temperature 0 --top-p 1 --seed 42 --do-sample false \
    --timeout-seconds 600 --openai-timeout-seconds 600
