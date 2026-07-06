#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:?set MODEL, e.g. qwen3.5-4b}"
API_BASE="${API_BASE:?set API_BASE, e.g. http://127.0.0.1:8003/v1}"
REPORT_PREFIX="${REPORT_PREFIX:?set REPORT_PREFIX, e.g. qwen35_4b_vllm_base_zip90}"
CASES="${CASES:-$ROOT_DIR/training/planner_dpo_train_seed_v1/eval/planner_routing_eval_90cases.json}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/results/planner_routing_eval}"
RUNS="${RUNS:-3}"
SEED="${SEED:-42}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1}"
DO_SAMPLE="${DO_SAMPLE:-false}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-180}"

export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

exec .venv-train/bin/python demo/eval/run_repeated_planner_routing_eval.py \
  --cases "$CASES" \
  --out-dir "$OUT_DIR" \
  --report-prefix "$REPORT_PREFIX" \
  --model "$MODEL" \
  --api-base "$API_BASE" \
  --runs "$RUNS" \
  --timeout-seconds "$TIMEOUT_SECONDS" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --seed "$SEED" \
  --do-sample "$DO_SAMPLE"
