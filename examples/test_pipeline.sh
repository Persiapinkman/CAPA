#!/usr/bin/env bash
set -euo pipefail

# Batch runner for full target-detection-evaluation skill.
#
# Usage:
#   bash examples/test_pipeline.sh [output_root] [api_base] [api_key] [qwen_base_url] [rex_base_url]
#
# Defaults:
#   output_root: results
#   api_base:    https://api.apiyi.com/v1
#   api_key:     from APIYI_API_KEY env; fallback to api_key.txt
#
# Notes:
#   - Runs the FULL pipeline (intent -> prompts -> flux -> qwen+rex -> report)
#   - Uses --num-images 3 to keep runtime reasonable.

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  echo "Usage:"
  echo "  bash examples/test_pipeline.sh [output_root] [api_base] [api_key] [qwen_base_url] [rex_base_url]"
  echo ""
  echo "Defaults:"
  echo "  output_root: results"
  echo "  api_base:    https://api.apiyi.com/v1"
  echo "  api_key:     APIYI_API_KEY env var; fallback to repo api_key.txt"
  echo ""
  exit 0
fi

OUTPUT_ROOT="${1:-results}"
API_BASE="${2:-https://api.apiyi.com/v1}"
API_KEY="${3:-${APIYI_API_KEY:-}}"
QWEN_BASE_URL="${4:-${DEMO_QWEN_BASE_URL:-}}"
REX_BASE_URL="${5:-${DEMO_REX_BASE_URL:-}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -z "${API_KEY}" ] && [ -f "${ROOT}/api_key.txt" ]; then
  API_KEY="$(head -n 1 "${ROOT}/api_key.txt" | tr -d '\r\n')"
fi

if [ -z "${API_KEY}" ]; then
  echo "Error: API key is empty."
  echo "Please set APIYI_API_KEY env var or pass api_key as arg #3."
  exit 1
fi

cd "${ROOT}"

run_case() {
  local case_name="$1"
  local image_path="$2"
  local task_text="$3"

  local case_dir="${OUTPUT_ROOT}/${case_name}"
  echo "=================================================="
  echo "Running pipeline case: ${case_name}"
  echo "Image: ${image_path}"
  echo "Task: ${task_text}"
  echo "Output: ${case_dir}"

  local cmd=(
    python3
    skills/target-detection-evaluation/scripts/run_pipeline.py
    --workspace "${case_dir}"
    --image "${image_path}"
    --text "${task_text}"
    --api-base "${API_BASE}"
    --api-key "${API_KEY}"
    --num-images 3
  )

  if [ -n "${QWEN_BASE_URL}" ]; then
    cmd+=( --qwen-base-url "${QWEN_BASE_URL}" )
  fi
  if [ -n "${REX_BASE_URL}" ]; then
    cmd+=( --rex-base-url "${REX_BASE_URL}" )
  fi

  "${cmd[@]}"
}

run_case "person_with_bag" "examples/images/person_with_bag.png" "检测背包的行人"
# run_case "banner" "examples/images/banner.jpg" "检测横幅"
# run_case "trash_truck" "examples/images/trash_truck.jpg" "检测垃圾车"
# run_case "smoke" "examples/images/smoke.jpg" "检测烟雾"
# run_case "fishing_person" "examples/images/fisherman.jpg" "检测钓鱼的人"

echo "=================================================="
echo "All pipeline cases finished. Outputs are under: ${OUTPUT_ROOT}"

