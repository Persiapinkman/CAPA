#!/usr/bin/env bash
set -euo pipefail

# Call solution-reports-generation and write JSON under results/.
#
# Usage:
#   bash examples/test_solution.sh [output_dir] [image_path] [api_base] [api_key] [model]
#
# Defaults:
#   output_dir: results/solution_report
#   image_path: examples/images/fisherman.jpg (relative to repo root)
#   api_base:   https://api.apiyi.com/v1
#   api_key:    APIYI_API_KEY env; fallback to repo api_key.txt
#   model:      gpt-4o
#
# Override the scenario text:
#   SOLUTION_TEST_TEXT="你的需求……" bash examples/test_solution.sh

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  echo "Usage:"
  echo "  bash examples/test_solution.sh [output_dir] [image_path] [api_base] [api_key] [model]"
  echo ""
  echo "Defaults:"
  echo "  output_dir: results/solution_report"
  echo "  image_path: examples/images/fisherman.jpg"
  echo "  api_key:    APIYI_API_KEY or ./api_key.txt"
  echo ""
  echo "Environment:"
  echo "  SOLUTION_TEST_TEXT  User requirement text (Chinese recommended)"
  exit 0
fi

OUT_DIR="${1:-results/solution_report}"
IMAGE_REL="${2:-examples/images/fisherman.jpg}"
API_BASE="${3:-https://api.apiyi.com/v1}"
API_KEY="${4:-${APIYI_API_KEY:-}}"
MODEL="${5:-gpt-4o}"

SOLUTION_TEST_TEXT="${SOLUTION_TEST_TEXT:-检测钓鱼的人}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -z "${API_KEY}" ] && [ -f "${ROOT}/api_key.txt" ]; then
  API_KEY="$(head -n 1 "${ROOT}/api_key.txt" | tr -d '\r\n')"
fi

if [ -z "${API_KEY}" ]; then
  echo "Error: API key is empty."
  echo "Set APIYI_API_KEY or pass api_key as argument #4."
  exit 1
fi

cd "${ROOT}"

IMAGE_ABS="${ROOT}/${IMAGE_REL}"
if [ ! -f "${IMAGE_ABS}" ]; then
  echo "Error: image not found: ${IMAGE_ABS}"
  exit 1
fi

mkdir -p "${OUT_DIR}"
OUT_JSON="${OUT_DIR}/solution_report.json"

echo "Skill:   solution-reports-generation"
echo "Image:   ${IMAGE_REL}"
echo "Out:     ${OUT_JSON}"
echo "Model:   ${MODEL}"
echo "--------------------------------------------------"

python3 skills/solution-reports-generation/scripts/run_solution_report.py \
  --text "${SOLUTION_TEST_TEXT}" \
  --image "${IMAGE_ABS}" \
  --api-base "${API_BASE}" \
  --api-key "${API_KEY}" \
  --model "${MODEL}" \
  --out "${OUT_JSON}"

echo "Done. Wrote: ${OUT_JSON}"
