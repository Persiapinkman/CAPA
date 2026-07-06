#!/usr/bin/env bash
set -euo pipefail

# Re-run evaluation only (eval-reports-generation) for existing cases.
# It expects each case already has:
#   - results/<case>/prediction.json
#   - results/<case>/generated_images/*
#   - results/<case>/intent.json (preferred for target_label)
#
# Usage:
#   bash examples/test_eval.sh [output_root] [api_base] [api_key]
#
# Defaults:
#   output_root: results
#   api_base:    https://api.apiyi.com/v1
#   api_key:     from APIYI_API_KEY env; fallback to api_key.txt

OUTPUT_ROOT="${1:-results}"
API_BASE="${2:-https://api.apiyi.com/v1}"
API_KEY="${3:-${APIYI_API_KEY:-}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -z "${API_KEY}" ] && [ -f "${ROOT}/api_key.txt" ]; then
  API_KEY="$(head -n 1 "${ROOT}/api_key.txt" | tr -d '\r\n')"
fi

if [ -z "${API_KEY}" ]; then
  echo "Error: API key is empty."
  echo "Please set APIYI_API_KEY or pass api_key as arg #3."
  exit 1
fi

cd "${ROOT}"

run_case() {
  local case_name="$1"
  local image_path="$2"
  local task_text="$3"

  local case_dir="${OUTPUT_ROOT}/${case_name}"
  local prediction_json="${case_dir}/prediction.json"
  local intent_json="${case_dir}/intent.json"
  local generated_dir="${case_dir}/generated_images"
  local evaluation_json="${case_dir}/evaluation.json"

  echo "=================================================="
  echo "Running eval case: ${case_name}"
  echo "Image: ${image_path}"
  echo "Task: ${task_text}"
  echo "Case dir: ${case_dir}"

  if [ ! -f "${prediction_json}" ]; then
    echo "Skip ${case_name}: missing ${prediction_json}"
    return
  fi
  if [ ! -d "${generated_dir}" ]; then
    echo "Skip ${case_name}: missing ${generated_dir}"
    return
  fi

  local target_label=""
  if [ -f "${intent_json}" ]; then
    target_label="$(python3 -c "import json,sys; d=json.load(open(sys.argv[1],'r',encoding='utf-8')); print(str(d.get('target_label','')).replace('_',' '))" "${intent_json}")"
  fi
  if [ -z "${target_label}" ]; then
    # Fallback: from task text
    target_label="${task_text}"
  fi

  # Build images list: original first, then generated images sorted by filename
  local images_json
  images_json="$(python3 - "${image_path}" "${generated_dir}" <<'PY'
import json, sys
from pathlib import Path

orig = str(Path(sys.argv[1]).resolve())
gen_dir = Path(sys.argv[2]).resolve()
exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
gens = [str(p.resolve()) for p in sorted(gen_dir.iterdir(), key=lambda x: x.name) if p.is_file() and p.suffix.lower() in exts]
print(json.dumps([orig] + gens, ensure_ascii=False))
PY
)"

  python3 skills/eval-reports-generation/scripts/run_eval_report_generation.py \
    --images "${images_json}" \
    --prediction "${prediction_json}" \
    --task-text "${task_text}" \
    --target-label "${target_label}" \
    --api-base "${API_BASE}" \
    --api-key "${API_KEY}" \
    --out "${evaluation_json}"
}

# Same cases as examples/test.sh
: <<'COMMENT'
run_case "person_with_bag" "examples/images/person_with_bag.png" "检测背包的行人"
run_case "banner" "examples/images/banner.jpg" "检测横幅"
run_case "trash_truck" "examples/images/trash_truck.jpg" "检测垃圾车"
run_case "smoke" "examples/images/smoke.jpg" "检测烟雾"
run_case "fishing_person" "examples/images/fisherman.jpg" "检测钓鱼的人"
COMMENT

run_case "person_with_bag" "examples/images/person_with_bag.png" "检测背包的行人"

echo "=================================================="
echo "Eval-only run finished. Outputs are under: ${OUTPUT_ROOT}"
