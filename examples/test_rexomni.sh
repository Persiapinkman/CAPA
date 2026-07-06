#!/usr/bin/env bash
set -euo pipefail

# Rex-Omni open-set detection on a single image (label extraction + detect + draw boxes).
#
# Usage:
#   bash examples/test_rexomni.sh [output_dir] [image_path] [rex_base_url] [llm_api_base] [llm_api_key]
#
# Defaults:
#   output_dir:    results/rexomni_helmet
#   image_path:    data/images.jpeg
#   annotated image is also copied to data/<stem>_rex_annotated<suffix>
#   rex_base_url:  DEMO_REX_BASE_URL or http://10.111.32.253:8000/v1
#   llm_api_base:  DEMO_MIGRATION_ADVISOR_API_BASE or same as rex_base_url
#   llm_api_key:   APIYI_API_KEY env; fallback to repo api_key.txt
#
# Override detection target text:
#   REX_TEST_LABEL="头盔" bash examples/test_rexomni.sh
#
# Override user query for label extraction:
#   REX_TEST_QUERY="检测图片中的头盔" bash examples/test_rexomni.sh

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  echo "Usage:"
  echo "  bash examples/test_rexomni.sh [output_dir] [image_path] [rex_base_url] [llm_api_base] [llm_api_key]"
  echo ""
  echo "Defaults:"
  echo "  output_dir:   results/rexomni_helmet"
  echo "  image_path:   data/images.jpeg"
  echo "  rex_base_url: DEMO_REX_BASE_URL or http://10.111.32.253:8000/v1"
  echo ""
  echo "Environment:"
  echo "  REX_TEST_LABEL   Detection target (default: 头盔)"
  echo "  REX_TEST_QUERY   Full text for LLM label extraction (default: 检测<REX_TEST_LABEL>)"
  exit 0
fi

OUT_DIR="${1:-results/rexomni_helmet}"
IMAGE_REL="${2:-data/images.jpeg}"
REX_BASE_URL="${3:-${DEMO_REX_BASE_URL:-http://10.111.32.253:8000/v1}}"
LLM_API_BASE="${4:-${DEMO_MIGRATION_ADVISOR_API_BASE:-${DEMO_ANSWER_API_BASE:-${DEMO_LLM_API_BASE:-$REX_BASE_URL}}}}"
LLM_API_KEY="${5:-${APIYI_API_KEY:-${DEMO_MIGRATION_ADVISOR_API_KEY:-}}}"

REX_TEST_LABEL="${REX_TEST_LABEL:-头盔}"
REX_TEST_QUERY="${REX_TEST_QUERY:-请检测图片中的${REX_TEST_LABEL}（helmet、安全帽、hard hat）}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -z "${LLM_API_KEY}" ] && [ -f "${ROOT}/api_key.txt" ]; then
  LLM_API_KEY="$(head -n 1 "${ROOT}/api_key.txt" | tr -d '\r\n')"
fi

if [ -z "${LLM_API_KEY}" ]; then
  echo "Error: LLM API key is empty (needed for label extraction)."
  echo "Set APIYI_API_KEY or pass llm_api_key as argument #5."
  exit 1
fi

cd "${ROOT}"

IMAGE_ABS="${ROOT}/${IMAGE_REL}"
if [[ "${IMAGE_REL}" = /* ]]; then
  IMAGE_ABS="${IMAGE_REL}"
fi
if [ ! -f "${IMAGE_ABS}" ]; then
  echo "Error: image not found: ${IMAGE_ABS}"
  exit 1
fi

mkdir -p "${OUT_DIR}"
DATA_DIR="${ROOT}/data"
mkdir -p "${DATA_DIR}"

IMAGE_STEM="$(basename "${IMAGE_ABS%.*}")"
IMAGE_SUFFIX=".${IMAGE_ABS##*.}"
ANNOTATED_DATA="${DATA_DIR}/${IMAGE_STEM}_rex_annotated${IMAGE_SUFFIX}"

echo "Skill:        rexomni-open-set-detection"
echo "Image:        ${IMAGE_ABS}"
echo "Target label: ${REX_TEST_LABEL}"
echo "Query:        ${REX_TEST_QUERY}"
echo "Rex API:      ${REX_BASE_URL}"
echo "LLM API:      ${LLM_API_BASE}"
echo "Out:          ${OUT_DIR}"
echo "Annotated:    ${ANNOTATED_DATA}"
echo "--------------------------------------------------"

export ROOT OUT_DIR IMAGE_ABS REX_BASE_URL LLM_API_BASE LLM_API_KEY REX_TEST_LABEL REX_TEST_QUERY ANNOTATED_DATA

python3 <<'PY'
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ["ROOT"])
OUT_DIR = Path(os.environ["OUT_DIR"])
IMAGE_ABS = Path(os.environ["IMAGE_ABS"])
REX_BASE = os.environ["REX_BASE_URL"].rstrip("/")
LLM_BASE = os.environ["LLM_API_BASE"].rstrip("/")
LLM_KEY = os.environ["LLM_API_KEY"]
LABEL = os.environ["REX_TEST_LABEL"]
QUERY = os.environ["REX_TEST_QUERY"]

sys.path.insert(0, str(ROOT))
from util.rex_label_extraction import extract_rex_detection_labels

spec = importlib.util.spec_from_file_location(
    "te_pipeline",
    ROOT / "skills/target-detection-evaluation/scripts/run_pipeline.py",
)
ph = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ph)

print("Step 1: LLM label extraction …")
detection = extract_rex_detection_labels(
    QUERY,
    api_key=LLM_KEY,
    base_url=LLM_BASE,
)
print(json.dumps(detection, ensure_ascii=False, indent=2))

prompt_path = OUT_DIR / "rex_prompt.json"
coco_out = OUT_DIR / "rex_detect_coco.json"
ph.write_rex_prompt_json(prompt_path, detection)
print(f"Wrote prompt: {prompt_path}")

print("Step 2: Rex-Omni detection …")
cmd = [
    sys.executable,
    str(ROOT / "skills/rexomni-open-set-detection/scripts/run_detection.py"),
    "--images",
    str(IMAGE_ABS),
    "--prompt",
    str(prompt_path),
    "--base-url",
    REX_BASE,
    "--api-key",
    LLM_KEY,
    "--out",
    str(coco_out),
]
print("RUN:", " ".join(cmd))
subprocess.run(cmd, check=True)

with open(coco_out, "r", encoding="utf-8") as f:
    coco = json.load(f)
ann_count = len(coco.get("annotations") or [])
print(f"Detections: {ann_count} box(es)")

by_id = ph.coco_to_pred_bboxes_by_image_id(coco)
boxes = by_id.get(0, [])
pred_path = OUT_DIR / "prediction.json"
with open(pred_path, "w", encoding="utf-8") as f:
    json.dump(
        [
            {
                "image": IMAGE_ABS.name,
                "source": "original",
                "image_idx": 0,
                "models": [{"model": "rex-omni", "pred_bboxes": boxes}],
            }
        ],
        f,
        indent=2,
        ensure_ascii=False,
    )

print("Step 3: Draw bounding boxes …")
viz_dir = OUT_DIR / "annotated_images"
image_map = {IMAGE_ABS.name: IMAGE_ABS.resolve()}
ph.draw_detection_overlays(image_map, pred_path, viz_dir)
annotated_src = viz_dir / IMAGE_ABS.name
if not annotated_src.is_file():
    candidates = [p for p in viz_dir.glob("*") if p.is_file()]
    annotated_src = candidates[0] if candidates else annotated_src

annotated_data = Path(os.environ["ANNOTATED_DATA"])
annotated_data.parent.mkdir(parents=True, exist_ok=True)
if annotated_src.is_file():
    shutil.copy2(annotated_src, annotated_data)
    print(f"Saved annotated image: {annotated_data}")
else:
    print("Warning: annotated image not found after draw step")

for p in sorted(viz_dir.glob("*")):
    if p.is_file():
        print(f"Annotated (work): {p}")

summary = {
    "image": str(IMAGE_ABS),
    "query": QUERY,
    "target_label": LABEL,
    "detection_targets": detection,
    "num_boxes": len(boxes),
    "coco_out": str(coco_out),
    "annotated_dir": str(viz_dir),
    "annotated_image": str(annotated_data) if annotated_data.is_file() else "",
}
(OUT_DIR / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "Done."
echo "  Work dir:        ${OUT_DIR}"
echo "  Annotated image: ${ANNOTATED_DATA}"
