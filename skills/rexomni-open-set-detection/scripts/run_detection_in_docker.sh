#!/bin/bash
# Run open-set detection inside Docker (cu118 or cu124 for 4090). Ensures RexOmni service is up, then runs run_detection.py in container.
# Usage: run_detection_in_docker.sh [--start-service] --images PATH --prompt JSON --out COCO.json
# Env: REXOMNI_BASE_URL (default http://127.0.0.1:9011/v1), REXOMNI_MODEL_PATH, IMAGE (docker image override)

set -e
SKILL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ROOT_PROJECT="$(cd "$SKILL_DIR/../.." && pwd)"
ROOT_DIR="$ROOT_PROJECT"
source "$ROOT_PROJECT/scripts/path_utils.sh"
REXOMNI_BASE_URL="${REXOMNI_BASE_URL:-http://127.0.0.1:9011/v1}"
REXOMNI_MODEL_PATH="$(resolve_model_dir "${REXOMNI_MODEL_PATH:-RexOmni}")"
export REXOMNI_MODEL_PATH

START_SERVICE=""
IMAGES=""
PROMPT=""
OUT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --start-service) START_SERVICE=1 ;;
        --images)  IMAGES="$2"; shift ;;
        --prompt)  PROMPT="$2"; shift ;;
        --out)     OUT="$2"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$IMAGES" ] || [ -z "$PROMPT" ] || [ -z "$OUT" ]; then
    echo "Usage: $0 [--start-service] --images IMG_OR_DIR --prompt PROMPT.json --out COCO.json"
    exit 1
fi

# Optional: ensure RexOmni is running
if [ -n "$START_SERVICE" ]; then
    bash "$SKILL_DIR/scripts/start_rexomni_service.sh"
fi

# Check service
check_service() { curl -sf --connect-timeout 2 "${REXOMNI_BASE_URL%/v1}/v1/models" >/dev/null 2>&1; }
if ! check_service; then
    echo "Error: RexOmni not reachable at $REXOMNI_BASE_URL. Start with: bash $SKILL_DIR/scripts/start_rexomni_service.sh"
    exit 1
fi

# Detect GPU for image
if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -qi "4090"; then
        DOCKER_IMAGE="${DOCKER_IMAGE:-agent_skill_base_image_cu124:v0.0.1}"
    else
        DOCKER_IMAGE="${DOCKER_IMAGE:-agent_skill_base_image_cu118:v0.0.1}"
    fi
else
    DOCKER_IMAGE="${DOCKER_IMAGE:-agent_skill_base_image_cu118:v0.0.1}"
fi

IMAGES_ABS="$(cd "$(dirname "$IMAGES")" 2>/dev/null && pwd)/$(basename "$IMAGES")" || IMAGES_ABS="$(pwd)/$IMAGES"
PROMPT_ABS="$(cd "$(dirname "$PROMPT")" 2>/dev/null && pwd)/$(basename "$PROMPT")" || PROMPT_ABS="$(pwd)/$PROMPT"
OUT_ABS="$(cd "$(dirname "$OUT")" 2>/dev/null && pwd)/$(basename "$OUT")" || OUT_ABS="$(pwd)/$OUT"

MNT_IMAGES="$(dirname "$IMAGES_ABS")"
MNT_PROMPT="$(dirname "$PROMPT_ABS")"
MNT_OUT="$(dirname "$OUT_ABS")"

docker run --rm --gpus=all --network=host \
  -v "$SKILL_DIR:/workspace/skill" \
  -v "$MNT_IMAGES:/workspace/images_host" \
  -v "$MNT_PROMPT:/workspace/prompt_host" \
  -v "$MNT_OUT:/workspace/out_host" \
  -e REXOMNI_BASE_URL="$REXOMNI_BASE_URL" \
  -w /workspace/skill \
  "$DOCKER_IMAGE" \
  python scripts/run_detection.py \
    --images "/workspace/images_host/$(basename "$IMAGES_ABS")" \
    --prompt "/workspace/prompt_host/$(basename "$PROMPT_ABS")" \
    --base-url "$REXOMNI_BASE_URL" \
    --out "/workspace/out_host/$(basename "$OUT_ABS")"

echo "Done. COCO format result: $OUT_ABS"
