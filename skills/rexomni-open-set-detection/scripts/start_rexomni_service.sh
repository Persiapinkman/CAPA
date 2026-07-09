#!/bin/bash
# Check RexOmni service; if not running, start via Docker (cu118 or cu124 for 4090).
# Model path: REXOMNI_MODEL_PATH or default /raid/zkq/models/RexOmni
# Port: REXOMNI_PORT (default 9011)

set -e
SKILL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ROOT_DIR="$(cd "$SKILL_DIR/../.." && pwd)"
source "$ROOT_DIR/scripts/path_utils.sh"

REXOMNI_PORT="${REXOMNI_PORT:-9011}"
REXOMNI_BASE_URL="${REXOMNI_BASE_URL:-http://127.0.0.1:9011/v1}"
REXOMNI_MODEL_PATH="$(resolve_model_dir "${REXOMNI_MODEL_PATH:-RexOmni}")"
# REXOMNI_CUDA_VISIBLE_DEVICES="${REXOMNI_CUDA_VISIBLE_DEVICES:-2}"
# GPU check and device selection: pick at least 2 idle GPUs (lowest memory.used) when VLM_DEVICES not set
if [ -z "$VLM_DEVICES" ] && command -v nvidia-smi >/dev/null 2>&1; then
    # Get index,memory_used (strip " MiB"), sort by memory, take first 2 indices
    IDLE_GPUS="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | \
        while IFS= read -r line; do
            idx="$(echo "$line" | cut -d',' -f1 | tr -d ' ')"
            mem="$(echo "$line" | cut -d',' -f2 | tr -d ' MiB')"
            printf "%s %s\n" "$mem" "$idx"
        done | sort -n | head -2 | awk '{print $2}' | paste -sd',' -)"
    if [ -n "$IDLE_GPUS" ]; then
        VLM_DEVICES="$IDLE_GPUS"
        echo "Auto-selected idle GPUs (lowest memory): VLM_DEVICES=$VLM_DEVICES"
    fi
fi
REXOMNI_CUDA_VISIBLE_DEVICES="${REXOMNI_CUDA_VISIBLE_DEVICES:-0,1}"


# Check if RexOmni is already up
check_service() {
    curl -sf --connect-timeout 2 "${REXOMNI_BASE_URL%/v1}/v1/models" >/dev/null 2>&1
}

if check_service; then
    echo "RexOmni service already running at ${REXOMNI_BASE_URL}"
    exit 0
fi

# Detect GPU: 4090 -> cu124, else cu118
if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -qi "4090"; then
        DOCKER_IMAGE="agent_skill_base_image_cu124:v0.0.1"
    else
        DOCKER_IMAGE="agent_skill_base_image_cu118:v0.0.1"
    fi
else
    DOCKER_IMAGE="agent_skill_base_image_cu118:v0.0.1"
fi

if [ ! -d "$REXOMNI_MODEL_PATH" ]; then
    echo "Error: RexOmni model path not found: $REXOMNI_MODEL_PATH"
    exit 1
fi

echo "Starting RexOmni with image $DOCKER_IMAGE on port $REXOMNI_PORT"
# Mount model and skill scripts; run vllm serve inside container
docker run -d --rm --shm-size=32g --gpus=all --network=host --name rexomni-service \
  -v "$REXOMNI_MODEL_PATH:/workspace/models/RexOmni" \
  -v "$SKILL_DIR/scripts/run_rexomni.sh:/workspace/run_rexomni.sh" \
  -w /workspace \
  "$DOCKER_IMAGE" \
  bash -c "chmod +x /workspace/run_rexomni.sh && /workspace/run_rexomni.sh $REXOMNI_PORT /workspace/models/RexOmni /workspace/models/RexOmni $REXOMNI_CUDA_VISIBLE_DEVICES"

# Wait and verify
for i in $(seq 1 60); do
    sleep 2
    if check_service; then
        echo "RexOmni service is up at ${REXOMNI_BASE_URL}"
        exit 0
    fi
    echo "Waiting for RexOmni... ($i/60)"
done
echo "Warning: RexOmni may still be starting. Check with: curl ${REXOMNI_BASE_URL%/v1}/v1/models"
exit 0
