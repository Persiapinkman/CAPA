#!/bin/bash
# Check Qwen VLM service; if not running, start via Docker (cu118 or cu124 for 4090).
# Model path: VLM_MODEL_PATH or default /media/nvme1n1p1/models/Qwen2.5-VL-7B-Instruct
# Port: VLM_PORT (default 9012)

set -e
VLM_PORT="${VLM_PORT:-9012}"
VLM_BASE_URL="${VLM_BASE_URL:-http://127.0.0.1:9012/v1}"
VLM_MODEL_PATH="${VLM_MODEL_PATH:-/media/nvme1n1p1/models/Qwen2.5-VL-7B-Instruct}"
# VLM_DEVICES="${VLM_DEVICES:-0,1}"
SKILL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
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
VLM_DEVICES="${VLM_DEVICES:-0,1}"
echo "Using VLM_DEVICES=$VLM_DEVICES"

# Check if Qwen VLM is already up
check_service() {
    curl -sf --connect-timeout 2 "${VLM_BASE_URL%/v1}/v1/models" >/dev/null 2>&1
}

if check_service; then
    echo "Qwen VLM service already running at ${VLM_BASE_URL}"
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

if [ ! -d "$VLM_MODEL_PATH" ]; then
    echo "Error: Qwen VLM model path not found: $VLM_MODEL_PATH"
    exit 1
fi

echo "Starting Qwen VLM with image $DOCKER_IMAGE on port $VLM_PORT"
docker run -d --rm --shm-size=32g --gpus=all --network=host --name qwen-vlm-service \
  -v "$VLM_MODEL_PATH:/workspace/models/Qwen2.5-VL-7B-Instruct" \
  -v "$SKILL_DIR/scripts/run_qwen_vlm.sh:/workspace/run_qwen_vlm.sh" \
  -e VLM_MODEL_PATH=/workspace/models/Qwen2.5-VL-7B-Instruct \
  -e VLM_PORT="$VLM_PORT" \
  -e VLM_DEVICES="$VLM_DEVICES" \
  -w /workspace \
  "$DOCKER_IMAGE" \
  bash -c "chmod +x /workspace/run_qwen_vlm.sh && /workspace/run_qwen_vlm.sh"

# Wait and verify
for i in $(seq 1 120); do
    sleep 2
    if check_service; then
        echo "Qwen VLM service is up at ${VLM_BASE_URL}"
        exit 0
    fi
    echo "Waiting for Qwen VLM... ($i/120)"
done
echo "Warning: Qwen VLM may still be starting. Check with: curl ${VLM_BASE_URL%/v1}/v1/models"
exit 0
