#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/path_utils.sh"

MODEL_DIR="$(resolve_model_dir "${MODEL_DIR:-/raid/zkq/models/Qwen3.5-4B}")"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-4b}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8010}"
GATEWAY_HOST="${GATEWAY_HOST:-0.0.0.0}"
GATEWAY_PORT="${GATEWAY_PORT:-8002}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
DEFAULT_TEMPERATURE="${DEFAULT_TEMPERATURE:-0}"
DEFAULT_TOP_P="${DEFAULT_TOP_P:-1}"
DEFAULT_SEED="${DEFAULT_SEED:-42}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-$ROOT_DIR/demo/chat_templates/qwen_thinking_off_full.jinja}"
MODEL_IMPL="${MODEL_IMPL:-transformers}"
HF_OVERRIDES="${HF_OVERRIDES:-{\"architectures\":[\"TransformersForCausalLM\"]}}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
LORA_MODULES="${LORA_MODULES:-}"
MAX_LORA_RANK="${MAX_LORA_RANK:-16}"

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$ROOT_DIR/demo/vllm_compat${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[$(date '+%F %T')] starting vLLM OpenAI server"
echo "model_dir=$MODEL_DIR"
echo "served_model_name=$SERVED_MODEL_NAME"
echo "vllm=${VLLM_HOST}:${VLLM_PORT}"
echo "gateway=${GATEWAY_HOST}:${GATEWAY_PORT}"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "default_temperature=$DEFAULT_TEMPERATURE"
echo "default_top_p=$DEFAULT_TOP_P"
echo "default_seed=$DEFAULT_SEED"
if [[ -n "$LORA_MODULES" ]]; then
  echo "lora_modules=$LORA_MODULES"
fi

VLLM_ARGS=(
  -m vllm.entrypoints.openai.api_server
  --model "$MODEL_DIR" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --hf-overrides "$HF_OVERRIDES" \
  --host "$VLLM_HOST" \
  --port "$VLLM_PORT" \
  --dtype float16 \
  --model-impl "$MODEL_IMPL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --trust-remote-code \
  --chat-template "$CHAT_TEMPLATE" \
  --chat-template-content-format string
)
if [[ "$ENFORCE_EAGER" == "1" || "$ENFORCE_EAGER" == "true" ]]; then
  VLLM_ARGS+=(--enforce-eager)
fi
if [[ -n "$LORA_MODULES" ]]; then
  VLLM_ARGS+=(--enable-lora --max-lora-rank "$MAX_LORA_RANK")
  # shellcheck disable=SC2206
  LORA_MODULE_ARRAY=($LORA_MODULES)
  VLLM_ARGS+=(--lora-modules "${LORA_MODULE_ARRAY[@]}")
fi

.venv-train/bin/python "${VLLM_ARGS[@]}" &
VLLM_PID="$!"

for _ in $(seq 1 180); do
  if curl --noproxy '*' -sf "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" >/dev/null; then
    break
  fi
  sleep 2
done

if ! curl --noproxy '*' -sf "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" >/dev/null; then
  echo "vLLM did not become ready" >&2
  exit 1
fi

echo "[$(date '+%F %T')] starting AI Model Gateway 0.1.0"
exec .venv-train/bin/python demo/ai_model_gateway.py \
  --upstream-base-url "http://${VLLM_HOST}:${VLLM_PORT}" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$GATEWAY_HOST" \
  --port "$GATEWAY_PORT" \
  --default-temperature "$DEFAULT_TEMPERATURE" \
  --default-top-p "$DEFAULT_TOP_P" \
  --default-seed "$DEFAULT_SEED"
