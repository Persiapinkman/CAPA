# Evaluation Policy

This file defines the required protocol for Planner routing evals after 2026-07-06.

## Serving

- Use vLLM OpenAI-compatible serving for all formal evals.
- Put `AI Model Gateway 0.1.0` in front of vLLM when the client may send OpenAI multimodal content lists.
- Do not report new formal results from the lightweight `demo/deploy_qwen_server.py` transformers server.

Recommended serving entry:

```bash
MODEL_DIR=/mnt/zkq/models/Qwen3.5-4B-vllm \
SERVED_MODEL_NAME=qwen3.5-4b \
CUDA_VISIBLE_DEVICES=3 \
GATEWAY_PORT=8003 \
VLLM_PORT=8013 \
DEFAULT_TEMPERATURE=0 \
DEFAULT_TOP_P=1 \
DEFAULT_SEED=42 \
bash scripts/run_qwen35_4b_vllm_gateway.sh
```

LoRA adapter evals must also use vLLM LoRA serving, for example:

```bash
MODEL_DIR=/mnt/zkq/models/Qwen3.5-4B-vllm \
SERVED_MODEL_NAME=qwen35-4b-newarch-dpo \
LORA_MODULES=qwen35-4b-newarch-dpo=outputs/planner-dpo-qwen35-4b-newarch-lora \
CUDA_VISIBLE_DEVICES=3 \
GATEWAY_PORT=8003 \
VLLM_PORT=8013 \
DEFAULT_TEMPERATURE=0 \
DEFAULT_TOP_P=1 \
DEFAULT_SEED=42 \
bash scripts/run_qwen35_4b_vllm_gateway.sh
```

## Generation

Formal evals must use deterministic generation:

- `temperature=0`
- `top_p=1`
- `do_sample=false`
- `seed=42`

`do_sample` is not an OpenAI standard field. The local gateway accepts `do_sample=false`, enforces greedy decoding via `temperature=0`, then removes `do_sample` before forwarding to vLLM.

## Repeats

Every formal eval must run 3 repeats and report the aggregate mean/stdev.

Recommended eval entry:

```bash
MODEL=qwen3.5-4b \
API_BASE=http://127.0.0.1:8003/v1 \
REPORT_PREFIX=qwen35_4b_vllm_base_zip90 \
bash scripts/run_vllm_repro_eval_3x.sh
```

The aggregate report is written to:

```text
results/planner_routing_eval/<REPORT_PREFIX>_aggregate.json
```

Individual repeat reports are written to:

```text
results/planner_routing_eval/<REPORT_PREFIX>_run1.json
results/planner_routing_eval/<REPORT_PREFIX>_run2.json
results/planner_routing_eval/<REPORT_PREFIX>_run3.json
```

## Timing

`run_planner_routing_eval.py` schema `1.1` records:

- per case: `started_at`, `finished_at`, `elapsed_ms`
- summary: total elapsed time, case total/average/min/max elapsed time

Older imported reports have `elapsed_ms=null` and `summary.timing.source=historical_import_no_eval_timing`.
