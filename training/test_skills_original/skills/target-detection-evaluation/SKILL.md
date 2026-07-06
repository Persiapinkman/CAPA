---
name: target-detection-evaluation
description: End-to-end target detection evaluation — intent, prompt expansion, Flux image generation, Qwen + RexOmni open-set detection, merged prediction, bbox overlays, and LLM evaluation report. Equivalent to examples/run_eval_pipeline_example.py.
---

# Target Detection Evaluation

## Overview

Runs the full evaluation pipeline in one command:

1. **user-intent-understanding** → `intent.json`
2. **llm-prompts-generation** → `prompts.json`
3. **flux-image-generation** → `generated_images/` + `generated_images.json`
4. **qwen-vlm-open-set-delection** + **rexomni-open-set-detection** → `prediction.json`
5. **annotated_images/** — Qwen boxes in red, Rex-Omni in blue
6. **eval-reports-generation** → `evaluation.json`

The orchestrator automatically `chdir`s to the repository root so sibling skills under `skills/` resolve correctly.

## Run

From the **repository root** (`test-skills/`):

**Split mode (default):** LLM steps use Qwen3.5-4B on `http://180.184.148.142:1032/v1` with the built-in default key; Flux still uses its own OpenAI-compatible image API (default `https://api.apiyi.com/v1`). You must pass `--api-key` for the Flux service.

```bash
python3 skills/target-detection-evaluation/scripts/run_pipeline.py \
  --workspace /path/to/output_dir \
  --image /path/to/reference.jpg \
  --text "检测目标描述" \
  --api-key "$APIYI_API_KEY" \
  --num-images 5 \
  --qwen-base-url http://127.0.0.1:9012/v1 \
  --rex-base-url http://127.0.0.1:23335/v1
```

**Legacy unified mode:** same `--api-base` and `--api-key` for LLM + Flux (e.g. one apiyi deployment).

```bash
python3 skills/target-detection-evaluation/scripts/run_pipeline.py \
  --workspace /path/to/output_dir \
  --image /path/to/reference.jpg \
  --text "检测目标描述" \
  --api-base https://api.apiyi.com/v1 \
  --api-key "$APIYI_API_KEY" \
  --num-images 5
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--workspace` | Output directory for all artefacts |
| `--image` | Reference input image |
| `--text` | User task (same as pipeline `--text`) |
| `--api-base` / `--api-key` | Optional legacy pair: if `--api-base` is set, both values apply to **all** LLM + Flux steps |
| `--llm-api-base` / `--llm-api-key` | Split mode only: intent / prompts / eval (defaults: Qwen endpoint + key) |
| `--flux-api-base` | Split mode: Flux image API base URL (default `https://api.apiyi.com/v1`) |
| `--api-key` | Split mode: **Flux** API key only. Required unless using legacy `--api-base` |
| `--num-images` | How many Flux generations (default `5`) |
| `--qwen-base-url` | Qwen VLM detection service |
| `--rex-base-url` | RexOmni detection service |

### Outputs (under `--workspace`)

- `intent.json`, `prompts.json`, `generated_images.json`
- `generated_images/`
- `prediction.json`
- `annotated_images/`
- `evaluation.json`

## Reference

Behaviour aligns with [examples/run_eval_pipeline_example.py](../../examples/run_eval_pipeline_example.py).
