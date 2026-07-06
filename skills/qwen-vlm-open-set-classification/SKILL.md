---
name: qwen-vlm-open-set-classification
description: Runs open-set image classification with Qwen VLM. Checks/starts Qwen VLM vLLM service (Docker cu118/cu124 by GPU), normalizes image path and prompt JSON input, calls Qwen VLM API, outputs MS COCO format. Use when classifying images into open-set categories, running Qwen VLM inference, or converting prompts and images to COCO classification results.
---

# Qwen VLM Open-Set Classification

## Overview

- **Input**: Image path or image folder path + prompt JSON (prompts + index2cls).
- **Output**: Classification result in MS COCO format (images, categories, annotations with category_id).
- **Runtime**: Qwen VLM service in Docker (agent_skill_base_image_cu118 or cu124 for 4090); classification runs inside the same image.

## Workflow

1. **Ensure Qwen VLM service**: Check if Qwen VLM is up; if not, start via Docker using [scripts/start_qwen_vlm_service.sh](scripts/start_qwen_vlm_service.sh).
2. **Normalize input**: Resolve image path → single file or list of files; load and validate prompt JSON (prompts, index2cls); build class prompts and class_prompt→class_name mapping.
3. **Run classification**: Execute [scripts/run_classification.py](scripts/run_classification.py) inside the same Docker image; script calls Qwen VLM API and outputs COCO format.

## 1. Check and Start Qwen VLM Service

**Check**: `curl -s http://127.0.0.1:9012/v1/models` (or env `VLM_BASE_URL`). If no response, start service.

**Check GPU**: use `nvidia-smi` to check the Idle GPU Device. And set the env var `VLM_DEVICES`. VLM need 2 GPUs at less.

**Start** (run from host):

```bash
bash skills/qwen-vlm-open-set-classification/scripts/start_qwen_vlm_service.sh
```

- Detects GPU: 4090 → image `agent_skill_base_image_cu124:v0.0.1`, else `agent_skill_base_image_cu118:v0.0.1`.
- Uses model path: `/media/nvme1n1p1/models/Qwen2.5-VL-7B-Instruct` (overridable by env `VLM_MODEL_PATH`).
- Starts container that runs [scripts/run_qwen_vlm.sh](scripts/run_qwen_vlm.sh) (vllm serve) on port 9012.

**Verify**: After start, run again `curl -s http://127.0.0.1:9012/v1/models` or use script’s built-in check.

## 2. Prompt JSON Format (Input Normalization)

Prompt file must be valid JSON with:

- **`prompts`**: Object mapping class key → list of prompt tokens (e.g. `"轿车": ["轿车", "汽车"]`).
- **`index2cls`**: Object mapping category index string → class key (e.g. `"1": "非机动车"`).

Example: see [references/prompt.json](references/prompt.json).

**Normalization steps** (handled by run_classification.py):

- **Images**: If path is a directory → collect image files (.jpg, .jpeg, .png, .bmp, .webp) as list; if path is a file → single-element list.
- **Prompt JSON**: Load and validate; require both `prompts` and `index2cls`. Build `class_prompts` (flat list of all label strings) and `class_prompt2classname` (label string → class key) from `prompts`; use `index2cls` for category id and name in COCO output.

## 3. Run Classification (Inside Docker)

Classification must run **inside** the same Docker image (cu118 or cu124). From host use the wrapper (recommended):

```bash
# Optional: start Qwen VLM if not running
bash skills/qwen-vlm-open-set-classification/scripts/run_classification_in_docker.sh --start-service \
  --images /path/to/image_or_folder \
  --prompt /path/to/prompt.json \
  --out /path/to/coco_output.json
```

Or without auto-start (service must already be up):

```bash
bash skills/qwen-vlm-open-set-classification/scripts/run_classification_in_docker.sh \
  --images /path/to/image_or_folder \
  --prompt /path/to/prompt.json \
  --out /path/to/coco_output.json
```

The wrapper picks cu118/cu124 by GPU, mounts the skill and paths, and runs [scripts/run_classification.py](scripts/run_classification.py) inside the container.

**run_classification.py**:

- **Args**: `--images` (file or dir), `--prompt` (JSON path), `--base-url`, `--out` (COCO JSON path).
- **Behavior**: Normalize images to list; load prompt JSON; build system/user prompts from prompts+index2cls; for each image, resize and base64 encode, call Qwen VLM chat/completions with JSON schema (desc, label); map response label to class name and category_id; write COCO (images, categories, annotations with image_id and category_id, no bbox).

## 4. Output Format (MS COCO)

- **images**: `[{ "id", "file_name", "width", "height" }]`
- **categories**: `[{ "id", "name" }]` from prompt `index2cls`
- **annotations**: `[{ "id", "image_id", "category_id" }]` (image-level classification, no bbox)

## Reference Files

- [references/prompt.json](references/prompt.json) — example prompt JSON (prompts + index2cls).
- [scripts/run_qwen_vlm.sh](scripts/run_qwen_vlm.sh) — vLLM serve command used inside the Qwen VLM container.

## Quick Checklist

- [ ] Qwen VLM service running (start_qwen_vlm_service.sh if not).
- [ ] Prompt JSON has `prompts` and `index2cls`; images path is file or folder.
- [ ] Run classification inside Docker (cu118 or cu124) with run_classification.py; output in COCO format.
