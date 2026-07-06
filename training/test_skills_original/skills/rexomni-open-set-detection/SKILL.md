---
name: rexomni-open-set-detection
description: Runs open-set object detection with RexOmni. Checks/starts RexOmni vLLM service (Docker cu118/cu124 by GPU), normalizes image and prompt JSON input, calls RexOmni API, outputs MS COCO format. Use when detecting open-set objects in images, running RexOmni inference, or converting prompts and images to COCO detection results.
---

# RexOmni Open-Set Detection

## Overview

- **Input**: Image path or image folder path + prompt JSON (class prompts + index2cls).
- **Output**: Detection result in MS COCO format (images, categories, annotations with bbox).
- **Runtime**: RexOmni service in Docker (agent_skill_base_image_cu118 or cu124 for 4090); detection runs inside the same image.

## Workflow

1. **Ensure RexOmni service**: Check if RexOmni is up; if not, start via Docker using [scripts/start_rexomni_service.sh](scripts/start_rexomni_service.sh).
2. **Normalize input**: Resolve image path → single file or list of files; load and validate prompt JSON (prompts, index2cls).
3. **Run detection**: Execute [scripts/run_detection.py](scripts/run_detection.py) inside the same Docker image; script calls RexOmni API and uses [references/parser.py](references/parser.py) `parse_prediction` to convert raw output to COCO.

## 1. Check and Start RexOmni Service

**Check**: `curl -s http://127.0.0.1:9011/v1/models` (or env `REXOMNI_BASE_URL`). If no response, start service.

**Check GPU**: use `nvidia-smi` to check the Idle GPU Device. And set the env var `REXOMNI_CUDA_VISIBLE_DEVICES`. RexOmni need 1 GPU at less.

**Start** (run from host):

```bash
bash skills/rexomni-open-set-detection/scripts/start_rexomni_service.sh
```

- Detects GPU: 4090 → image `agent_skill_base_image_cu124:v0.0.0`, else `agent_skill_base_image_cu118:v0.0.0`.
- Uses model path: `/media/nvme1n1p1/zhuangzhenzhou/inferpipeline/model/models/RexOmni` (overridable by env).
- Starts container that runs [scripts/run_rexomni.sh](scripts/run_rexomni.sh) (vllm serve) on port 9011.

**Verify**: After start, run again `curl -s http://127.0.0.1:9011/v1/models` or use script’s built-in check.

## 2. Prompt JSON Format (Input Normalization)

Prompt file must be valid JSON with:

- **`prompts`**: Object mapping class key → list of prompt tokens (e.g. `"banner_slogan": ["banner", "slogan"]`).
- **`index2cls`**: Object mapping category index string → class key (e.g. `"1": "banner_slogan"`).

Example: see [references/prompt_example.json](references/prompt_example.json).

**Normalization steps** (handled by run_detection.py):

- **Images**: If path is a directory → collect image files (.jpg, .jpeg, .png, .bmp, .webp) as list; if path is a file → single-element list.
- **Prompt JSON**: Load and validate; require both `prompts` (class → list of prompt tokens) and `index2cls` (index string → class key). Build detection text from `index2cls` order and `prompts[cls]`, e.g. "Detect A,B. Output the bounding box coordinates in [x0, y0, x1, y1] format."

## 3. Run Detection (Inside Docker)

Detection must run **inside** the same Docker image (cu118 or cu124). From host use the wrapper (recommended):

```bash
# Optional: start RexOmni if not running
bash skills/rexomni-open-set-detection/scripts/run_detection_in_docker.sh --start-service \
  --images /path/to/image_or_folder \
  --prompt /path/to/prompt.json \
  --out /path/to/coco_output.json
```

Or without auto-start (service must already be up):

```bash
bash skills/rexomni-open-set-detection/scripts/run_detection_in_docker.sh \
  --images /path/to/image_or_folder \
  --prompt /path/to/prompt.json \
  --out /path/to/coco_output.json
```

The wrapper picks cu118/cu124 by GPU, mounts the skill and paths, and runs [scripts/run_detection.py](scripts/run_detection.py) inside the container.

**run_detection.py**:

- **Args**: `--images` (file or dir), `--prompt` (JSON path), `--base-url`, `--out` (COCO JSON path).
- **Behavior**: Normalize images to list; load prompt JSON; for each image, build request (base64 image + text from prompts/index2cls), call RexOmni chat/completions; parse response with `parse_prediction(image_w, image_h)` from [references/parser.py](references/parser.py); convert to COCO (images, categories, annotations with bbox [x, y, width, height]); write `--out`.

## 4. Output Format (MS COCO)

- **images**: `[{ "id", "file_name", "width", "height" }]`
- **categories**: `[{ "id", "name" }]` from prompt `index2cls`
- **annotations**: `[{ "id", "image_id", "category_id", "bbox": [x, y, width, height], "area" }]`

RexOmni raw output is parsed by `parse_prediction` in [references/parser.py](references/parser.py); boxes `[x0,y0,x1,y1]` are converted to COCO `[x, y, width, height]`.

## Reference Files

- [references/parser.py](references/parser.py) — `parse_prediction(text, w, h, task_type="detection")` for RexOmni output.
- [references/prompt_example.json](references/prompt_example.json) — example prompt JSON (prompts + index2cls).
- [scripts/run_rexomni.sh](scripts/run_rexomni.sh) — vLLM serve command used inside the RexOmni container.

## Quick Checklist

- [ ] RexOmni service running (start_rexomni_service.sh if not).
- [ ] Prompt JSON has `prompts` and `index2cls`; images path is file or folder.
- [ ] Run detection inside Docker (cu118 or cu124) with run_detection.py; output in COCO format.
