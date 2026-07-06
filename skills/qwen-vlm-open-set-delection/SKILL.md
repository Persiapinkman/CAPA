---
name: qwen-vlm-open-set-delection
description: Runs open-set bbox detection with Qwen VLM. Inputs image file/folder and label text, calls Qwen VLM with JSON schema, and outputs per-image bbox results. Use when you need prompt-driven object detection from Qwen VLM.
---

# Qwen VLM Open-Set Delection

## Overview

- **Input**: image path (file or folder) + label text.
- **Output**: JSON file with `results` list; each item contains `image`, `label`, `bboxes`.
- **Runtime**: local/remote Qwen OpenAI-compatible endpoint.

## Workflow

1. Prepare image input path and target label text.
2. Run [scripts/run_detection.py](scripts/run_detection.py).
3. Read output JSON for bbox list.

## Run

```bash
python3 skills/qwen-vlm-open-set-delection/scripts/run_detection.py \
  --images /path/to/image_or_folder \
  --label "fishing person" \
  --base-url http://127.0.0.1:9012/v1 \
  --model Qwen2.5-VL-7B-Instruct \
  --out /path/to/qwen_detection.json
```

## Input / Output

- `--images`: single image or directory.
- `--label`: detection target label.
- `--out`: output JSON path.

Output example: see [references/output_example.json](references/output_example.json).
