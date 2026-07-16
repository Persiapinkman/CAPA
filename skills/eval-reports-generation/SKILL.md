---
name: eval-reports-generation
description: Generates no-GT agreement reports from image lists and detection predictions. Uses deterministic count/IoU metrics by default; optional VLM text is non-authoritative.
---

# Eval Reports Generation

## Overview

- **Input**: image list + prediction/annotation JSON + task text + target label.
- **Default output**: deterministic cross-model box-count and IoU agreement. Accuracy and model recommendation remain unavailable without independent GT.
- **Optional**: `--enable-vlm-summary` attaches qualitative VLM text, but it must not be used as benchmark truth or RL reward.
- **Output**: structured evaluation report JSON (same schema as former reports-generation).

## Run

```bash
python3 skills/eval-reports-generation/scripts/run_eval_report_generation.py \
  --images '["/path/to/orig.jpg","/path/to/gen1.jpg"]' \
  --prediction /path/to/prediction.json \
  --task-text "检测钓鱼的人" \
  --target-label "fishing_person" \
  --api-base http://180.184.148.142:1032/v1 \
  --api-key "token.sdc@2026" \
  --model Qwen3.5-4B \
  --out /path/to/evaluation.json
```

Prediction input can follow `agent.py` style (`image/source/image_idx/models`) or flat `reports_json`.
