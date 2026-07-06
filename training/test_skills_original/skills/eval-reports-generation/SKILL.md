---
name: eval-reports-generation
description: Generates evaluation reports from image list and annotation/prediction results. Uses LLM to summarize per-image and per-model quality into structured report JSON.
---

# Eval Reports Generation

## Overview

- **Input**: image list + prediction/annotation JSON + task text + target label.
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
