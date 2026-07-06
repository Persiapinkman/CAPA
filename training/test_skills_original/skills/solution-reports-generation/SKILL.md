---
name: solution-reports-generation
description: From one text plus one image, calls a VLM to produce a structured solution report (goals, training plan, annotation format/howto, data scale, metrics, deployment, performance estimate).
---

# Solution Reports Generation

## Overview

- **Input**: user requirement text + optional reference image.
- **Output**: structured JSON solution report suitable for handing off to engineering.

Report sections (fields in output JSON):

- Background and goals  
- Model training plan  
- Annotation data format  
- How to annotate  
- Data volume requirements  
- Evaluation metrics  
- Model deployment plan  
- Performance estimate  

## Run

```bash
python3 skills/solution-reports-generation/scripts/run_solution_report.py \
  --text "在河岸监控场景中检测违规垂钓人员，需要对接现有球机视频流" \
  --api-base http://180.184.148.142:1032/v1 \
  --api-key "token.sdc@2026" \
  --model Qwen3.5-4B \
  --out /path/to/solution_report.json
```

With reference image (optional):

```bash
python3 skills/solution-reports-generation/scripts/run_solution_report.py \
  --text "在河岸监控场景中检测违规垂钓人员，需要对接现有球机视频流" \
  --image /path/to/reference.jpg \
  --api-base http://180.184.148.142:1032/v1 \
  --api-key "token.sdc@2026" \
  --model Qwen3.5-4B \
  --out /path/to/solution_report.json
```

Example shape: [references/solution_report_example.json](references/solution_report_example.json).
