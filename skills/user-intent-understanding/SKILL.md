---
name: user-intent-understanding
description: Understands user intent from one sentence plus one image, and outputs structured intent JSON (task, target, scene, camera, prompts).
---

# User Intent Understanding

## Overview

- **Input**: user sentence + optional image.
- **Output**: structured intent JSON for downstream prompt generation and evaluation.

## Run

```bash
python3 skills/user-intent-understanding/scripts/run_intent.py \
  --text "检测钓鱼的人" \
  --image /path/to/reference.jpg \
  --api-base http://180.184.148.142:1032/v1 \
  --api-key "token.sdc@2026" \
  --model Qwen3.5-4B \
  --out /path/to/intent.json
```

Output schema example: see [references/intent_example.json](references/intent_example.json).
