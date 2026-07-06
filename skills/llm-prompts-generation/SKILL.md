---
name: llm-prompts-generation
description: Generates polished image-generation prompts from intent understanding JSON, returning structured scene/target/camera descriptions and final prompt strings.
---

# LLM Prompts Generation

## Overview

- **Input**: intent JSON from `user-intent-understanding`.
- **Output**: polished prompt JSON for image generation.

## Run

```bash
python3 skills/llm-prompts-generation/scripts/run_prompt_generation.py \
  --intent /path/to/intent.json \
  --task-text "检测钓鱼的人" \
  --api-base http://180.184.148.142:1032/v1 \
  --api-key "token.sdc@2026" \
  --model Qwen3.5-4B \
  --out /path/to/polished_prompts.json
```

Output example: see [references/prompts_output_example.json](references/prompts_output_example.json).
