---
name: flux-image-generation
description: Generates one image with Flux from one prompt, optionally conditioned on a source image. Supports both text-to-image and image-to-image.
---

# Flux Image Generation

## Overview

- **Input**: one prompt + output image path; `source-image` is optional.
- **Output**: one generated image file.
- **Runtime**:
  - with `--source-image`: OpenAI-compatible image edit API (`images.edit`)
  - without `--source-image`: OpenAI-compatible text generation API (`images.generate`)

## Run

### Image-to-image

```bash
python3 skills/flux-image-generation/scripts/run_generation.py \
  --source-image /path/to/source.jpg \
  --prompt "A person fishing beside a river under cloudy sky" \
  --api-base https://api.apiyi.com/v1 \
  --api-key "$APIYI_API_KEY" \
  --out /path/to/generated_images/generated_0.jpg
```

### Text-to-image

```bash
python3 skills/flux-image-generation/scripts/run_generation.py \
  --prompt "A red dog in the snow, cinematic lighting, 4k" \
  --api-base https://api.apiyi.com/v1 \
  --api-key "$APIYI_API_KEY" \
  --out /path/to/generated_images/generated_0.jpg
```
