#!/usr/bin/env python3
import argparse
from io import BytesIO
import sys
from pathlib import Path

import requests
from openai import OpenAI
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from util.prompts import image_to_image_prompt_template

def download_image(url: str, save_path: Path, timeout: int = 20) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    try:
        with Image.open(BytesIO(resp.content)) as image:
            image.load()
            suffix = save_path.suffix.lower()
            if suffix in {".jpg", ".jpeg"}:
                image.convert("RGB").save(save_path, format="JPEG", quality=95)
            elif suffix == ".png":
                image.save(save_path, format="PNG")
            elif suffix == ".webp":
                image.save(save_path, format="WEBP", quality=95)
            else:
                image.save(save_path)
    except UnidentifiedImageError as exc:
        content_type = str(resp.headers.get("Content-Type") or "unknown")
        raise ValueError(
            f"Flux download did not return a decodable image (Content-Type: {content_type})"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Flux image generation skill")
    parser.add_argument("--source-image", default=None, help="Optional source image path for image-to-image")
    parser.add_argument("--prompt", required=True, help="Single prompt string")
    parser.add_argument("--api-base", default="https://api.apiyi.com/v1", help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", required=True, help="API key")
    parser.add_argument("--model", default="flux-kontext-pro", help="Flux model name")
    parser.add_argument("--out", required=True, help="Output image file path")
    args = parser.parse_args()

    prompt = str(args.prompt).strip()
    if not prompt:
        raise ValueError("Prompt is empty")

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=args.api_key, base_url=args.api_base.rstrip("/"))

    # If source image exists, run image-edit; otherwise run text-to-image.
    if args.source_image:
        source_image = Path(args.source_image).resolve()
        if not source_image.exists():
            raise FileNotFoundError(f"Source image not found: {source_image}")
        edit_prompt = image_to_image_prompt_template.format(prompt)
        with open(source_image, "rb") as f:
            resp = client.images.edit(model=args.model, image=f, prompt=edit_prompt)
    else:
        resp = client.images.generate(model=args.model, prompt=prompt)
    image_url = resp.data[0].url
    download_image(image_url, out_path)
    print(f"Wrote generated image to {out_path}")


if __name__ == "__main__":
    main()
