#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from openai import OpenAI

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DETECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "bboxes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "bbox": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "score": {"type": "number"},
                },
                "required": ["label", "bbox", "score"],
            },
        }
    },
    "required": ["bboxes"],
}


def normalize_images(path: str) -> list[str]:
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        files = [str(f) for f in sorted(p.iterdir()) if f.suffix.lower() in IMAGE_EXTENSIONS]
        if not files:
            raise ValueError(f"No image files found in directory: {p}")
        return files
    raise ValueError(f"Not a file or directory: {p}")


def image_data_url(image_path: str) -> str:
    import base64
    ext = Path(image_path).suffix.lower()
    mime = "jpeg" if ext in {".jpg", ".jpeg"} else ext.lstrip(".")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{mime};base64,{b64}"


def detect_one(client: OpenAI, model: str, image_path: str, label: str) -> dict:
    prompt = (
        f"You are a vision detection model. Detect object: {label}. "
        "Return JSON with key bboxes, each bbox uses absolute pixel coordinates [x1,y1,x2,y2]."
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
                ],
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "qwen_detection", "schema": DETECTION_SCHEMA},
        },
    )
    content = (response.choices[0].message.content or "").strip()
    return json.loads(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen VLM open-set detection")
    parser.add_argument("--images", required=True, help="Image file or directory")
    parser.add_argument("--label", required=True, help="Target label text")
    parser.add_argument(
        "--base-url",
        default="http://10.111.32.254:9012/v1",
        help="Qwen OpenAI-compatible base URL",
    )
    parser.add_argument("--model", default="Qwen2.5-VL-7B-Instruct", help="Model name")
    parser.add_argument("--api-key", default="token.sdc@2026", help="OpenAI-compatible API key")
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args()

    images = normalize_images(args.images)
    client = OpenAI(base_url=args.base_url.rstrip("/"), api_key=args.api_key)
    results = []
    for image in images:
        det = detect_one(client, args.model, image, args.label)
        results.append({"image": image, "label": args.label, "bboxes": det.get("bboxes", [])})

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2, ensure_ascii=False)
    print(f"Wrote detection result to {out_path}")


if __name__ == "__main__":
    main()
