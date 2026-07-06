#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Open-set detection with RexOmni. Input: image path or folder + prompt JSON. Output: MS COCO format.
Run inside Docker (agent_skill_base_image_cu118 or cu124). Uses references/parser.parse_prediction.
"""
import argparse
import base64
import json
import os
import sys
from pathlib import Path

from PIL import Image
from openai import OpenAI

# Resolve skill root (parent of scripts/) so references/parser is importable
SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))
try:
    from references.parser import parse_prediction
except ImportError:
    # Fallback when run from workspace with references beside script
    _ref = SKILL_ROOT / "references" / "parser.py"
    if _ref.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("parser", _ref)
        _mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_mod)
        parse_prediction = _mod.parse_prediction
    else:
        raise

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_images(path: str) -> list:
    """Resolve path to list of image file paths (file or directory)."""
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        files = []
        for f in sorted(p.iterdir()):
            if f.suffix.lower() in IMAGE_EXTENSIONS:
                files.append(str(f))
        if not files:
            raise ValueError(f"No image files found in directory: {p}")
        return files
    raise ValueError(f"Not a file or directory: {p}")


def load_prompt_config(prompt_path: str) -> tuple:
    """Load prompt JSON; return (prompts dict, index2cls dict). Validate prompts and index2cls."""
    with open(prompt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "prompts" not in data or "index2cls" not in data:
        raise ValueError("Prompt JSON must contain 'prompts' and 'index2cls'")
    return data["prompts"], data["index2cls"]


def build_detection_text(prompts: dict, index2cls: dict) -> str:
    """Build Rex Detect line: one natural-language phrase per class (not all synonyms)."""
    parts = []
    for idx in sorted(index2cls.keys(), key=int):
        cls = index2cls[idx]
        if cls not in prompts:
            raise ValueError(f"index2cls references class '{cls}' not in prompts")
        tokens = prompts[cls]
        if isinstance(tokens, list) and tokens:
            parts.append(str(tokens[0]).strip())
        else:
            parts.append(str(tokens or cls).strip())
    detect_line = ",".join(parts)
    return f"Detect {detect_line}. Output the bounding box coordinates in [x0, y0, x1, y1] format."


def build_phrase_alias_to_class_id(prompts: dict, index2cls: dict) -> dict[str, int]:
    """Map Rex object_ref phrases and synonyms back to COCO category_id."""
    cls_to_id = {index2cls[idx]: int(idx) for idx in index2cls}
    alias: dict[str, int] = {}
    for idx in sorted(index2cls.keys(), key=int):
        cls = index2cls[idx]
        cat_id = int(idx)
        alias[cls] = cat_id
        tokens = prompts.get(cls, [])
        if not isinstance(tokens, list):
            continue
        for token in tokens:
            t = str(token or "").strip()
            if t:
                alias[t] = cat_id
    return alias


def encode_image(img_path: str) -> str:
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _image_mime(img_path: str) -> str:
    ext = Path(img_path).suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "jpeg"
    if ext == ".png":
        return "png"
    if ext == ".webp":
        return "webp"
    if ext == ".bmp":
        return "bmp"
    return "jpeg"


def detect_one(client: OpenAI, model: str, image_path: str, detection_text: str) -> str:
    """Call Rex-Omni chat/completions for one image; return response text."""
    b64 = encode_image(image_path)
    mime = _image_mime(image_path)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{mime};base64,{b64}", "detail": "auto"},
                    },
                    {"type": "text", "text": detection_text},
                ],
            },
        ],
        temperature=0.0,
        top_p=0.05,
        extra_body={"skip_special_tokens": False},
    )
    return (resp.choices[0].message.content or "").strip()


def box_to_coco_bbox(coords: list) -> list:
    """Convert [x0, y0, x1, y1] to COCO [x, y, width, height]."""
    x0, y0, x1, y1 = coords
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


def main():
    parser = argparse.ArgumentParser(description="RexOmni open-set detection -> MS COCO format")
    parser.add_argument("--images", required=True, help="Image file or directory")
    parser.add_argument("--prompt", required=True, help="Path to prompt JSON (prompts + index2cls)")
    parser.add_argument(
        "--base-url",
        default="http://180.184.148.142:1032/v1",
        help="OpenAI-compatible API base URL (same gateway as Qwen if Rex is routed there)",
    )
    parser.add_argument(
        "--api-key",
        default="token.sdc@2026",
        help="API key for the Rex endpoint",
    )
    parser.add_argument(
        "--model",
        default="Rex-Omni",
        help="Served model name (OpenAI-compatible chat model id)",
    )
    parser.add_argument("--out", required=True, help="Output COCO JSON path")
    args = parser.parse_args()

    # Normalize input
    image_list = normalize_images(args.images)
    prompts, index2cls = load_prompt_config(args.prompt)
    detection_text = build_detection_text(prompts, index2cls)

    base_url = args.base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1" if "/v1" not in base_url else base_url
    client = OpenAI(base_url=base_url, api_key=args.api_key)

    # Build COCO structures
    coco_images = []
    coco_categories = [{"id": int(idx), "name": index2cls[idx]} for idx in sorted(index2cls.keys(), key=int)]
    coco_annotations = []
    ann_id = 1

    for im_idx, img_path in enumerate(image_list):
        pil = Image.open(img_path)
        w, h = pil.size
        coco_images.append({"id": im_idx, "file_name": os.path.basename(img_path), "width": w, "height": h})

        text = detect_one(client, args.model, img_path, detection_text)
        parsed = parse_prediction(text, w, h, task_type="detection")

        phrase_alias = build_phrase_alias_to_class_id(prompts, index2cls)

        for cat_name, preds in parsed.items():
            cat_id = phrase_alias.get(str(cat_name).strip())
            if cat_id is None:
                continue
            for pred in preds:
                if pred.get("type") != "box":
                    continue
                coords = pred.get("coords", [])
                if len(coords) != 4:
                    continue
                x, y, ww, hh = box_to_coco_bbox(coords)
                area = ww * hh
                coco_annotations.append({
                    "id": ann_id,
                    "image_id": im_idx,
                    "category_id": cat_id,
                    "bbox": [x, y, ww, hh],
                    "area": area,
                })
                ann_id += 1

    result = {"images": coco_images, "categories": coco_categories, "annotations": coco_annotations}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Wrote COCO format to {out_path}")


if __name__ == "__main__":
    main()
