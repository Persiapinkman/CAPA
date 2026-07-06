#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Open-set image classification with Qwen VLM. Input: image path or folder + prompt JSON. Output: MS COCO format.
Run inside Docker (agent_skill_base_image_cu118 or cu124).
"""
import argparse
import json
import math
import os
import sys
from io import BytesIO
from pathlib import Path

import base64
from PIL import Image
from openai import OpenAI

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODEL_NAME = "Qwen2.5-VL-7B-Instruct"

CLASS_BASE_PROMPT = '''
你需要进行分类，首先描述图片，然后从类别中选择一项输出。比如说:
user:使用图片分类,类别名称为:[车,人,海,船,老虎],只允许是这几个类别中的一种，请告诉我你为什么这么选择?
assistant:{"desc":"图片中有一辆车，所以类别是车","label":"车"}
'''

CLASS_USE_PROMPT = '使用图片分类,类别名称为:@LABELS, 只允许是这几个类别中的一种，请告诉我你为什么这么选择'

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "desc": {"type": "string"},
        "label": {"type": "string"},
    },
    "required": ["desc", "label"],
}


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


def build_class_prompts_and_mapping(prompts: dict, index2cls: dict) -> tuple:
    """
    From prompts (class_key -> list of label strings) build:
    - class_prompts: flat list of all label strings for the user prompt
    - class_prompt2classname: label string -> class key (for mapping model output back)
    - cls2index: class key -> index string (for category_id)
    """
    class_prompts = []
    class_prompt2classname = {}
    for cls_key in prompts:
        for label in prompts[cls_key]:
            class_prompts.append(label)
            class_prompt2classname[label] = cls_key
    cls2index = {}
    for idx, cls_key in index2cls.items():
        cls2index[cls_key] = idx
    return class_prompts, class_prompt2classname, cls2index


def get_system_and_user_prompt(class_prompts: list) -> tuple:
    """Build system and user prompt for Qwen VLM classification."""
    user_classes_repr = repr(class_prompts)
    user_prompt = CLASS_USE_PROMPT.replace("@LABELS", user_classes_repr)
    return CLASS_BASE_PROMPT.strip(), user_prompt


def qwen_resize_and_base64(img_path: str, min_pixels=16 * 28 * 28, max_pixels=960 * 28 * 28) -> str:
    """Load image, resize by pixel count, encode as JPEG base64."""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    pixels = w * h
    if pixels < min_pixels:
        target = min_pixels
    elif pixels > max_pixels:
        target = max_pixels
    else:
        target = pixels
    scale = math.sqrt(target / pixels)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = img.resize((new_w, new_h), Image.BICUBIC)
    buf = BytesIO()
    resized.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def classify_one(
    client: OpenAI,
    image_path: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Call Qwen VLM chat/completions for one image; return response content (JSON string)."""
    b64 = qwen_resize_and_base64(image_path)
    response_format = dict(type="json_schema", json_schema=dict(name="classification", schema=OUTPUT_SCHEMA))
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                },
            ],
        },
    ]
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        response_format=response_format,
    )
    return (resp.choices[0].message.content or "").strip()


def main():
    parser = argparse.ArgumentParser(description="Qwen VLM open-set classification -> MS COCO format")
    parser.add_argument("--images", required=True, help="Image file or directory")
    parser.add_argument("--prompt", required=True, help="Path to prompt JSON (prompts + index2cls)")
    parser.add_argument("--base-url", default="http://127.0.0.1:9012/v1", help="Qwen VLM API base URL")
    parser.add_argument("--out", required=True, help="Output COCO JSON path")
    args = parser.parse_args()

    # Input normalization
    image_list = normalize_images(args.images)
    prompts, index2cls = load_prompt_config(args.prompt)
    class_prompts, class_prompt2classname, cls2index = build_class_prompts_and_mapping(prompts, index2cls)
    system_prompt, user_prompt = get_system_and_user_prompt(class_prompts)

    base_url = args.base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1" if "/v1" not in base_url else base_url
    client = OpenAI(base_url=base_url, api_key="YOUR_API_KEY")

    # Build COCO structures (classification: no bbox in annotations)
    coco_images = []
    coco_categories = [{"id": int(idx), "name": index2cls[idx]} for idx in sorted(index2cls.keys(), key=int)]
    coco_annotations = []
    ann_id = 1

    for im_idx, img_path in enumerate(image_list):
        pil = Image.open(img_path)
        w, h = pil.size
        coco_images.append({"id": im_idx, "file_name": os.path.basename(img_path), "width": w, "height": h})

        content = classify_one(client, img_path, system_prompt, user_prompt)
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = {"label": "", "desc": content}
        label = parsed.get("label", "").strip()
        class_name = class_prompt2classname.get(label)
        if class_name is None:
            # Try exact match first; if not, use first class as fallback or skip
            for k, v in class_prompt2classname.items():
                if k == label or (isinstance(k, str) and k.strip() == label):
                    class_name = v
                    break
        if class_name is not None and class_name in cls2index:
            cat_id = int(cls2index[class_name])
            coco_annotations.append({
                "id": ann_id,
                "image_id": im_idx,
                "category_id": cat_id,
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
