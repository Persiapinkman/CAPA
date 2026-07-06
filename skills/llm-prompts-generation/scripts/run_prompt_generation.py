#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from util.prompts import image_generate_prompt_template  # noqa: E402
from util.schemas import image_generate_response_format, image_generate_response_schema  # noqa: E402
from util.vlm_service import VLMService  # noqa: E402


def _str_or_empty(v: object) -> str:
    if v is None:
        return ""
    return str(v).strip()


def normalize_expand_descriptions(result: dict, intent: dict) -> None:
    """
    模型偶尔会漏字段（如缺 scene）。补全后再做 jsonschema 校验，避免整条流水线失败。
    """
    expand = result.get("expand_descriptions")
    if not isinstance(expand, list):
        return

    fallback_scene = _str_or_empty(intent.get("scene"))
    fallback_target = _str_or_empty(intent.get("target"))
    fallback_camera = _str_or_empty(intent.get("camera"))

    out: list[dict] = []
    for item in expand:
        if not isinstance(item, dict):
            out.append(
                {
                    "scene": fallback_scene,
                    "target": fallback_target,
                    "camera": fallback_camera,
                }
            )
            continue
        scene = _str_or_empty(item.get("scene")) or fallback_scene
        target = _str_or_empty(item.get("target")) or fallback_target
        camera = _str_or_empty(item.get("camera")) or fallback_camera
        out.append({"scene": scene, "target": target, "camera": camera})

    # schema 要求至少 10 条；不足时用最后一条或 intent 兜底补齐
    while len(out) < 10:
        tail = out[-1] if out else {
            "scene": fallback_scene,
            "target": fallback_target,
            "camera": fallback_camera,
        }
        out.append(dict(tail))

    result["expand_descriptions"] = out


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM prompts generation skill")
    parser.add_argument("--intent", required=True, help="Path to intent JSON")
    parser.add_argument("--task-text", required=True, help="Original user task text")
    parser.add_argument(
        "--api-base",
        default="http://180.184.148.142:1032/v1",
        help="OpenAI-compatible base URL",
    )
    parser.add_argument("--api-key", default="token.sdc@2026", help="API key")
    parser.add_argument("--model", default="Qwen3.5-4B", help="Model name")
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args()

    with open(args.intent, "r", encoding="utf-8") as f:
        intent = json.load(f)

    label = str(intent["target_label"]).replace("_", " ")
    prompt = image_generate_prompt_template.format(
        args.task_text,
        label,
        intent.get("scene", ""),
        intent.get("camera", ""),
        intent.get("target", ""),
    )

    vlm = VLMService(api_key=args.api_key, base_url=args.api_base.rstrip("/"))
    response = vlm.generate_text(
        prompt=prompt,
        image_paths=[],
        response_format=image_generate_response_format,
        model=args.model,
    )
    result = json.loads(response)
    normalize_expand_descriptions(result, intent)
    jsonschema.validate(result, image_generate_response_schema)

    expand = result.get("expand_descriptions", [])
    polished_prompts = [
        f"{item.get('camera', '')} {item.get('scene', '')} {item.get('target', '')} {label} must be in the image".strip()
        for item in expand
        if isinstance(item, dict)
    ]
    output = {"expand_descriptions": expand, "polished_prompts": polished_prompts}

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Wrote polished prompts to {out_path}")


if __name__ == "__main__":
    main()
