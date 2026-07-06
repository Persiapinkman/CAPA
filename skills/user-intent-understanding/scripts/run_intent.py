#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from util.prompts import intent_prompt  # noqa: E402
from util.schemas import intent_response_format, intent_response_schema  # noqa: E402
from util.vlm_service import VLMService  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="User intent understanding skill")
    parser.add_argument("--text", required=True, help="User text input")
    parser.add_argument("--image", default=None, help="Optional image path")
    parser.add_argument(
        "--api-base",
        default="http://180.184.148.142:1032/v1",
        help="OpenAI-compatible base URL",
    )
    parser.add_argument("--api-key", default="token.sdc@2026", help="API key")
    parser.add_argument("--model", default="Qwen3.5-4B", help="Model name")
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args()

    image_paths = None
    if args.image:
        image_path = Path(args.image).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image_paths = [str(image_path)]

    vlm = VLMService(api_key=args.api_key, base_url=args.api_base.rstrip("/"))
    response = vlm.generate_text(
        prompt=intent_prompt.format(args.text),
        model=args.model,
        response_format=intent_response_format,
        image_paths=image_paths,
    )
    result = json.loads(response)
    jsonschema.validate(result, intent_response_schema)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Wrote intent result to {out_path}")


if __name__ == "__main__":
    main()
