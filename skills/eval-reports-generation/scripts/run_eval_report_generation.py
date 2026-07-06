#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from util.prompts import evaluation_summary_prompt  # noqa: E402
from util.schemas import evaluation_summary_response_format  # noqa: E402
from util.vlm_service import VLMService  # noqa: E402


def parse_images(images_arg: str) -> list[str]:
    images_arg = images_arg.strip()
    if images_arg.startswith("["):
        data = json.loads(images_arg)
        if not isinstance(data, list):
            raise ValueError("--images JSON must be a list")
        return [str(x) for x in data]
    p = Path(images_arg).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Image list file not found: {p}")
    lines = [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    return lines


def normalize_reports(prediction_data: object, image_path_list: list[str]) -> list[dict]:
    if isinstance(prediction_data, dict) and "reports_json" in prediction_data:
        return prediction_data["reports_json"]
    if isinstance(prediction_data, list):
        if prediction_data and isinstance(prediction_data[0], dict) and "models" in prediction_data[0]:
            idx_by_name = {Path(p).name: i for i, p in enumerate(image_path_list)}
            reports = []
            for rec in prediction_data:
                image_name = rec.get("image")
                source = rec.get("source", "unknown")
                image_idx = rec.get("image_idx", idx_by_name.get(image_name))
                for model_data in rec.get("models", []):
                    pred_bboxes = model_data.get("pred_bboxes", [])
                    reports.append(
                        {
                            "image_idx": image_idx,
                            "image": image_name,
                            "source": source,
                            "model": model_data.get("model", ""),
                            "pred_bboxes": pred_bboxes,
                            "num_boxes": len(pred_bboxes),
                        }
                    )
            return sorted(reports, key=lambda x: (x.get("image_idx") is None, x.get("image_idx", 10**9), x.get("model", "")))
        return prediction_data
    raise ValueError("Unsupported prediction JSON format")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation reports generation (detection quality summary)")
    parser.add_argument("--images", required=True, help="JSON list string or txt file path")
    parser.add_argument("--prediction", required=True, help="Prediction/annotation JSON file")
    parser.add_argument("--task-text", required=True, help="Original task text")
    parser.add_argument("--target-label", required=True, help="Target label")
    parser.add_argument(
        "--api-base",
        default="http://180.184.148.142:1032/v1",
        help="OpenAI-compatible base URL",
    )
    parser.add_argument("--api-key", default="token.sdc@2026", help="API key")
    parser.add_argument("--model", default="Qwen3.5-4B", help="Model name")
    parser.add_argument("--out", required=True, help="Output report JSON path")
    args = parser.parse_args()

    images = parse_images(args.images)
    with open(args.prediction, "r", encoding="utf-8") as f:
        prediction = json.load(f)
    reports = normalize_reports(prediction, images)
    reports_json = json.dumps(reports, indent=2, ensure_ascii=False)

    prompt = evaluation_summary_prompt.format(
        task_text=args.task_text,
        target_label=args.target_label,
        reports_json=reports_json,
    )
    vlm = VLMService(api_key=args.api_key, base_url=args.api_base.rstrip("/"))
    response = vlm.generate_text(
        prompt=prompt,
        image_paths=images,
        model=args.model,
        response_format=evaluation_summary_response_format,
    )
    summary = json.loads(response)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Wrote evaluation report to {out_path}")


if __name__ == "__main__":
    main()
