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


def _box_iou(left: list, right: list) -> float:
    try:
        lx1, ly1, lx2, ly2 = (float(value) for value in left)
        rx1, ry1, rx2, ry2 = (float(value) for value in right)
    except (TypeError, ValueError):
        return 0.0
    intersection_width = max(0.0, min(lx2, rx2) - max(lx1, rx1))
    intersection_height = max(0.0, min(ly2, ry2) - max(ly1, ry1))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def cross_model_agreement(prediction_data: object) -> list[dict]:
    """Compute deterministic Qwen/Rex agreement without treating either model as GT."""
    if not isinstance(prediction_data, list):
        return []
    output: list[dict] = []
    for record in prediction_data:
        if not isinstance(record, dict):
            continue
        by_model: dict[str, list[list]] = {}
        for model_row in record.get("models", []):
            if not isinstance(model_row, dict):
                continue
            name = str(model_row.get("model") or "").lower()
            boxes = [
                box
                for box in model_row.get("pred_bboxes", [])
                if isinstance(box, list) and len(box) == 4
            ]
            if "qwen" in name:
                by_model["qwen"] = boxes
            elif "rex" in name:
                by_model["rex"] = boxes
        qwen_boxes = by_model.get("qwen", [])
        rex_boxes = by_model.get("rex", [])
        candidates = sorted(
            (
                (_box_iou(qwen_box, rex_box), qwen_idx, rex_idx)
                for qwen_idx, qwen_box in enumerate(qwen_boxes)
                for rex_idx, rex_box in enumerate(rex_boxes)
            ),
            reverse=True,
        )
        used_qwen: set[int] = set()
        used_rex: set[int] = set()
        matched_ious: list[float] = []
        for iou, qwen_idx, rex_idx in candidates:
            if iou < 0.5 or qwen_idx in used_qwen or rex_idx in used_rex:
                continue
            used_qwen.add(qwen_idx)
            used_rex.add(rex_idx)
            matched_ious.append(iou)
        denominator = max(len(qwen_boxes), len(rex_boxes), 1)
        output.append(
            {
                "image_idx": record.get("image_idx"),
                "image": record.get("image"),
                "source": record.get("source"),
                "qwen_box_count": len(qwen_boxes),
                "rex_box_count": len(rex_boxes),
                "matched_at_iou_0_5": len(matched_ious),
                "match_rate": round(len(matched_ious) / denominator, 6),
                "mean_matched_iou": round(
                    sum(matched_ious) / len(matched_ious), 6
                )
                if matched_ious
                else 0.0,
            }
        )
    return output


def build_deterministic_report(agreement: list[dict]) -> dict:
    per_image: list[dict] = []
    total_qwen = 0
    total_rex = 0
    total_matched = 0
    matched_iou_sum = 0.0
    for row in agreement:
        qwen_count = int(row.get("qwen_box_count") or 0)
        rex_count = int(row.get("rex_box_count") or 0)
        matched = int(row.get("matched_at_iou_0_5") or 0)
        mean_iou = float(row.get("mean_matched_iou") or 0.0)
        total_qwen += qwen_count
        total_rex += rex_count
        total_matched += matched
        matched_iou_sum += mean_iou * matched
        agreement_text = (
            f"Qwen 返回 {qwen_count} 个框，Rex-Omni 返回 {rex_count} 个框；"
            f"以跨模型 IoU>=0.5 贪心匹配得到 {matched} 对，"
            f"匹配框平均 IoU={mean_iou:.3f}。该指标只表示模型间一致性。"
        )
        per_image.append(
            {
                "image_idx": row.get("image_idx"),
                "image": row.get("image"),
                "source": row.get("source"),
                "qwen3-vl-8b": {
                    "accuracy": "N/A（无人工GT）",
                    "reason": agreement_text,
                },
                "rex-omni": {
                    "accuracy": "N/A（无人工GT）",
                    "reason": agreement_text,
                },
            }
        )
    aggregate_iou = matched_iou_sum / total_matched if total_matched else 0.0
    conclusion = (
        f"本次共评估 {len(agreement)} 张图片，Qwen 共返回 {total_qwen} 个框，"
        f"Rex-Omni 共返回 {total_rex} 个框；跨模型匹配 {total_matched} 对，"
        f"匹配框平均 IoU={aggregate_iou:.3f}。这些数字只能衡量两个模型输出的一致性，"
        "不能替代人工 GT，也不能据此计算准确率、漏检率或误检率。"
        "在补充独立标注前，不选择胜出模型。"
    )
    return {
        "per_image_evaluation": per_image,
        "overall_conclusion": conclusion,
        "model_results": {
            "qwen3-vl-8b": {
                "accuracy": "N/A（无人工GT）",
                "reason": f"共输出 {total_qwen} 个框；只能与 Rex-Omni 比较一致性，不能判断正确性。",
            },
            "rex-omni": {
                "accuracy": "N/A（无人工GT）",
                "reason": f"共输出 {total_rex} 个框；只能与 Qwen 比较一致性，不能判断正确性。",
            },
        },
        "recommendation": "inconclusive",
    }


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
    parser.add_argument(
        "--enable-vlm-summary",
        action="store_true",
        help="Attach an optional qualitative VLM assessment; never replaces deterministic metrics.",
    )
    parser.add_argument("--out", required=True, help="Output report JSON path")
    args = parser.parse_args()

    images = parse_images(args.images)
    with open(args.prediction, "r", encoding="utf-8") as f:
        prediction = json.load(f)
    reports = normalize_reports(prediction, images)
    reports_json = json.dumps(reports, indent=2, ensure_ascii=False)
    agreement = cross_model_agreement(prediction)
    agreement_json = json.dumps(agreement, indent=2, ensure_ascii=False)

    summary = build_deterministic_report(agreement)
    if args.enable_vlm_summary:
        prompt = evaluation_summary_prompt.format(
            task_text=args.task_text,
            target_label=args.target_label,
            reports_json=reports_json,
            agreement_json=agreement_json,
        )
        vlm = VLMService(api_key=args.api_key, base_url=args.api_base.rstrip("/"))
        response = vlm.generate_text(
            prompt=prompt,
            image_paths=images,
            model=args.model,
            response_format=evaluation_summary_response_format,
        )
        summary["qualitative_vlm_assessment"] = json.loads(response)
    summary["evaluation_basis"] = {
        "ground_truth_available": False,
        "visual_input": "annotated images with Qwen red boxes and Rex-Omni blue boxes",
        "cross_model_agreement": agreement,
        "warning": (
            "Accuracy cannot be computed without independent ground truth. "
            "Cross-model IoU measures agreement, not correctness."
        ),
    }

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Wrote evaluation report to {out_path}")


if __name__ == "__main__":
    main()
