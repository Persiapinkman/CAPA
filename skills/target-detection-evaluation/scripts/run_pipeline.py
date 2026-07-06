#!/usr/bin/env python3
"""
端到端目标检测评测流水线（与 examples/run_eval_pipeline_example.py 对齐）。

步骤：
1) user-intent-understanding
2) llm-prompts-generation
3) flux-image-generation（按 prompts 批量单张生成）
4) qwen + rexomni 开集检测 → prediction.json
5) annotated_images：Qwen 红框、Rex-Omni 蓝框
6) eval-reports-generation

须在仓库根目录执行子进程路径；本脚本会自动 chdir 到仓库根。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

COLOR_QWEN = (255, 0, 0)
COLOR_REX = (0, 0, 255)
BOX_WIDTH = 4


def repo_root() -> Path:
    # .../skills/target-detection-evaluation/scripts/run_pipeline.py -> parents[3] = repo root
    return Path(__file__).resolve().parents[3]


def run(cmd: list[str]) -> None:
    masked: list[str] = []
    i = 0
    while i < len(cmd):
        tok = cmd[i]
        if tok in ("--api-key", "--llm-api-key", "--flux-api-key"):
            masked.append(tok)
            if i + 1 < len(cmd):
                masked.append("***")
                i += 2
                continue
        elif tok.startswith("--api-key=") or tok.startswith("--llm-api-key=") or tok.startswith("--flux-api-key="):
            key, _, _val = tok.partition("=")
            masked.append(f"{key}=***")
            i += 1
            continue
        masked.append(tok)
        i += 1
    print("RUN:", " ".join(masked))
    subprocess.run(cmd, check=True)


def load_prompts_for_generation(prompts_json_path: Path) -> list[str]:
    with open(prompts_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if isinstance(data.get("polished_prompts"), list):
            return [str(x).strip() for x in data["polished_prompts"] if str(x).strip()]
        if isinstance(data.get("prompts"), list):
            return [str(x).strip() for x in data["prompts"] if str(x).strip()]
    if isinstance(data, list):
        return [str(x).strip() for x in data if str(x).strip()]
    raise ValueError(f"Unsupported prompts format: {prompts_json_path}")


def slug_class_key(label: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", label.strip()).strip("_")
    return s or "target"


def write_rex_prompt_json(path: Path, intent: dict) -> None:
    """
    写入 Rex prompt JSON。优先使用 LLM 抽取的 classes[]（多类别）；
    否则使用 class_key + tokens 或 target_label / target_keywords 回退。
    """
    classes_in = intent.get("classes")
    prompts: dict[str, list[str]] = {}
    index2cls: dict[str, str] = {}
    if isinstance(classes_in, list) and classes_in:
        for idx, row in enumerate(classes_in, start=1):
            if not isinstance(row, dict):
                continue
            label = str(row.get("label") or "").strip()
            cls_key = str(row.get("class_key") or slug_class_key(label)).strip() or slug_class_key(label)
            detect_phrase = str(row.get("detect_phrase") or "").strip()
            tokens_raw = row.get("tokens")
            tokens: list[str] = []
            seen: set[str] = set()
            if detect_phrase:
                tokens.append(detect_phrase)
                seen.add(detect_phrase.lower())
            if isinstance(tokens_raw, list):
                for item in tokens_raw:
                    t = str(item or "").strip()
                    if not t or t.lower() in seen:
                        continue
                    seen.add(t.lower())
                    tokens.append(t)
            if label and label.lower() not in seen:
                tokens.append(label)
            if not tokens:
                continue
            prompts[cls_key] = tokens
            index2cls[str(idx)] = cls_key
    if not prompts:
        cls_key = str(intent.get("class_key") or "").strip()
        tokens_raw = intent.get("tokens")
        tokens = []
        if isinstance(tokens_raw, list):
            seen: set[str] = set()
            for item in tokens_raw:
                t = str(item or "").strip()
                if not t or t.lower() in seen:
                    continue
                seen.add(t.lower())
                tokens.append(t)
        if not cls_key or len(tokens) < 2:
            target_label = str(intent.get("target_label", "")).strip()
            keywords = [str(k).strip() for k in (intent.get("target_keywords") or []) if str(k).strip()]
            seen = {t.lower() for t in tokens}
            if target_label and target_label.lower() not in seen:
                tokens.insert(0, target_label)
                seen.add(target_label.lower())
            for k in keywords:
                if k.lower() not in seen:
                    tokens.append(k)
                    seen.add(k.lower())
            cls_key = cls_key or slug_class_key(target_label) or "detect_object"
            if len(tokens) < 2 and target_label:
                tokens.append("object")
        if not tokens:
            raise ValueError("rex prompt requires classes or class_key+tokens or target_label")
        prompts = {cls_key: tokens}
        index2cls = {"1": cls_key}

    data = {"prompts": prompts, "index2cls": index2cls}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def coco_to_pred_bboxes_by_image_id(coco: dict) -> dict[int, list[list[float]]]:
    out: dict[int, list[list[float]]] = {}
    for ann in coco.get("annotations", []):
        iid = ann.get("image_id")
        bbox = ann.get("bbox")
        if iid is None or not bbox or len(bbox) != 4:
            continue
        x, y, w, h = bbox
        box = [float(x), float(y), float(x + w), float(y + h)]
        out.setdefault(int(iid), []).append(box)
    return out


def _model_to_color(model_name: str) -> tuple[int, int, int] | None:
    m = (model_name or "").lower()
    if "qwen" in m:
        return COLOR_QWEN
    if "rex" in m:
        return COLOR_REX
    return None


def draw_detection_overlays(
    image_map: dict[str, Path],
    prediction_path: Path,
    out_dir: Path,
) -> None:
    with open(prediction_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()

    for rec in records:
        name = rec.get("image")
        if not name:
            continue
        src = image_map.get(name)
        if not src or not src.is_file():
            continue

        img = Image.open(src).convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size

        models = [m for m in rec.get("models", []) if isinstance(m, dict)]
        models_sorted = sorted(
            models,
            key=lambda x: (0 if "rex" in str(x.get("model", "")).lower() else 1),
        )

        for m in models_sorted:
            color = _model_to_color(str(m.get("model", "")))
            if color is None:
                continue
            for box in m.get("pred_bboxes", []):
                if not isinstance(box, (list, tuple)) or len(box) != 4:
                    continue
                try:
                    x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
                except (TypeError, ValueError):
                    continue
                x1i = int(max(0, min(w - 1, round(x1))))
                y1i = int(max(0, min(h - 1, round(y1))))
                x2i = int(max(0, min(w - 1, round(x2))))
                y2i = int(max(0, min(h - 1, round(y2))))
                if x2i <= x1i or y2i <= y1i:
                    continue
                draw.rectangle(
                    [x1i, y1i, x2i, y2i],
                    outline=color,
                    width=BOX_WIDTH,
                )

        draw.rectangle([2, 2, 210, 24], fill=(0, 0, 0))
        draw.text((6, 5), "RED=Qwen BLUE=Rex-Omni", fill=(255, 255, 255), font=font)

        dest = out_dir / name
        if dest.suffix.lower() in (".jpg", ".jpeg"):
            img.save(dest, quality=95)
        else:
            img.save(dest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Target detection evaluation pipeline")
    parser.add_argument("--workspace", required=True, help="Working output directory")
    parser.add_argument("--image", required=True, help="Reference input image path")
    parser.add_argument("--text", required=True, help="User task text")
    parser.add_argument(
        "--api-base",
        default=None,
        help="Legacy unified mode: if set with --api-key, same base URL for LLM + Flux (overrides split defaults).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="With --api-base: key for all steps. Split mode (no --api-base): key for Flux only (required).",
    )
    parser.add_argument(
        "--llm-api-base",
        default="http://180.184.148.142:1032/v1",
        help="OpenAI-compatible base URL for intent / prompts / eval (split mode only)",
    )
    parser.add_argument(
        "--llm-api-key",
        default="token.sdc@2026",
        help="API key for intent / prompts / eval (split mode only)",
    )
    parser.add_argument(
        "--flux-api-base",
        default="https://api.apiyi.com/v1",
        help="OpenAI-compatible base URL for Flux image generation",
    )
    parser.add_argument("--num-images", type=int, default=5, help="Number of generated images")
    parser.add_argument(
        "--qwen-base-url",
        default="http://180.184.148.142:1032/v1",
        help="Qwen VLM detection OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--rex-base-url",
        default="http://180.184.148.142:1032/v1",
        help="RexOmni OpenAI-compatible base URL",
    )
    args = parser.parse_args()

    if args.api_base is not None:
        if not args.api_key:
            parser.error("--api-key is required when --api-base is set (legacy unified mode).")
        llm_api_base = args.api_base
        llm_api_key = args.api_key
        flux_api_base = args.api_base
        flux_api_key = args.api_key
    else:
        llm_api_base = args.llm_api_base
        llm_api_key = args.llm_api_key
        flux_api_base = args.flux_api_base
        flux_api_key = args.api_key
        if not flux_api_key:
            parser.error(
                "--api-key is required for Flux in split mode (or use legacy --api-base + --api-key for one URL for all steps)."
            )

    root = repo_root()
    os.chdir(root)

    ws = Path(args.workspace).resolve()
    ws.mkdir(parents=True, exist_ok=True)

    intent_json = ws / "intent.json"
    prompts_json = ws / "prompts.json"
    gen_json = ws / "generated_images.json"
    prediction_json = ws / "prediction.json"
    evaluation_json = ws / "evaluation.json"

    run(
        [
            "python3",
            "skills/user-intent-understanding/scripts/run_intent.py",
            "--text",
            args.text,
            "--image",
            args.image,
            "--api-base",
            llm_api_base,
            "--api-key",
            llm_api_key,
            "--out",
            str(intent_json),
        ]
    )

    run(
        [
            "python3",
            "skills/llm-prompts-generation/scripts/run_prompt_generation.py",
            "--intent",
            str(intent_json),
            "--task-text",
            args.text,
            "--api-base",
            llm_api_base,
            "--api-key",
            llm_api_key,
            "--out",
            str(prompts_json),
        ]
    )

    prompts_for_generation = load_prompts_for_generation(prompts_json)
    if not prompts_for_generation:
        raise ValueError("No prompts found in prompts.json")

    generated_dir = ws / "generated_images"
    generated_dir.mkdir(parents=True, exist_ok=True)
    generated_records: list[dict] = []
    for i in range(args.num_images):
        prompt = prompts_for_generation[i % len(prompts_for_generation)]
        out_image = generated_dir / f"generated_{i}.jpg"
        run(
            [
                "python3",
                "skills/flux-image-generation/scripts/run_generation.py",
                "--source-image",
                args.image,
                "--prompt",
                prompt,
                "--api-base",
                flux_api_base,
                "--api-key",
                flux_api_key,
                "--out",
                str(out_image),
            ]
        )
        generated_records.append(
            {
                "prompt": prompt,
                "saved_path": str(out_image.resolve()),
            }
        )

    with open(gen_json, "w", encoding="utf-8") as f:
        json.dump({"generated_images": generated_records}, f, indent=2, ensure_ascii=False)

    with open(intent_json, "r", encoding="utf-8") as f:
        intent = json.load(f)
    target_label = str(intent["target_label"]).replace("_", " ")

    gen_dir = ws / "generated_images"
    orig_image = Path(args.image).resolve()
    gen_files = sorted(
        (p for p in gen_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: p.name,
    )

    image_paths = [orig_image] + gen_files
    images_abs = [str(p.resolve()) for p in image_paths]
    image_map = {p.name: p for p in image_paths}

    qwen_orig_tmp = ws / ".qwen_orig_tmp.json"
    qwen_gen_tmp = ws / ".qwen_gen_tmp.json"
    rex_orig_tmp = ws / ".rex_orig_tmp.json"
    rex_gen_tmp = ws / ".rex_gen_tmp.json"
    rex_prompt_path = ws / ".rex_prompt_tmp.json"
    try:
        write_rex_prompt_json(rex_prompt_path, intent)

        run(
            [
                "python3",
                "skills/qwen-vlm-open-set-delection/scripts/run_detection.py",
                "--images",
                str(orig_image),
                "--label",
                target_label,
                "--base-url",
                args.qwen_base_url,
                "--out",
                str(qwen_orig_tmp),
            ]
        )
        run(
            [
                "python3",
                "skills/qwen-vlm-open-set-delection/scripts/run_detection.py",
                "--images",
                str(gen_dir.resolve()),
                "--label",
                target_label,
                "--base-url",
                args.qwen_base_url,
                "--out",
                str(qwen_gen_tmp),
            ]
        )

        with open(qwen_orig_tmp, "r", encoding="utf-8") as f:
            qwen_orig_result = json.load(f)
        with open(qwen_gen_tmp, "r", encoding="utf-8") as f:
            qwen_gen_result = json.load(f)

        qwen_results = qwen_orig_result.get("results", []) + qwen_gen_result.get("results", [])
        if len(qwen_results) != len(image_paths):
            raise RuntimeError(
                f"Qwen 结果条数 ({len(qwen_results)}) 与图片数 ({len(image_paths)}) 不一致。"
            )

        run(
            [
                "python3",
                "skills/rexomni-open-set-detection/scripts/run_detection.py",
                "--images",
                str(orig_image),
                "--prompt",
                str(rex_prompt_path),
                "--base-url",
                args.rex_base_url,
                "--out",
                str(rex_orig_tmp),
            ]
        )
        run(
            [
                "python3",
                "skills/rexomni-open-set-detection/scripts/run_detection.py",
                "--images",
                str(gen_dir.resolve()),
                "--prompt",
                str(rex_prompt_path),
                "--base-url",
                args.rex_base_url,
                "--out",
                str(rex_gen_tmp),
            ]
        )

        with open(rex_orig_tmp, "r", encoding="utf-8") as f:
            rex_orig_coco = json.load(f)
        with open(rex_gen_tmp, "r", encoding="utf-8") as f:
            rex_gen_coco = json.load(f)

        rex_orig_by_id = coco_to_pred_bboxes_by_image_id(rex_orig_coco)
        rex_gen_by_id = coco_to_pred_bboxes_by_image_id(rex_gen_coco)

        rex_boxes_by_idx: list[list[list[float]]] = []
        rex_boxes_by_idx.append(rex_orig_by_id.get(0, []))
        for i in range(len(gen_files)):
            rex_boxes_by_idx.append(rex_gen_by_id.get(i, []))
    finally:
        for p in (qwen_orig_tmp, qwen_gen_tmp, rex_orig_tmp, rex_gen_tmp, rex_prompt_path):
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    pred_records: list[dict] = []
    for i, item in enumerate(qwen_results):
        name = Path(item["image"]).name
        qwen_boxes = [x["bbox"] for x in item.get("bboxes", []) if isinstance(x, dict) and "bbox" in x]
        rex_boxes = rex_boxes_by_idx[i] if i < len(rex_boxes_by_idx) else []
        pred_records.append(
            {
                "image": name,
                "source": "original" if i == 0 else "generated",
                "image_idx": i,
                "models": [
                    {"model": "qwen3-vl-8b", "pred_bboxes": qwen_boxes},
                    {"model": "rex-omni", "pred_bboxes": rex_boxes},
                ],
            }
        )

    with open(prediction_json, "w", encoding="utf-8") as f:
        json.dump(pred_records, f, indent=2, ensure_ascii=False)

    viz_dir = ws / "annotated_images"
    draw_detection_overlays(image_map, prediction_json, viz_dir)

    run(
        [
            "python3",
            "skills/eval-reports-generation/scripts/run_eval_report_generation.py",
            "--images",
            json.dumps(images_abs, ensure_ascii=False),
            "--prediction",
            str(prediction_json),
            "--task-text",
            args.text,
            "--target-label",
            target_label,
            "--api-base",
            llm_api_base,
            "--api-key",
            llm_api_key,
            "--out",
            str(evaluation_json),
        ]
    )

    print("Pipeline done.")
    print(f"Intent: {intent_json}")
    print(f"Prompts: {prompts_json}")
    print(f"Generated list: {gen_json}")
    print(f"Annotated: {viz_dir}")
    print(f"Prediction: {prediction_json}")
    print(f"Evaluation: {evaluation_json}")


if __name__ == "__main__":
    main()
