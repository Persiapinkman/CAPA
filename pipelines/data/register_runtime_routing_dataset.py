#!/usr/bin/env python3
"""Audit and register a demo-runtime Planner routing dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from capa.evaluation.dataset_audit import (  # noqa: E402
    case_stats,
    load_jsonl,
    nearest_same_category_similarity,
    overlap,
    step_stats,
)
from capa.experiments.registry import sha256_file  # noqa: E402


DATASET_ID = "planner_runtime_routing_v1"
BASE = Path("training/planner_grpo_seed_v1")
DATASET_LAYOUTS = {
    "planner_runtime_routing_v1": "sft_data_runtime_routing_v1_chatml",
    "planner_runtime_probe_curriculum_v1": (
        "sft_data_runtime_probe_curriculum_v1_chatml"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    for split in ("train", "dev", "test"):
        parser.add_argument(
            f"--{split}-cases",
            type=Path,
            default=None,
        )
        parser.add_argument(
            f"--{split}-steps",
            type=Path,
            default=None,
        )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _default_paths(dataset_id: str, split: str) -> tuple[Path, Path]:
    try:
        step_directory = DATASET_LAYOUTS[dataset_id]
    except KeyError as exc:
        supported = ", ".join(sorted(DATASET_LAYOUTS))
        raise ValueError(
            f"unknown runtime dataset {dataset_id!r}; supported: {supported}"
        ) from exc
    return (
        BASE / f"cases/{dataset_id}_{split}_cases.jsonl",
        BASE / step_directory / f"{split}.jsonl",
    )


def _metadata_overlap(
    left: list[dict[str, Any]], right: list[dict[str, Any]], key: str
) -> int:
    return len(
        {str(row.get(key) or "") for row in left}
        & {str(row.get(key) or "") for row in right}
    )


def _render_card(manifest: dict[str, Any]) -> str:
    splits = manifest["splits"]
    return "\n".join(
        [
            f"# Dataset Card: {manifest['dataset_id']}",
            "",
            "## 研究问题",
            "",
            "GRPO 是否能改善 Demo Agent 运行时真正交给 Planner 的动作：首步路由、参数、结束标志，以及视觉探针后的迁移决策？",
            "",
            "## 数据规模",
            "",
            "| Split | Entity groups | Cases | Steps |",
            "|---|---:|---:|---:|",
            *[
                f"| {split} | {splits[split]['entity_groups']} | "
                f"{splits[split]['cases']['cases']} | {splits[split]['steps']['rows']} |"
                for split in ("train", "dev", "test")
            ],
            "",
            "## 场景",
            "",
            "覆盖九个工具、`clarify`、`end`、Qwen/Rex 单图探针、完整评测、Flux、Adela，以及探针后继续迁移顾问的两步路径。",
            "主对照是 `qwen_probe_then_migration` 与 `qwen_probe_only_contrast`：两者首步工具相同，但 `finish_after_tool` 必须不同。",
            "",
            "## 来源与隐私",
            "",
            "数据依据 `demo/sessions` 与 `demo/llm_debug` 的聚合动作分布、三轮 miss 结构和请求类型合成。",
            "未复制原始用户 query、回答、客户端地址、session ID、模型资产 ID 或 RAG 文档内容。",
            "所有项目名、模型名和实体均为合成值；图片只使用仓库 fixtures。",
            "",
            "## 完整性",
            "",
            "train/dev/test 的 entity、case ID、精确 query 与 template ID 均不重叠。",
            "Test 在开发门预注册后保持封存，只有固定候选通过开发门才可打开一次。",
            "同一实体下的不同场景相关，统计必须按 `entity_id` 聚类。",
            "",
            "## 边界",
            "",
            "这是路由策略评测，不执行 Flux、Adela 或完整 pipeline，因此不能证明外部服务效果。",
            "RAG miss 后的强制改写与重试由编排器控制，不属于本数据的 GRPO 主张范围。",
            "",
            "人工审阅顺序、实际 train 样例和拒绝条件见 `HUMAN_REVIEW.md`，哈希与分布见 `manifest.json`。",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    paths: dict[str, Path] = {}
    cases: dict[str, list[dict[str, Any]]] = {}
    steps: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "dev", "test"):
        default_case_path, default_step_path = _default_paths(args.dataset_id, split)
        case_path = _resolve(
            getattr(args, f"{split}_cases") or default_case_path
        )
        step_path = _resolve(
            getattr(args, f"{split}_steps") or default_step_path
        )
        paths[f"{split}_cases"] = case_path
        paths[f"{split}_steps"] = step_path
        cases[split] = load_jsonl(case_path)
        steps[split] = load_jsonl(step_path)

    integrity: dict[str, Any] = {"status": "sealed_test_created_unopened"}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        integrity[f"{left}_{right}"] = {
            **overlap(cases[left], cases[right]),
            "entity_overlap": _metadata_overlap(
                cases[left], cases[right], "entity_id"
            ),
            "template_overlap": _metadata_overlap(
                cases[left], cases[right], "template_id"
            ),
        }
    integrity["dev_similarity_to_train"] = nearest_same_category_similarity(
        cases["train"], cases["dev"]
    )
    integrity["test_similarity_to_train"] = nearest_same_category_similarity(
        cases["train"], cases["test"]
    )

    missing_fixtures = sorted(
        {
            str((row.get("setup") or {}).get("image_fixture") or "")
            for split_rows in cases.values()
            for row in split_rows
            if (row.get("setup") or {}).get("has_image")
            and not _resolve(Path(str((row.get("setup") or {}).get("image_fixture")))).is_file()
        }
    )
    manifest = {
        "schema_version": "1.0",
        "dataset_id": args.dataset_id,
        "role": "train_dev_and_sealed_test",
        "experimental_unit": "case_id",
        "bootstrap_cluster": "entity_id",
        "description": (
            "Runtime-owned Planner routing and two-step probe-to-migration benchmark."
        ),
        "provenance": {
            "class": "deidentified_synthetic_from_demo_patterns",
            "source_records": ["demo/sessions", "demo/llm_debug"],
            "copies_user_text": False,
            "copies_answers": False,
            "copies_identifiers": False,
        },
        "stats": {
            "cases": sum(len(cases[split]) for split in ("train", "dev", "test")),
            "steps": sum(len(steps[split]) for split in ("train", "dev", "test")),
            "entity_groups": sum(
                len({str(row.get("entity_id") or "") for row in cases[split]})
                for split in ("train", "dev", "test")
            ),
        },
        "splits": {
            split: {
                "cases": case_stats(cases[split]),
                "steps": step_stats(steps[split]),
                "entity_groups": len(
                    {str(row.get("entity_id") or "") for row in cases[split]}
                ),
                "template_ids": sorted(
                    {str(row.get("template_id") or "") for row in cases[split]}
                ),
            }
            for split in ("train", "dev", "test")
        },
        "files": {
            key: str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
            for key, path in paths.items()
        },
        "sha256": {key: sha256_file(path) for key, path in paths.items()},
        "integrity": {**integrity, "missing_image_fixtures": missing_fixtures},
    }
    if missing_fixtures:
        raise RuntimeError(f"missing image fixtures: {missing_fixtures}")
    output_dir = _resolve(
        args.output_dir or Path(f"data/datasets/{args.dataset_id}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "DATASET_CARD.md").write_text(
        _render_card(manifest), encoding="utf-8"
    )
    print(json.dumps({"status": "registered", "dataset_id": args.dataset_id}))


if __name__ == "__main__":
    main()
