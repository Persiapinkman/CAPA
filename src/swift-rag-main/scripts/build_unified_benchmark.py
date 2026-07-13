#!/usr/bin/env python3
"""Build a unified single-step benchmark from document/table/adela sources.

This script keeps compatibility with existing benchmark files while adding
source_type-aware questions for unified gateway evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DOC_QTYPE_MAP = {
    "release_note": "发布时间",
    "io_spec": "输入输出",
    "threshold": "阈值",
    "boundary": "输入约束",
    "metric": "评测指标",
}


def _normalize_text(value: Any) -> str:
    """Normalize text to single-line content for JSONL/TXT/CSV compatibility."""
    if value is None:
        return ""
    text = str(value)
    # Collapse line breaks/tabs and repeated spaces.
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return " ".join(text.split())


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _valid_text(v: Any) -> bool:
    if v is None:
        return False
    s = _normalize_text(v)
    if not s:
        return False
    return s.lower() not in {"nan", "null", "none", "无", "-"}


def _pick(rows: List[Dict[str, Any]], k: int, rng: random.Random) -> List[Dict[str, Any]]:
    if k <= 0:
        return []
    if k >= len(rows):
        return list(rows)
    return rng.sample(rows, k)


def build_document_items(rows: List[Dict[str, Any]], count: int, rng: random.Random) -> List[Dict[str, Any]]:
    # If mixed benchmark is passed in, only keep document-origin rows.
    document_rows = [
        r for r in rows
        if (r.get("source_type") in (None, "", "document"))
    ]
    if not document_rows:
        document_rows = rows

    picked = _pick(document_rows, count, rng)
    output: List[Dict[str, Any]] = []
    for row in picked:
        item = dict(row)
        item["question"] = _normalize_text(item.get("question", ""))
        item["reference_answer"] = _normalize_text(item.get("reference_answer", ""))
        item["evidence"] = _normalize_text(item.get("evidence", ""))
        item["source_type"] = "document"
        item["source_id"] = _normalize_text(row.get("source_doc"))
        item["source_category"] = DOC_QTYPE_MAP.get(row.get("question_type", ""), "文档问答")
        item["retrieval_source_types"] = ["document"]
        output.append(item)
    return output


def _table_question(row: Dict[str, Any], template_id: int) -> Tuple[str, str, str, List[str]]:
    model_name = _normalize_text(row.get("model_name", ""))
    if template_id == 0 and _valid_text(row.get("owner")):
        answer = _normalize_text(row["owner"])
        q = f"{model_name} 这个模型当前负责人是谁？"
        ev = f"owner: {answer}"
        return q, answer, ev, [answer]
    if template_id == 1 and _valid_text(row.get("supported_device")):
        answer = _normalize_text(row["supported_device"])
        q = f"{model_name} 支持部署在哪些设备上（按记录回答）？"
        ev = f"supported_device: {answer}"
        return q, answer, ev, [answer]
    if template_id == 2 and _valid_text(row.get("recommended_config")):
        answer = _normalize_text(row["recommended_config"])
        q = f"{model_name} 推荐配置是什么？"
        ev = f"recommended_config: {answer}"
        return q, answer, ev, [answer]
    if template_id == 3 and _valid_text(row.get("last_updated")):
        answer = _normalize_text(row["last_updated"])
        q = f"{model_name} 最近一次更新时间是什么时候？"
        ev = f"last_updated: {answer}"
        return q, answer, ev, [answer]

    # fallback
    answer = _normalize_text(row.get("algorithm_name") or row.get("application_scene") or "")
    q = f"{model_name} 对应的算法名称是什么？"
    ev = f"algorithm_name: {answer}"
    return q, answer, ev, [answer] if answer else []


def build_table_items(rows: List[Dict[str, Any]], count: int, rng: random.Random) -> List[Dict[str, Any]]:
    valid_rows = [r for r in rows if _valid_text(r.get("model_name"))]
    picked = _pick(valid_rows, count, rng)

    output: List[Dict[str, Any]] = []
    for idx, row in enumerate(picked):
        q, answer, evidence, keywords = _table_question(row, idx % 4)
        output.append(
            {
                "question": q,
                "reference_answer": answer,
                "source_doc": "model_release_records.jsonl",
                "source_page": row.get("source_row_number") or 1,
                "evidence": evidence,
                "question_type": "table_lookup",
                "difficulty": "easy",
                "expected_keywords": keywords,
                "source_type": "table",
                "source_id": row.get("row_id"),
                "source_category": "表格字段",
                "retrieval_source_types": ["table"],
            }
        )
    return output


def _adela_question(row: Dict[str, Any], template_id: int) -> Tuple[str, str, str, List[str]]:
    model = _normalize_text(row.get("model_name") or row.get("name") or "")
    if template_id == 0 and row.get("did") is not None:
        answer = _normalize_text(row["did"])
        q = f"{model} 这条部署记录对应的 did 是多少？"
        ev = f"did: {answer}"
        return q, answer, ev, [answer]
    if template_id == 1 and row.get("rid") is not None:
        answer = _normalize_text(row["rid"])
        q = f"{model} 这条部署记录对应的 rid 是多少？"
        ev = f"rid: {answer}"
        return q, answer, ev, [answer]
    if template_id == 2 and _valid_text(row.get("platform")):
        answer = _normalize_text(row["platform"])
        q = f"{model} 在这条 adela 记录中的部署平台是什么？"
        ev = f"platform: {answer}"
        return q, answer, ev, [answer]
    if template_id == 3 and _valid_text(row.get("version")):
        answer = _normalize_text(row["version"])
        q = f"{model} 在这条 adela 记录中的部署版本号是多少？"
        ev = f"version: {answer}"
        return q, answer, ev, [answer]

    answer = _normalize_text(row.get("status") or "")
    q = f"{model} 在 adela 的部署状态是什么？"
    ev = f"status: {answer}"
    return q, answer, ev, [answer] if answer else []


def build_adela_items(rows: List[Dict[str, Any]], count: int, rng: random.Random) -> List[Dict[str, Any]]:
    valid_rows = [r for r in rows if _valid_text(r.get("model_name") or r.get("name"))]
    picked = _pick(valid_rows, count, rng)

    output: List[Dict[str, Any]] = []
    for idx, row in enumerate(picked):
        q, answer, evidence, keywords = _adela_question(row, idx % 4)
        output.append(
            {
                "question": q,
                "reference_answer": answer,
                "source_doc": "adela_release_records.jsonl",
                "source_page": 1,
                "evidence": evidence,
                "question_type": "adela_lookup",
                "difficulty": "easy",
                "expected_keywords": keywords,
                "source_type": "adela",
                "source_id": row.get("row_id"),
                "source_category": "部署记录",
                "retrieval_source_types": ["adela"],
            }
        )
    return output


def to_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "问题",
                "标准答案",
                "问题来源段落截取",
                "问题文件来源",
                "来源页码",
                "问题类型",
                "难度",
                "来源类型",
                "来源ID",
                "检索源",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "问题": row["question"],
                    "标准答案": row["reference_answer"],
                    "问题来源段落截取": row.get("evidence", ""),
                    "问题文件来源": row.get("source_doc", ""),
                    "来源页码": row.get("source_page", 1),
                    "问题类型": row.get("question_type", ""),
                    "难度": row.get("difficulty", "easy"),
                    "来源类型": row.get("source_type", ""),
                    "来源ID": row.get("source_id", ""),
                    "检索源": ",".join(row.get("retrieval_source_types", [])),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 unified 三源 benchmark（document/table/adela）")
    parser.add_argument("--out-dir", default="benchmark/single_step_rag_100_v1", help="输出目录")
    parser.add_argument(
        "--doc-benchmark-jsonl",
        default="benchmark/single_step_rag_100_v1/benchmark_100.jsonl",
        help="文档题库来源（已有 JSONL）",
    )
    parser.add_argument(
        "--table-jsonl",
        default="data_source/tables/model_release_records.jsonl",
        help="table 数据源 JSONL",
    )
    parser.add_argument(
        "--adela-jsonl",
        default="data_source/adela/adela_release_records.jsonl",
        help="adela 数据源 JSONL",
    )
    parser.add_argument("--total", type=int, default=100, help="总题量")
    parser.add_argument("--doc-count", type=int, default=34, help="document 题量")
    parser.add_argument("--table-count", type=int, default=33, help="table 题量")
    parser.add_argument("--adela-count", type=int, default=33, help="adela 题量")
    parser.add_argument("--seed", type=int, default=42, help="采样随机种子")
    args = parser.parse_args()

    if args.doc_count + args.table_count + args.adela_count != args.total:
        raise ValueError("doc/table/adela 题量之和必须等于 total")

    rng = random.Random(args.seed)

    doc_rows = read_jsonl(Path(args.doc_benchmark_jsonl))
    table_rows = read_jsonl(Path(args.table_jsonl))
    adela_rows = read_jsonl(Path(args.adela_jsonl))

    doc_items = build_document_items(doc_rows, args.doc_count, rng)
    table_items = build_table_items(table_rows, args.table_count, rng)
    adela_items = build_adela_items(adela_rows, args.adela_count, rng)

    if len(doc_items) != args.doc_count:
        raise ValueError(f"document 题量不足: 期望 {args.doc_count}, 实际 {len(doc_items)}")
    if len(table_items) != args.table_count:
        raise ValueError(f"table 题量不足: 期望 {args.table_count}, 实际 {len(table_items)}")
    if len(adela_items) != args.adela_count:
        raise ValueError(f"adela 题量不足: 期望 {args.adela_count}, 实际 {len(adela_items)}")

    all_items = doc_items + table_items + adela_items
    rng.shuffle(all_items)

    for i, row in enumerate(all_items, start=1):
        row["id"] = f"ssr100-{i:03d}"

    out_dir = Path(args.out_dir)
    jsonl_out = out_dir / "benchmark_100.jsonl"
    txt_out = out_dir / "questions.txt"
    csv_out = out_dir / "benchmark_100_for_eval.csv"

    write_jsonl(jsonl_out, all_items)

    txt_out.parent.mkdir(parents=True, exist_ok=True)
    txt_out.write_text("\n".join(r["question"] for r in all_items) + "\n", encoding="utf-8")

    to_csv(all_items, csv_out)

    source_counter = {"document": 0, "table": 0, "adela": 0}
    for r in all_items:
        source_counter[r.get("source_type", "")] = source_counter.get(r.get("source_type", ""), 0) + 1

    print("构建完成:")
    print(f"- 输出目录: {out_dir}")
    print(f"- 题目总量: {len(all_items)}")
    print(f"- 来源分布: {source_counter}")


if __name__ == "__main__":
    main()
