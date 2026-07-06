#!/usr/bin/env python3
"""Apply human review CSV to planner DPO train-seed pairs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_review(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = str(row.get("case_id") or "").strip()
            if cid:
                out[cid] = row
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply review CSV to DPO train seed.")
    parser.add_argument("--pairs", required=True, help="Input planner_dpo_train_seed_pairs.jsonl")
    parser.add_argument("--review", required=True, help="Input planner_dpo_train_seed_review.csv")
    parser.add_argument("--out", required=True, help="Output approved JSONL")
    parser.add_argument("--allow-todo", action="store_true", help="Treat todo as approve for smoke experiments")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs_path = Path(args.pairs)
    review_path = Path(args.review)
    out_path = Path(args.out)
    pairs = load_jsonl(pairs_path)
    review = load_review(review_path)
    approved: list[dict[str, Any]] = []
    counts = {"approve": 0, "reject": 0, "fix": 0, "todo": 0, "missing": 0}
    for pair in pairs:
        meta = pair.get("meta") if isinstance(pair.get("meta"), dict) else {}
        cid = str(meta.get("case_id") or "").strip()
        row = review.get(cid)
        status = str((row or {}).get("review_status") or "missing").strip().lower()
        if status not in counts:
            status = "missing"
        counts[status] += 1
        meta["human_review_status"] = status
        meta["human_review_note"] = str((row or {}).get("reviewer_note") or "").strip()
        pair["meta"] = meta
        if status == "approve" or (status == "todo" and args.allow_todo):
            approved.append(pair)
    write_jsonl(out_path, approved)
    print(
        "Applied DPO train seed review:",
        f"approved={len(approved)}",
        f"counts={json.dumps(counts, ensure_ascii=False)}",
        f"out={out_path}",
    )


if __name__ == "__main__":
    main()
