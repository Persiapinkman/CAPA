#!/usr/bin/env python3
"""Build a balanced step-level subset for GRPO stochastic-support auditing."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_rows(
    rows: list[dict[str, Any]],
    categories: list[str],
    entities_per_category: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for category in categories:
        category_rows = [row for row in rows if str(row.get("category") or "") == category]
        entity_order: list[str] = []
        for row in category_rows:
            entity = str(row.get("entity_id") or "")
            if entity and entity not in entity_order:
                entity_order.append(entity)
        chosen = set(entity_order[:entities_per_category])
        if len(chosen) < entities_per_category:
            raise ValueError(
                f"{category}: requested {entities_per_category} entities, found {len(chosen)}"
            )
        selected.extend(row for row in category_rows if str(row.get("entity_id") or "") in chosen)
    keys = [(str(row.get("case_id") or ""), int(row.get("step_index") or 0)) for row in selected]
    if len(keys) != len(set(keys)):
        raise ValueError("selected step rows overlap")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--accepted-scenarios", type=Path, required=True)
    parser.add_argument("--entities-per-category", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metadata-out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.entities_per_category < 1:
        raise ValueError("--entities-per-category must be positive")
    categories = [
        line.strip()
        for line in args.accepted_scenarios.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not categories or len(categories) != len(set(categories)):
        raise ValueError("accepted scenario file must contain unique non-empty names")
    rows = load_jsonl(args.data)
    selected = select_rows(rows, categories, args.entities_per_category)
    write_jsonl(args.out, selected)
    payload = {
        "schema_version": "1.0",
        "source": str(args.data.resolve()),
        "source_sha256": sha256(args.data),
        "accepted_scenarios": categories,
        "entities_per_category": args.entities_per_category,
        "rows": len(selected),
        "cases": len({str(row.get("case_id") or "") for row in selected}),
        "entities": len({str(row.get("entity_id") or "") for row in selected}),
        "categories": dict(sorted(Counter(str(row.get("category") or "") for row in selected).items())),
        "output": str(args.out.resolve()),
        "output_sha256": sha256(args.out),
    }
    metadata_path = args.metadata_out or args.out.with_suffix(".metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
