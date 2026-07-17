#!/usr/bin/env python3
"""Replace a frozen set of Planner prediction rows after runtime-only retries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: expected an object")
        rows.append(value)
    return rows


def case_id(row: dict[str, Any], *, source: Path) -> str:
    value = str(row.get("case_id") or "").strip()
    if not value:
        raise ValueError(f"{source}: prediction row is missing case_id")
    return value


def runtime_error_count(row: dict[str, Any]) -> int:
    errors = row.get("errors")
    count = len(errors) if isinstance(errors, list) else int(bool(errors))
    decisions = row.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        return count + 1
    for decision in decisions:
        if not isinstance(decision, dict):
            count += 1
            continue
        metrics = decision.get("_planner_metrics")
        metrics = metrics if isinstance(metrics, dict) else {}
        count += int(bool(str(metrics.get("error_type") or "").strip()))
        count += int(bool(str(metrics.get("error") or "").strip()))
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--replacement", type=Path, nargs="+", required=True)
    parser.add_argument("--expected-replacements", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_rows = load_jsonl(args.base)
    base_ids = [case_id(row, source=args.base) for row in base_rows]
    if len(base_ids) != len(set(base_ids)):
        raise ValueError("base predictions contain duplicate case IDs")

    replacements: dict[str, dict[str, Any]] = {}
    replacement_sources: dict[str, str] = {}
    for path in args.replacement:
        for row in load_jsonl(path):
            cid = case_id(row, source=path)
            if cid in replacements:
                raise ValueError(f"duplicate replacement case ID: {cid}")
            if cid not in set(base_ids):
                raise ValueError(f"replacement case is absent from base predictions: {cid}")
            errors = runtime_error_count(row)
            if errors:
                raise ValueError(f"replacement {cid} has {errors} runtime errors")
            replacements[cid] = row
            replacement_sources[cid] = str(path)

    if len(replacements) != args.expected_replacements:
        raise ValueError(
            f"expected {args.expected_replacements} replacements, found {len(replacements)}"
        )
    output_rows = [replacements.get(cid, row) for cid, row in zip(base_ids, base_rows)]
    output_text = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output_text, encoding="utf-8")
    print(
        json.dumps(
            {
                "base_rows": len(base_rows),
                "replacement_rows": len(replacements),
                "replacement_case_ids": sorted(replacements),
                "replacement_sources": dict(sorted(replacement_sources.items())),
                "output": str(args.output),
                "output_sha256": hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
