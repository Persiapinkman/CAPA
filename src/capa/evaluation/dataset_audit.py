"""Dataset integrity summaries for Planner case and step JSONL files."""

from __future__ import annotations

import difflib
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

from capa.experiments.registry import sha256_file


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected object")
        rows.append(value)
    return rows


def normalize_query(value: str) -> str:
    text = str(value or "").lower()
    return re.sub(r"[\s，。！？、；：,.!?;:'\"“”‘’（）()\[\]{}]", "", text)


def case_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    categories = Counter(str(row.get("category") or "unknown") for row in rows)
    actions: Counter[str] = Counter()
    steps = 0
    for row in rows:
        expected = row.get("expected_decisions") if isinstance(row.get("expected_decisions"), list) else []
        steps += len(expected)
        for decision in expected:
            if not isinstance(decision, dict):
                continue
            decision_type = str(decision.get("decision_type") or "tool")
            action = decision_type if decision_type in {"clarify", "end"} else decision.get("action")
            actions[str(action or "unknown")] += 1
    normalized_queries = [normalize_query(str(row.get("user_query") or "")) for row in rows]
    return {
        "cases": len(rows),
        "steps": steps,
        "unique_case_ids": len({str(row.get("case_id") or "") for row in rows}),
        "unique_queries": len(set(normalized_queries)),
        "categories": dict(sorted(categories.items())),
        "expected_actions": dict(sorted(actions.items())),
    }


def step_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    serialized = [
        (str(row.get("prompt") or ""), str(row.get("completion") or ""))
        for row in rows
    ]
    case_counts = Counter(str(row.get("case_id") or "") for row in rows)
    return {
        "rows": len(rows),
        "cases": len(case_counts),
        "exact_prompt_completion_duplicates": len(rows) - len(set(serialized)),
        "duplicate_rate": (len(rows) - len(set(serialized))) / len(rows) if rows else 0.0,
        "max_rows_per_case": max(case_counts.values(), default=0),
        "categories": dict(sorted(Counter(str(row.get("category") or "unknown") for row in rows).items())),
    }


def overlap(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, int]:
    left_ids = {str(row.get("case_id") or "") for row in left}
    right_ids = {str(row.get("case_id") or "") for row in right}
    left_queries = {normalize_query(str(row.get("user_query") or "")) for row in left}
    right_queries = {normalize_query(str(row.get("user_query") or "")) for row in right}
    return {
        "case_id_overlap": len(left_ids & right_ids),
        "exact_query_overlap": len(left_queries & right_queries),
    }


def nearest_same_category_similarity(
    train: list[dict[str, Any]], dev: list[dict[str, Any]]
) -> dict[str, Any]:
    values: list[float] = []
    examples: list[dict[str, Any]] = []
    for dev_row in dev:
        candidates = [row for row in train if row.get("category") == dev_row.get("category")]
        if not candidates:
            continue
        dev_query = normalize_query(str(dev_row.get("user_query") or ""))
        score, nearest = max(
            (
                (
                    difflib.SequenceMatcher(
                        None,
                        dev_query,
                        normalize_query(str(train_row.get("user_query") or "")),
                    ).ratio(),
                    train_row,
                )
                for train_row in candidates
            ),
            key=lambda item: item[0],
        )
        values.append(score)
        examples.append(
            {
                "dev_case_id": dev_row.get("case_id"),
                "train_case_id": nearest.get("case_id"),
                "category": dev_row.get("category"),
                "similarity": round(score, 6),
            }
        )
    examples.sort(key=lambda item: float(item["similarity"]), reverse=True)
    return {
        "count": len(values),
        "mean": mean(values) if values else 0.0,
        "median": median(values) if values else 0.0,
        "count_ge_0_8": sum(value >= 0.8 for value in values),
        "count_ge_0_9": sum(value >= 0.9 for value in values),
        "top_examples": examples[:10],
        "method": "difflib.SequenceMatcher on normalized query text; diagnostic only",
    }


def build_manifest(
    *,
    root: Path,
    dataset_id: str,
    source_cases: Path,
    train_cases: Path,
    dev_cases: Path,
    regression_cases: Path,
    train_steps: Path,
    dev_steps: Path,
    hard_v4_steps: Path,
    hard_v5_steps: Path,
) -> dict[str, Any]:
    paths = {
        "source_cases": source_cases,
        "train_cases": train_cases,
        "dev_cases": dev_cases,
        "regression_cases": regression_cases,
        "train_steps": train_steps,
        "dev_steps": dev_steps,
        "hard_v4_steps": hard_v4_steps,
        "hard_v5_steps": hard_v5_steps,
    }
    resolved = {key: path if path.is_absolute() else root / path for key, path in paths.items()}
    loaded = {key: load_jsonl(path) for key, path in resolved.items()}
    train_dev_overlap = overlap(loaded["train_cases"], loaded["dev_cases"])
    train_regression_overlap = overlap(loaded["train_cases"], loaded["regression_cases"])
    issues = [
        "The dev split has been reused for model selection and is not a sealed test set.",
        "Case-level grouping prevents exact ID leakage, but template families span train and dev.",
        f"The regression set overlaps train by {train_regression_overlap['case_id_overlap']} case IDs.",
    ]
    return {
        "schema_version": "1.0",
        "dataset_id": dataset_id,
        "role": "development_train_dev_and_regression",
        "description": "Focused CAPA Planner routing cases and derived ChatML step data.",
        "stats": case_stats(loaded["source_cases"]),
        "splits": {
            "train_cases": case_stats(loaded["train_cases"]),
            "dev_cases": case_stats(loaded["dev_cases"]),
            "regression_cases": case_stats(loaded["regression_cases"]),
            "train_steps": step_stats(loaded["train_steps"]),
            "dev_steps": step_stats(loaded["dev_steps"]),
            "hard_v4_steps": step_stats(loaded["hard_v4_steps"]),
            "hard_v5_steps": step_stats(loaded["hard_v5_steps"]),
        },
        "files": {
            key: str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
            for key, path in resolved.items()
        },
        "sha256": {key: sha256_file(path) for key, path in resolved.items()},
        "integrity": {
            "status": "development_only_not_sealed",
            "train_dev_overlap": train_dev_overlap,
            "train_regression_overlap": train_regression_overlap,
            "train_dev_similarity": nearest_same_category_similarity(
                loaded["train_cases"], loaded["dev_cases"]
            ),
            "issues": issues,
        },
    }
