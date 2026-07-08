#!/usr/bin/env python3
"""Prepare reviewed Planner DPO pairs for common DPO trainers.

Outputs:
- chat JSONL: {"messages": [...], "chosen": "...", "rejected": "...", "meta": {...}}
- text JSONL: {"prompt": "...", "chosen": "...", "rejected": "...", "meta": {...}}
- train/val splits for both formats.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
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
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _ensure_json_string(value: Any, *, field: str, row_id: str) -> str:
    if isinstance(value, str):
        text = value.strip()
        json.loads(text)
        return text
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    raise ValueError(f"{row_id}: {field} must be JSON string or object")


def _prompt_to_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "").strip()
        parts.append(f"<|{role}|>\n{content}")
    return "\n\n".join(parts).strip() + "\n\n<|assistant|>\n"


def normalize_pair(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    row_id = str(meta.get("case_id") or "unknown_case")
    if str(meta.get("human_review_status") or "").strip() != "approve":
        raise ValueError(f"{row_id}: only approved rows can be prepared for training")

    messages = row.get("prompt")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{row_id}: prompt must be a non-empty messages list")
    normalized_messages: list[dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            raise ValueError(f"{row_id}: each prompt item must be an object")
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "").strip()
        if role not in {"system", "user", "assistant"} or not content:
            raise ValueError(f"{row_id}: invalid prompt message")
        normalized_messages.append({"role": role, "content": content})

    chosen = _ensure_json_string(row.get("chosen"), field="chosen", row_id=row_id)
    rejected = _ensure_json_string(row.get("rejected"), field="rejected", row_id=row_id)
    for field, value in {"chosen": chosen, "rejected": rejected}.items():
        decision = json.loads(value)
        if not isinstance(decision, dict):
            raise ValueError(f"{row_id}: {field} must decode to object")
        if str(decision.get("decision_type") or "") != "tool":
            raise ValueError(f"{row_id}: {field} must be a tool decision")
        if not str(decision.get("action") or "").strip():
            raise ValueError(f"{row_id}: {field} missing action")

    compact_meta = {
        "case_id": row_id,
        "category": str(meta.get("category") or ""),
        "error_type": str(meta.get("error_type") or ""),
        "user_query": str(meta.get("user_query") or ""),
        "human_review_status": str(meta.get("human_review_status") or ""),
        "human_review_note": str(meta.get("human_review_note") or ""),
    }
    chat_row = {
        "messages": normalized_messages,
        "chosen": chosen,
        "rejected": rejected,
        "meta": compact_meta,
    }
    text_row = {
        "prompt": _prompt_to_text(normalized_messages),
        "chosen": chosen,
        "rejected": rejected,
        "meta": compact_meta,
    }
    return chat_row, text_row


def stratified_split(rows: list[dict[str, Any]], *, val_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        error_type = str((row.get("meta") or {}).get("error_type") or "unknown")
        by_type.setdefault(error_type, []).append(row)
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for group in by_type.values():
        shuffled = list(group)
        rng.shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Planner DPO training data.")
    parser.add_argument("--input", required=True, help="Reviewed approved JSONL")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Split random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.input)
    out_dir = Path(args.out_dir)
    rows = load_jsonl(src)
    chat_rows: list[dict[str, Any]] = []
    text_rows: list[dict[str, Any]] = []
    for row in rows:
        chat_row, text_row = normalize_pair(row)
        chat_rows.append(chat_row)
        text_rows.append(text_row)

    val_ratio = min(0.5, max(0.0, float(args.val_ratio)))
    chat_train, chat_val = stratified_split(chat_rows, val_ratio=val_ratio, seed=int(args.seed))
    text_train, text_val = stratified_split(text_rows, val_ratio=val_ratio, seed=int(args.seed))

    write_jsonl(out_dir / "planner_dpo_chat.jsonl", chat_rows)
    write_jsonl(out_dir / "planner_dpo_text.jsonl", text_rows)
    write_jsonl(out_dir / "planner_dpo_chat_train.jsonl", chat_train)
    write_jsonl(out_dir / "planner_dpo_chat_val.jsonl", chat_val)
    write_jsonl(out_dir / "planner_dpo_text_train.jsonl", text_train)
    write_jsonl(out_dir / "planner_dpo_text_val.jsonl", text_val)

    report = {
        "source": str(src.resolve()),
        "total": len(chat_rows),
        "train": len(chat_train),
        "val": len(chat_val),
        "val_ratio": val_ratio,
        "seed": int(args.seed),
        "by_error_type": dict(Counter(row["meta"]["error_type"] for row in chat_rows)),
        "outputs": {
            "chat": str((out_dir / "planner_dpo_chat.jsonl").resolve()),
            "text": str((out_dir / "planner_dpo_text.jsonl").resolve()),
            "chat_train": str((out_dir / "planner_dpo_chat_train.jsonl").resolve()),
            "chat_val": str((out_dir / "planner_dpo_chat_val.jsonl").resolve()),
            "text_train": str((out_dir / "planner_dpo_text_train.jsonl").resolve()),
            "text_val": str((out_dir / "planner_dpo_text_val.jsonl").resolve()),
        },
    }
    (out_dir / "planner_dpo_training_data_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "Prepared Planner DPO training data:",
        f"total={len(chat_rows)}",
        f"train={len(chat_train)}",
        f"val={len(chat_val)}",
        f"out={out_dir}",
    )


if __name__ == "__main__":
    main()
