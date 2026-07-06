#!/usr/bin/env python3
"""从 demo/sessions 抽取去重用户问题，生成评测集（ground_truth 留空）。"""
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SESSIONS_DIR = ROOT / "demo" / "sessions"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent

SKIP_EXACT = {
    "",
    "你是谁",
    "你好",
    "hi",
    "hello",
    "测试",
    "test",
    "迁移顾问",
    "继续",
    "取消",
    "ok",
    "好的",
}


def normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def norm_key(text: str) -> str:
    return normalize_question(text).casefold()


def _cjk_count(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text))


def should_skip(text: str, *, is_clarification_reply: bool) -> bool:
    t = normalize_question(text)
    if not t:
        return True
    if t.casefold() in {s.casefold() for s in SKIP_EXACT if s}:
        return True
    if len(t) <= 2 and is_clarification_reply:
        return True
    # 过短且无实质语义（如单独 "RD"）
    if len(t) < 4 and "?" not in t and "？" not in t and _cjk_count(t) < 2:
        return True
    return False


def collect_questions(sessions_dir: Path) -> tuple[list[dict], dict]:
    entries_by_key: dict[str, dict] = {}
    total_inputs = 0
    session_files = list(sessions_dir.rglob("*.json"))

    for path in sorted(session_files):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        session_id = str(data.get("session_id") or path.stem).strip()
        date_prefix = path.parent.name if path.parent.name.isdigit() else ""

        ledgers: list[tuple[str, list]] = []
        if isinstance(data.get("raw_ledger"), list):
            ledgers.append(("", data["raw_ledger"]))
        threads = data.get("threads")
        if isinstance(threads, dict):
            for tid, tdata in threads.items():
                if isinstance(tdata, dict) and isinstance(tdata.get("raw_ledger"), list):
                    ledgers.append((str(tid), tdata["raw_ledger"]))

        for thread_id, ledger in ledgers:
            for ev in ledger:
                if not isinstance(ev, dict):
                    continue
                if str(ev.get("event_type") or "").upper() != "USER_INPUT":
                    continue
                payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
                text = str(payload.get("effective_text") or payload.get("text") or "").strip()
                if not text:
                    continue
                is_clar = bool(payload.get("is_clarification_reply"))
                if should_skip(text, is_clarification_reply=is_clar):
                    continue
                total_inputs += 1
                key = norm_key(text)
                q = normalize_question(text)
                if key not in entries_by_key:
                    entries_by_key[key] = {
                        "id": "",
                        "question": q,
                        "ground_truth": "",
                        "tags": [],
                        "notes": "",
                        "metadata": {
                            "first_seen_date": date_prefix,
                            "occurrence_count": 0,
                            "is_clarification_reply_seen": False,
                            "session_ids": [],
                            "thread_ids": [],
                            "query_ids": [],
                        },
                    }
                rec = entries_by_key[key]
                meta = rec["metadata"]
                meta["occurrence_count"] += 1
                if is_clar:
                    meta["is_clarification_reply_seen"] = True
                if date_prefix and (
                    not meta.get("first_seen_date") or date_prefix < meta["first_seen_date"]
                ):
                    meta["first_seen_date"] = date_prefix
                if session_id and session_id not in meta["session_ids"]:
                    meta["session_ids"].append(session_id)
                if thread_id and thread_id not in meta["thread_ids"]:
                    meta["thread_ids"].append(thread_id)
                qid = str(payload.get("query_id") or "").strip()
                if qid and qid not in meta["query_ids"]:
                    meta["query_ids"].append(qid)

    items = sorted(
        entries_by_key.values(),
        key=lambda x: (x["metadata"]["first_seen_date"], x["question"]),
    )
    for i, item in enumerate(items, start=1):
        item["id"] = f"q_{i:04d}"

    stats = {
        "session_files_scanned": len(session_files),
        "user_input_events": total_inputs,
        "unique_questions": len(items),
    }
    return items, stats


def write_outputs(*, items: list[dict], stats: dict, sessions_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = {
        "schema_version": "1.0",
        "description": "从 demo/sessions 抽取的去重用户问题评测集；ground_truth 留空待补充。",
        "source_dir": str(sessions_dir.resolve()),
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "stats": stats,
        "items": items,
    }
    json_path = out_dir / "session_questions_eval.json"
    json_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = out_dir / "session_questions_eval.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "id",
                "question",
                "ground_truth",
                "tags",
                "notes",
                "occurrence_count",
                "first_seen_date",
                "is_clarification_reply_seen",
            ]
        )
        for item in items:
            m = item["metadata"]
            w.writerow(
                [
                    item["id"],
                    item["question"],
                    item["ground_truth"],
                    "|".join(item.get("tags") or []),
                    item.get("notes") or "",
                    m.get("occurrence_count", 0),
                    m.get("first_seen_date", ""),
                    m.get("is_clarification_reply_seen", False),
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deduplicated eval set from session files")
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        default=DEFAULT_SESSIONS_DIR,
        help="Session JSON root (default: demo/sessions)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory (default: demo/eval)",
    )
    args = parser.parse_args()
    items, stats = collect_questions(args.sessions_dir)
    write_outputs(items=items, stats=stats, sessions_dir=args.sessions_dir, out_dir=args.out_dir)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"wrote {args.out_dir / 'session_questions_eval.json'}")
    print(f"wrote {args.out_dir / 'session_questions_eval.csv'}")


if __name__ == "__main__":
    main()
