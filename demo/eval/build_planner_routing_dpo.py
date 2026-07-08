#!/usr/bin/env python3
"""Build DPO preference pairs from planner routing eval failures.

The builder converts failed P0 routing eval rows into repair-style preference
pairs:

- prompt: Planner messages reconstructed with the same prompt builders used by
  the routing eval.
- chosen: a synthetic decision from the case expectation.
- rejected: the model's actual failed Planner decision from the report.

The output is intentionally small and reviewable. Treat it as a seed set, not a
fully automated training corpus.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEMO_DIR = ROOT / "demo"
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import agent  # noqa: E402

DEFAULT_CASES = ROOT / "dataset" / "planner_routing_eval.json"
DEFAULT_REPORTS = [
    ROOT / "results" / "planner_routing_eval" / "planner_routing_report_Qwen3.5-4B.json",
    ROOT / "results" / "planner_routing_eval" / "planner_routing_report_Qwen3.5-9B.json",
]
DEFAULT_OUT_DIR = ROOT / "results" / "planner_routing_eval" / "dpo"
DEFAULT_REJECT_CASE_IDS = {"ROUTE-MIG-004", "ROUTE-MIG-005", "ROUTE-VIS-004"}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def cases_by_id(cases_path: Path) -> dict[str, dict[str, Any]]:
    data = load_json(cases_path)
    cases = data.get("cases")
    if not isinstance(cases, list):
        raise ValueError("cases file must contain a cases array")
    out: dict[str, dict[str, Any]] = {}
    for case in cases:
        if not isinstance(case, dict):
            continue
        cid = str(case.get("case_id") or "").strip()
        if cid:
            out[cid] = case
    return out


def resolve_image_path(case: dict[str, Any]) -> str:
    setup = case.get("setup") if isinstance(case.get("setup"), dict) else {}
    if not bool(setup.get("has_image")):
        return ""
    fixture = str(setup.get("image_fixture") or "").strip()
    if not fixture:
        return ""
    path = Path(fixture)
    if not path.is_absolute():
        path = ROOT / path
    return str(path.resolve()) if path.is_file() else ""


def build_planner_messages(case: dict[str, Any], *, model: str) -> list[dict[str, Any]]:
    image_path = resolve_image_path(case)
    system_prompt = agent.build_agent_system_prompt(max_steps=agent.AGENT_MAX_STEPS)
    user_prompt = agent.build_agent_user_prompt(
        str(case.get("user_query") or ""),
        image_path or None,
        planner_context={
            "session_id": f"planner_routing_eval_{case.get('case_id')}",
            "query_trajectories": [],
        },
        step_index=1,
        max_steps=agent.AGENT_MAX_STEPS,
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _first_contains(required_slots: dict[str, Any], key: str, fallback: str = "") -> str:
    values = required_slots.get(key)
    items = [str(x or "").strip() for x in _as_list(values) if str(x or "").strip()]
    return " ".join(items) if items else fallback


def _infer_eval_type(query: str) -> int:
    low = str(query or "").lower()
    if any(token in low for token in ("性能", "速度", "耗时", "延迟", "latency", "throughput", "fps", "performance")):
        return 1
    return 0


def _infer_model_name(query: str) -> str:
    match = re.search(r"\b[A-Za-z][A-Za-z0-9_./-]{2,}\b", str(query or ""))
    if not match:
        return ""
    value = match.group(0)
    if value.lower() in {"what", "open", "object", "detection", "accuracy", "normal", "performance"}:
        return ""
    return value


def _chosen_action_input(case: dict[str, Any]) -> dict[str, Any]:
    expected = case.get("expected") if isinstance(case.get("expected"), dict) else {}
    action = str(expected.get("primary_action") or "").strip()
    required_slots = expected.get("required_slots") if isinstance(expected.get("required_slots"), dict) else {}
    query = str(case.get("user_query") or "").strip()

    if action == agent.TOOL_RAG_ANSWER:
        return {
            "query": query,
            "finish_after_tool": True,
        }
    if action == agent.TOOL_ANSWERER:
        return {
            "mode": "direct",
            "finish_after_tool": True,
        }
    if action == agent.TOOL_ADELA_CLI_EVAL:
        out: dict[str, Any] = {
            "platform": str(required_slots.get("platform") or _first_contains(required_slots, "platform_contains")).strip(),
            "eval_type": int(required_slots.get("eval_type") if required_slots.get("eval_type") is not None else _infer_eval_type(query)),
            "finish_after_tool": True,
        }
        model_name = str(required_slots.get("model_name") or "").strip() or _infer_model_name(query)
        if model_name:
            out["model_name"] = model_name
        if required_slots.get("rawmodel_id") is not None:
            try:
                out["rawmodel_id"] = int(required_slots.get("rawmodel_id"))
            except (TypeError, ValueError):
                pass
        return out
    if action == agent.TOOL_FLUX_IMAGE_GENERATION:
        return {
            "task_text": _first_contains(required_slots, "task_text_contains", query),
            "source_image_required": bool(required_slots.get("source_image_required", bool(resolve_image_path(case)))),
            "num_images": int(required_slots.get("num_images") or 1),
            "finish_after_tool": True,
        }
    if action in {agent.TOOL_QWEN_DETECTION, agent.TOOL_REXOMNI_DETECTION}:
        return {
            "label": _first_contains(required_slots, "label_contains", query),
            "finish_after_tool": True,
        }
    if action == agent.TOOL_PIPELINE_EVAL:
        return {
            "task_text": _first_contains(required_slots, "task_text_contains", query),
            "finish_after_tool": True,
        }
    if action == agent.TOOL_MIGRATION_ADVISOR:
        return {
            "user_query": query,
            "use_image": bool(required_slots.get("use_image", bool(resolve_image_path(case)))),
            "use_visual_probe": bool(required_slots.get("use_visual_probe", bool(resolve_image_path(case)))),
            "finish_after_tool": True,
        }
    return {"finish_after_tool": True}


def _decision_action(decision: dict[str, Any]) -> str:
    if not isinstance(decision, dict):
        return ""
    decision_type = str(decision.get("decision_type") or "").strip()
    if decision_type in {"clarify", "end"}:
        return decision_type
    return str(decision.get("action") or "").strip()


def _pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _markdown_escape(text: str) -> str:
    return str(text or "").replace("|", "\\|").strip()


def build_chosen_decision(case: dict[str, Any]) -> dict[str, Any]:
    expected = case.get("expected") if isinstance(case.get("expected"), dict) else {}
    action = str(expected.get("primary_action") or "").strip()
    reason = str(case.get("reason") or "").strip()
    if not action:
        raise ValueError(f"case {case.get('case_id')} missing expected.primary_action")
    return {
        "thought": reason or "根据路由评测期望，选择最能满足用户核心诉求的工具。",
        "decision_type": "tool",
        "action": action,
        "action_input": _chosen_action_input(case),
        "final_answer": "",
    }


def normalize_rejected(step: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(step, dict):
        return {}
    decision_type = str(step.get("decision_type") or "tool").strip() or "tool"
    out = {
        "thought": str(step.get("thought") or "").strip(),
        "decision_type": decision_type,
    }
    if decision_type == "clarify":
        out["clarification_question"] = str(step.get("clarification_question") or "").strip()
        return out
    if decision_type == "end":
        out["end_reason"] = str(step.get("end_reason") or "").strip()
        out["final_answer"] = str(step.get("final_answer") or "").strip()
        return out
    out["action"] = agent.normalize_agent_action(str(step.get("action") or "").strip())
    action_input = step.get("action_input") if isinstance(step.get("action_input"), dict) else {}
    out["action_input"] = action_input
    out["final_answer"] = str(step.get("final_answer") or "").strip()
    return out


def pair_key(pair: dict[str, Any]) -> tuple[str, str, str, str]:
    meta = pair.get("meta") if isinstance(pair.get("meta"), dict) else {}
    chosen = pair.get("chosen") if isinstance(pair.get("chosen"), dict) else {}
    rejected = pair.get("rejected") if isinstance(pair.get("rejected"), dict) else {}
    return (
        str(meta.get("case_id") or ""),
        str(meta.get("model") or ""),
        str(chosen.get("action") or chosen.get("decision_type") or ""),
        str(rejected.get("action") or rejected.get("decision_type") or ""),
    )


def build_pairs(*, cases: dict[str, dict[str, Any]], reports: list[Path]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for report_path in reports:
        data = load_json(report_path)
        model = str(data.get("model") or report_path.stem).strip()
        for row in data.get("results") if isinstance(data.get("results"), list) else []:
            if not isinstance(row, dict) or row.get("passed") is not False:
                continue
            cid = str(row.get("case_id") or "").strip()
            case = cases.get(cid)
            if not case:
                continue
            rejected = normalize_rejected(row.get("actual_step") if isinstance(row.get("actual_step"), dict) else {})
            if not rejected:
                continue
            chosen = build_chosen_decision(case)
            prompt_messages = build_planner_messages(case, model=model)
            pair = {
                "prompt": prompt_messages,
                "chosen": json.dumps(chosen, ensure_ascii=False),
                "rejected": json.dumps(rejected, ensure_ascii=False),
                "meta": {
                    "case_id": cid,
                    "title": str(case.get("title") or row.get("title") or ""),
                    "category": str(case.get("category") or row.get("category") or ""),
                    "user_query": str(case.get("user_query") or row.get("user_query") or ""),
                    "model": model,
                    "source_report": str(report_path),
                    "pair_type": "repair",
                    "chosen_synthetic": True,
                    "rejected_synthetic": False,
                    "expected_action": str(row.get("expected_action") or ""),
                    "actual_action": str(row.get("actual_action") or ""),
                    "failures": row.get("failures") if isinstance(row.get("failures"), list) else [],
                    "reason": str(case.get("reason") or ""),
                    "dpo_preference": case.get("dpo_preference") if isinstance(case.get("dpo_preference"), dict) else {},
                    "needs_human_review": True,
                },
            }
            key = pair_key(pair)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(pair)
    return pairs


def write_review_csv(path: Path, pairs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "review_status",
        "case_id",
        "model",
        "category",
        "title",
        "user_query",
        "expected_action",
        "actual_action",
        "chosen_action",
        "rejected_action",
        "failures",
        "reason",
        "chosen_json",
        "rejected_json",
        "reviewer_note",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for pair in pairs:
            meta = pair.get("meta") if isinstance(pair.get("meta"), dict) else {}
            case_id = str(meta.get("case_id") or "")
            review_status = "reject" if case_id in DEFAULT_REJECT_CASE_IDS else "approve"
            chosen = json.loads(str(pair.get("chosen") or "{}"))
            rejected = json.loads(str(pair.get("rejected") or "{}"))
            writer.writerow(
                {
                    "review_status": review_status,
                    "case_id": case_id,
                    "model": meta.get("model", ""),
                    "category": meta.get("category", ""),
                    "title": meta.get("title", ""),
                    "user_query": meta.get("user_query", ""),
                    "expected_action": meta.get("expected_action", ""),
                    "actual_action": meta.get("actual_action", ""),
                    "chosen_action": _decision_action(chosen),
                    "rejected_action": _decision_action(rejected),
                    "failures": "; ".join(str(x) for x in meta.get("failures", [])),
                    "reason": meta.get("reason", ""),
                    "chosen_json": json.dumps(chosen, ensure_ascii=False),
                    "rejected_json": json.dumps(rejected, ensure_ascii=False),
                    "reviewer_note": (
                        "人工审核：action space 已改为显式 migration_advisor，旧样本不进训练。"
                        if case_id in {"ROUTE-MIG-004", "ROUTE-MIG-005"}
                        else ("人工审核：槽位规则过窄，rejected 不明显更差。" if case_id == "ROUTE-VIS-004" else "")
                    ),
                }
            )


def render_review_markdown(pairs: list[dict[str, Any]], report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Planner Routing DPO Review")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Pair count: {report.get('pair_count', 0)}")
    lines.append(f"- By model: `{json.dumps(report.get('by_model', {}), ensure_ascii=False)}`")
    lines.append(f"- By category: `{json.dumps(report.get('by_category', {}), ensure_ascii=False)}`")
    lines.append("")
    lines.append("## Review Instructions")
    lines.append("")
    lines.append("- Mark each pair as `approve`, `reject`, or `fix` in the CSV.")
    lines.append("- Approve only if the chosen decision is clearly better than the rejected decision.")
    lines.append("- Reject if the case expectation is wrong, the chosen slot is over-specified, or the rejected decision is actually acceptable.")
    lines.append("- Fix if the action is correct but a chosen slot should be edited before training.")
    lines.append("- Keep in mind: `chosen` is synthetic and must be human-reviewed; `rejected` is a real Planner output.")
    lines.append("")
    lines.append("## Index")
    lines.append("")
    lines.append("| # | Case | Model | Category | Chosen > Rejected | Status |")
    lines.append("|---:|---|---|---|---|---|")
    parsed: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for pair in pairs:
        meta = pair.get("meta") if isinstance(pair.get("meta"), dict) else {}
        chosen = json.loads(str(pair.get("chosen") or "{}"))
        rejected = json.loads(str(pair.get("rejected") or "{}"))
        parsed.append((meta, chosen, rejected))
    for idx, (meta, chosen, rejected) in enumerate(parsed, start=1):
        lines.append(
            "| {idx} | {case_id} | {model} | {category} | `{chosen}` > `{rejected}` | TODO |".format(
                idx=idx,
                case_id=_markdown_escape(str(meta.get("case_id") or "")),
                model=_markdown_escape(str(meta.get("model") or "")),
                category=_markdown_escape(str(meta.get("category") or "")),
                chosen=_markdown_escape(_decision_action(chosen)),
                rejected=_markdown_escape(_decision_action(rejected)),
            )
        )
    lines.append("")
    lines.append("## Pairs")
    for idx, (meta, chosen, rejected) in enumerate(parsed, start=1):
        lines.append("")
        lines.append(f"### {idx}. {meta.get('case_id', '')} - {meta.get('title', '')}")
        lines.append("")
        lines.append(f"- Model: `{meta.get('model', '')}`")
        lines.append(f"- Category: `{meta.get('category', '')}`")
        lines.append(f"- User query: {meta.get('user_query', '')}")
        lines.append(f"- Expected action: `{meta.get('expected_action', '')}`")
        lines.append(f"- Actual action: `{meta.get('actual_action', '')}`")
        failures = meta.get("failures") if isinstance(meta.get("failures"), list) else []
        lines.append(f"- Failures: {', '.join(str(x) for x in failures) if failures else 'None'}")
        lines.append(f"- Reason: {meta.get('reason', '')}")
        lines.append("")
        default_status = "reject" if str(meta.get("case_id") or "") in DEFAULT_REJECT_CASE_IDS else "approve"
        lines.append(f"Review decision: `{default_status}`")
        lines.append("")
        lines.append("Chosen:")
        lines.append("")
        lines.append("```json")
        lines.append(_pretty_json(chosen))
        lines.append("```")
        lines.append("")
        lines.append("Rejected:")
        lines.append("")
        lines.append("```json")
        lines.append(_pretty_json(rejected))
        lines.append("```")
    lines.append("")
    return "\n".join(lines)


def approved_pairs(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for pair in pairs:
        meta = pair.get("meta") if isinstance(pair.get("meta"), dict) else {}
        if str(meta.get("case_id") or "") in DEFAULT_REJECT_CASE_IDS:
            continue
        clean = dict(pair)
        clean_meta = dict(meta)
        clean_meta["human_review_status"] = "approve"
        clean["meta"] = clean_meta
        out.append(clean)
    return out


def summarize(pairs: list[dict[str, Any]], *, reports: list[Path], out_dir: Path) -> dict[str, Any]:
    by_model = Counter()
    by_category = Counter()
    chosen_actions = Counter()
    rejected_actions = Counter()
    for pair in pairs:
        meta = pair.get("meta") if isinstance(pair.get("meta"), dict) else {}
        by_model[str(meta.get("model") or "")] += 1
        by_category[str(meta.get("category") or "")] += 1
        try:
            chosen = json.loads(str(pair.get("chosen") or "{}"))
        except Exception:
            chosen = {}
        try:
            rejected = json.loads(str(pair.get("rejected") or "{}"))
        except Exception:
            rejected = {}
        chosen_actions[str(chosen.get("action") or chosen.get("decision_type") or "")] += 1
        rejected_actions[str(rejected.get("action") or rejected.get("decision_type") or "")] += 1
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pair_count": len(pairs),
        "reports": [str(p) for p in reports],
        "out_dir": str(out_dir),
        "by_model": dict(by_model),
        "by_category": dict(by_category),
        "chosen_actions": dict(chosen_actions),
        "rejected_actions": dict(rejected_actions),
        "notes": [
            "Pairs are repair-style: chosen is synthesized from routing eval expectations and should be human-reviewed before training.",
            "Rejected decisions come from actual Planner outputs in routing eval reports.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DPO pairs from planner routing eval failures.")
    parser.add_argument("--cases", default=str(DEFAULT_CASES), help="Path to planner_routing_eval.json")
    parser.add_argument("--report", action="append", default=[], help="Routing report JSON path; may repeat")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases_path = Path(args.cases)
    if not cases_path.is_absolute():
        cases_path = ROOT / cases_path
    report_paths = [Path(p) for p in args.report] if args.report else list(DEFAULT_REPORTS)
    report_paths = [p if p.is_absolute() else ROOT / p for p in report_paths]
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    cases = cases_by_id(cases_path)
    pairs = build_pairs(cases=cases, reports=report_paths)
    report = summarize(pairs, reports=report_paths, out_dir=out_dir)

    write_jsonl(out_dir / "planner_routing_dpo_pairs.jsonl", pairs)
    write_jsonl(out_dir / "planner_routing_dpo_pairs.approved.jsonl", approved_pairs(pairs))
    write_json(out_dir / "planner_routing_dpo_report.json", report)
    write_review_csv(out_dir / "planner_routing_dpo_review.csv", pairs)
    write_text(out_dir / "planner_routing_dpo_review.md", render_review_markdown(pairs, report))
    print(
        "Planner routing DPO:",
        f"pairs={len(pairs)}",
        f"out={out_dir / 'planner_routing_dpo_pairs.jsonl'}",
        f"review={out_dir / 'planner_routing_dpo_review.md'}",
    )


if __name__ == "__main__":
    main()
