#!/usr/bin/env python3
"""Audit recorded agent traces against a small set of trace-quality checks.

This script intentionally stays lightweight:
- reuses existing case definitions in demo/eval/session_cases_eval.json
- matches cases against recorded demo/sessions/*.json threads
- inspects query_trajectories, raw_ledger and demo/runs/* observation files
- emits a compact JSON report with a few counts and per-case findings
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = Path(__file__).resolve().parent / "session_cases_eval.json"
DEFAULT_SESSIONS_DIR = ROOT / "demo" / "sessions"

UNCERTAINTY_HINTS = (
    "无法",
    "不确定",
    "未找到",
    "证据不足",
    "需要补充",
    "请提供",
    "请补充",
    "澄清",
    "失败",
    "未返回有效结果",
    "空结果",
    "暂时",
)

OBS_COMPARE_KEYS = (
    "answer",
    "summary",
    "error",
    "success",
    "rewritten_query",
    "clarification_question",
    "final_answer",
    "action",
)


@dataclass
class CaseMatch:
    session_path: Path
    session_id: str
    thread_id: str
    thread_data: dict[str, Any]
    user_event_indexes: list[int]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def extract_case_turn_texts(case: dict[str, Any]) -> list[str]:
    turns = case.get("turns")
    if not isinstance(turns, list):
        return []
    out: list[str] = []
    for item in turns:
        if not isinstance(item, dict):
            continue
        if str(item.get("role") or "") != "user":
            continue
        text = normalize_text(str(item.get("text") or ""))
        if text:
            out.append(text)
    return out


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_session_threads(sessions_dir: Path):
    for path in sorted(sessions_dir.rglob("*.json")):
        try:
            data = load_json(path)
        except Exception:
            continue
        threads = data.get("threads")
        if not isinstance(threads, dict):
            continue
        sid = str(data.get("session_id") or path.stem)
        for tid, thread_data in threads.items():
            if isinstance(thread_data, dict):
                yield path, sid, str(tid), thread_data


def match_case(case: dict[str, Any], sessions_dir: Path) -> CaseMatch | None:
    target = extract_case_turn_texts(case)
    if not target:
        return None

    for path, sid, tid, thread_data in iter_session_threads(sessions_dir):
        ledger = thread_data.get("raw_ledger")
        if not isinstance(ledger, list):
            continue
        user_events: list[tuple[int, str]] = []
        for idx, ev in enumerate(ledger):
            if str(ev.get("event_type") or "").upper() != "USER_INPUT":
                continue
            payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
            text = normalize_text(str(payload.get("text") or ""))
            if text:
                user_events.append((idx, text))
        if len(user_events) < len(target):
            continue
        for start in range(0, len(user_events) - len(target) + 1):
            chunk = user_events[start : start + len(target)]
            if [text for _, text in chunk] == target:
                return CaseMatch(
                    session_path=path,
                    session_id=sid,
                    thread_id=tid,
                    thread_data=thread_data,
                    user_event_indexes=[idx for idx, _ in chunk],
                )
    return None


def build_event_map(ledger: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for ev in ledger:
        eid = str(ev.get("event_id") or "").strip()
        if eid:
            out[eid] = ev
    return out


def find_assistant_outputs_between(
    ledger: list[dict[str, Any]], start_idx: int, end_idx: int | None
) -> list[dict[str, Any]]:
    hi = len(ledger) if end_idx is None else end_idx
    out: list[dict[str, Any]] = []
    for ev in ledger[start_idx + 1 : hi]:
        if str(ev.get("event_type") or "").upper() == "ASSISTANT_OUTPUT":
            out.append(ev)
    return out


def query_id_for_user_event(event: dict[str, Any]) -> str:
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    return str(payload.get("query_id") or "").strip()


def derive_required_tools(case: dict[str, Any]) -> list[str]:
    audit = case.get("audit") if isinstance(case.get("audit"), dict) else {}
    explicit = audit.get("required_tools")
    if isinstance(explicit, list):
        return [str(x).strip() for x in explicit if str(x).strip()]

    expect = case.get("expect") if isinstance(case.get("expect"), dict) else {}
    primary = expect.get("tool_sequence")
    alternatives = expect.get("tool_sequence_alternatives")
    seqs: list[list[Any]] = []
    if isinstance(primary, list) and primary:
        seqs.append(primary)
    if isinstance(alternatives, list):
        for alt in alternatives:
            if isinstance(alt, list) and alt:
                seqs.append(alt)

    best: list[str] = []
    for seq in seqs:
        tools: list[str] = []
        for item in seq:
            if not isinstance(item, list) or not item:
                continue
            name = str(item[0] or "").strip()
            if name and name not in ("final_answer", "answerer"):
                tools.append(name)
        if tools and (not best or len(tools) < len(best)):
            best = tools
    return list(dict.fromkeys(best))


def flatten_trace_actions(trajectories: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for traj in trajectories:
        for step in traj.get("steps", []):
            name = str(step.get("action") or "").strip()
            if name:
                out.append(name)
    return out


def load_external_observation(path_text: str) -> dict[str, Any] | None:
    path = Path(path_text)
    if not path_text or not path.is_file():
        return None
    if path.suffix.lower() != ".json":
        return None
    try:
        data = load_json(path)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def compare_observation_payloads(
    ledger_observation: dict[str, Any], external_observation: dict[str, Any]
) -> bool | None:
    compared = False
    for key in OBS_COMPARE_KEYS:
        if key not in ledger_observation or key not in external_observation:
            continue
        compared = True
        if ledger_observation.get(key) != external_observation.get(key):
            return False
    return True if compared else None


def extract_text_blobs(observation: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key in ("answer", "summary", "error", "clarification_question", "final_answer", "result_summary"):
        value = str(observation.get(key) or "").strip()
        if value:
            out.append(value)
    return out


def has_uncertainty(text: str) -> bool:
    text = str(text or "")
    return any(tok in text for tok in UNCERTAINTY_HINTS)


def observation_failed_or_empty(observation: dict[str, Any]) -> bool:
    if observation.get("success") is False:
        return True
    if str(observation.get("error") or "").strip():
        return True
    answer = str(observation.get("answer") or "").strip()
    summary = str(observation.get("summary") or "").strip()
    if "未返回有效结果" in answer or "未返回有效结果" in summary:
        return True
    return False


def salient_tokens(text: str) -> set[str]:
    text = str(text or "")
    out: set[str] = set()
    for token in re.findall(r"\b[A-Za-z][A-Za-z0-9_.\-]{2,}\b", text):
        out.add(token)
    for token in re.findall(r"\b\d+(?:\.\d+)?%?\b", text):
        out.add(token)
    return out


def token_overlap(left: str, right: str) -> bool:
    a = salient_tokens(left)
    b = salient_tokens(right)
    return bool(a and b and (a & b))


def choose_final_text(assistant_outputs: list[dict[str, Any]]) -> str:
    texts: list[str] = []
    for ev in assistant_outputs:
        payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
        for key in ("final_answer", "clarification_question"):
            text = str(payload.get(key) or "").strip()
            if text:
                texts.append(text)
    return texts[-1] if texts else ""


def evaluate_case(case: dict[str, Any], match: CaseMatch | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "case_id": str(case.get("case_id") or ""),
        "title": str(case.get("title") or ""),
        "matched": bool(match),
    }
    if not match:
        result["status"] = "unmatched"
        return result

    ledger = match.thread_data.get("raw_ledger") if isinstance(match.thread_data.get("raw_ledger"), list) else []
    event_map = build_event_map(ledger)
    qmap = match.thread_data.get("query_trajectories") if isinstance(match.thread_data.get("query_trajectories"), dict) else {}

    trajectories: list[dict[str, Any]] = []
    final_texts: list[str] = []
    failed_observations: list[dict[str, Any]] = []
    text_observations: list[str] = []
    observation_compare_results: list[bool] = []
    earlier_summaries: list[str] = []

    for i, ev_idx in enumerate(match.user_event_indexes):
        user_ev = ledger[ev_idx]
        qid = query_id_for_user_event(user_ev)
        traj = qmap.get(qid) if isinstance(qmap.get(qid), dict) else {"query": "", "steps": [], "result_summary": ""}
        trajectories.append(traj)
        if i < len(match.user_event_indexes) - 1:
            next_idx = match.user_event_indexes[i + 1]
        else:
            next_idx = None
        outputs = find_assistant_outputs_between(ledger, ev_idx, next_idx)
        final_text = choose_final_text(outputs)
        final_texts.append(final_text)
        rs = str(traj.get("result_summary") or "").strip()
        if rs:
            earlier_summaries.append(rs)

        for step in traj.get("steps", []):
            if not isinstance(step, dict):
                continue
            obs_ev = event_map.get(str(step.get("observation_event_id") or "").strip())
            if not obs_ev:
                continue
            payload = obs_ev.get("payload") if isinstance(obs_ev.get("payload"), dict) else {}
            ext = load_external_observation(str(obs_ev.get("external_ref") or ""))
            cmp_res = compare_observation_payloads(payload, ext) if ext else None
            if cmp_res is not None:
                observation_compare_results.append(bool(cmp_res))
            if observation_failed_or_empty(payload):
                failed_observations.append(payload)
            text_observations.extend(extract_text_blobs(payload))

    actual_actions = flatten_trace_actions(trajectories)
    required_tools = derive_required_tools(case)
    required_tools_called = all(tool in actual_actions for tool in required_tools)

    final_report = final_texts[-1] if final_texts else ""
    disclosure_needed = bool(failed_observations)
    disclosed_failure = (not disclosure_needed) or has_uncertainty(final_report)

    support_checks: list[bool] = []
    for blob in text_observations:
        if final_report and (
            final_report in blob
            or blob in final_report
            or token_overlap(final_report, blob)
        ):
            support_checks.append(True)
            break
    final_supported = bool(support_checks) or (disclosure_needed and disclosed_failure)

    memory_applicable = len(trajectories) > 1 and bool(earlier_summaries[:-1] or earlier_summaries)
    memory_pass = None
    if memory_applicable and final_report:
        prior = "\n".join(earlier_summaries[:-1] or earlier_summaries[:1])
        memory_pass = token_overlap(final_report, prior) or prior in final_report

    exposed_critical_failure = (not disclosure_needed) or disclosed_failure
    observation_faithful = None
    if observation_compare_results:
        observation_faithful = all(observation_compare_results)

    result.update(
        {
            "status": "matched",
            "required_tools": required_tools,
            "actual_actions": actual_actions,
            "checks": {
                "called_required_tools": required_tools_called,
                "observation_reflects_output": observation_faithful,
                "evaluator_disclosed_failure_or_uncertainty": disclosed_failure,
                "final_report_supported_by_observation": final_supported,
                "working_memory_carried_early_observation": memory_pass,
                "critical_failure_exposed": exposed_critical_failure,
            },
            "final_report": final_report,
            "notes": {
                "failed_observation_count": len(failed_observations),
                "observation_compare_samples": len(observation_compare_results),
                "session_id": match.session_id,
                "thread_id": match.thread_id,
                "session_path": str(match.session_path),
            },
        }
    )
    return result


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    matched = [r for r in results if r.get("matched")]
    checks = [
        "called_required_tools",
        "observation_reflects_output",
        "evaluator_disclosed_failure_or_uncertainty",
        "final_report_supported_by_observation",
        "working_memory_carried_early_observation",
        "critical_failure_exposed",
    ]
    out: dict[str, Any] = {
        "cases_total": len(results),
        "cases_matched": len(matched),
        "counts": {},
    }
    for key in checks:
        applicable = 0
        passed = 0
        for item in matched:
            value = (item.get("checks") or {}).get(key)
            if value is None:
                continue
            applicable += 1
            if value is True:
                passed += 1
        out["counts"][key] = f"{passed}/{applicable}"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit recorded traces using compact trace-quality checks")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES, help="Case JSON file")
    parser.add_argument("--sessions-dir", type=Path, default=DEFAULT_SESSIONS_DIR, help="Recorded sessions directory")
    parser.add_argument("--limit", type=int, default=12, help="Only audit the first N cases")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "trace_audit_report.json")
    args = parser.parse_args()

    payload = load_json(args.cases)
    raw_cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(raw_cases, list):
        raise ValueError("cases file must contain a top-level 'cases' list")

    selected = raw_cases[: max(0, args.limit)]
    results: list[dict[str, Any]] = []
    for case in selected:
        if not isinstance(case, dict):
            continue
        match = match_case(case, args.sessions_dir)
        results.append(evaluate_case(case, match))

    report = {
        "summary": summarize(results),
        "results": results,
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
