#!/usr/bin/env python3
"""Run deterministic P0 Planner routing eval cases.

This evaluates the first Planner decision only. It does not execute selected
tools, so heavy actions such as Flux, Qwen, RexOmni, and Adela are safe.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = ROOT / "demo"
DEFAULT_CASES = ROOT / "training" / "planner_dpo_train_seed_v1" / "eval" / "planner_routing_eval_90cases.json"
DEFAULT_OUT = ROOT / "results" / "planner_routing_eval" / "planner_routing_report.json"

TOKEN_ALIASES = {
    "烟雾": ["smoke", "烟尘"],
    "烟尘": ["smoke", "烟雾"],
    "车辆": ["vehicle", "car", "truck", "车"],
    "公告牌": ["文字牌", "标识牌", "sign", "signboard"],
    "标识牌": ["公告牌", "文字牌", "sign", "signboard"],
    "垃圾车": ["garbage truck", "trash truck"],
    "背包": ["backpack", "bag"],
    "火焰": ["fire", "flame"],
    "横幅": ["banner"],
    "人": ["person", "people"],
}

agent = None


class PlannerCaseTimeout(TimeoutError):
    """Raised when one planner eval case exceeds the configured timeout."""


def configure_environment(args: argparse.Namespace) -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if str(DEMO_DIR) not in sys.path:
        sys.path.insert(0, str(DEMO_DIR))

    if args.api_base:
        os.environ["DEMO_ROUTE_API_BASE"] = str(args.api_base).rstrip("/")
    if args.api_key:
        os.environ["DEMO_ROUTE_API_KEY"] = str(args.api_key)
    if args.model:
        os.environ["DEMO_ROUTE_MODEL"] = str(args.model)
    if args.temperature is not None:
        os.environ["DEMO_OPENAI_TEMPERATURE"] = str(args.temperature)
    if args.top_p is not None:
        os.environ["DEMO_OPENAI_TOP_P"] = str(args.top_p)
    if args.seed is not None:
        os.environ["DEMO_OPENAI_SEED"] = str(args.seed)
    if args.do_sample:
        os.environ["DEMO_OPENAI_DO_SAMPLE"] = str(args.do_sample)
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost,10.111.32.253")
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost,10.111.32.253")


def import_agent_module():
    global agent
    if agent is None:
        import agent as agent_module  # type: ignore

        agent = agent_module
    return agent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _alarm_handler(signum: int, frame: Any) -> None:
    raise PlannerCaseTimeout("planner case timed out")


def selected_cases(data: dict[str, Any], *, case_ids: set[str], limit: int) -> list[dict[str, Any]]:
    cases = data.get("cases")
    if not isinstance(cases, list):
        raise ValueError("cases file must contain a cases array")
    out: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        cid = str(case.get("case_id") or "").strip()
        if case_ids and cid not in case_ids:
            continue
        out.append(case)
        if limit > 0 and len(out) >= limit:
            break
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


def normalize_actual_action(step: dict[str, Any]) -> str:
    ag = import_agent_module()
    decision_type = str(step.get("decision_type") or "").strip().lower()
    if decision_type == ag.DECISION_TYPE_CLARIFY:
        return "clarify"
    if decision_type == ag.DECISION_TYPE_END:
        return ag.ACTION_FINAL_ANSWER
    return ag.normalize_agent_action(str(step.get("action") or "").strip())


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _text_contains_all(text: str, tokens: list[Any]) -> bool:
    haystack = str(text or "").lower()
    for token in tokens:
        normalized = str(token or "").strip().lower()
        if not normalized:
            continue
        aliases = [str(item or "").lower() for item in TOKEN_ALIASES.get(str(token or "").strip(), [])]
        if normalized not in haystack and not any(alias and alias in haystack for alias in aliases):
            return False
    return True


def _slot_value(action_input: dict[str, Any], slot: str, user_query: str) -> Any:
    if slot == "user_query":
        return action_input.get("user_query") or user_query
    if slot == "query":
        return action_input.get("query") or user_query
    if slot == "task_text":
        return action_input.get("task_text") or user_query
    if slot == "label":
        return action_input.get("label")
    if slot == "platform":
        return action_input.get("platform")
    return action_input.get(slot)


def check_required_slots(
    *,
    required_slots: dict[str, Any],
    action_input: dict[str, Any],
    user_query: str,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for raw_key, expected in required_slots.items():
        key = str(raw_key)
        if key.endswith("_contains"):
            slot = key[: -len("_contains")]
            actual = _slot_value(action_input, slot, user_query)
            if not _text_contains_all(str(actual or ""), _as_list(expected)):
                failures.append(f"{key}: expected tokens {_as_list(expected)!r}, got {actual!r}")
            continue

        if key == "query_contains":
            actual = action_input.get("query") or user_query
            if not _text_contains_all(str(actual or ""), _as_list(expected)):
                failures.append(f"{key}: expected tokens {_as_list(expected)!r}, got {actual!r}")
            continue

        actual = action_input.get(key)
        if isinstance(expected, int):
            try:
                actual_int = int(actual)
            except (TypeError, ValueError):
                failures.append(f"{key}: expected {expected!r}, got {actual!r}")
                continue
            if actual_int != expected:
                failures.append(f"{key}: expected {expected!r}, got {actual!r}")
            continue

        if isinstance(expected, bool):
            if bool(actual) is not expected:
                failures.append(f"{key}: expected {expected!r}, got {actual!r}")
            continue

        if str(actual or "").strip() != str(expected or "").strip():
            failures.append(f"{key}: expected {expected!r}, got {actual!r}")

    return not failures, failures


def evaluate_case(case: dict[str, Any], step: dict[str, Any] | None) -> dict[str, Any]:
    expected = case.get("expected") if isinstance(case.get("expected"), dict) else {}
    primary = str(expected.get("primary_action") or "").strip()
    acceptable = {primary}
    acceptable.update(str(x).strip() for x in _as_list(expected.get("acceptable_actions")) if str(x).strip())
    forbidden = {str(x).strip() for x in _as_list(expected.get("forbidden_actions")) if str(x).strip()}

    if step is None:
        return {
            "passed": None,
            "action_match": None,
            "slot_match": None,
            "forbidden_violation": None,
            "actual_action": "",
            "actual_step": None,
            "failures": ["not_run"],
        }

    actual_action = normalize_actual_action(step)
    action_input = step.get("action_input") if isinstance(step.get("action_input"), dict) else {}
    required_slots = expected.get("required_slots") if isinstance(expected.get("required_slots"), dict) else {}
    slot_match, slot_failures = check_required_slots(
        required_slots=required_slots,
        action_input=action_input,
        user_query=str(case.get("user_query") or ""),
    )
    action_match = actual_action in acceptable
    forbidden_violation = actual_action in forbidden
    failures: list[str] = []
    if not action_match:
        failures.append(f"action: expected one of {sorted(acceptable)!r}, got {actual_action!r}")
    if forbidden_violation:
        failures.append(f"forbidden action used: {actual_action}")
    failures.extend(slot_failures)

    return {
        "passed": bool(action_match and slot_match and not forbidden_violation),
        "action_match": action_match,
        "slot_match": slot_match,
        "forbidden_violation": forbidden_violation,
        "actual_action": actual_action,
        "actual_step": step,
        "failures": failures,
    }


def _planner_metrics(step: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(step, dict):
        return {}
    value = step.get("_planner_metrics")
    return value if isinstance(value, dict) else {}


def _metric_number(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, (int, float)):
        return round(float(value), 3)
    return None


def _metric_int(metrics: dict[str, Any], key: str) -> int | None:
    value = metrics.get(key)
    if isinstance(value, int):
        return value
    return None


def _error_type_from_message(error: str) -> str:
    text = str(error or "")
    if not text:
        return ""
    if "timed out" in text.lower() or "timeout" in text.lower():
        return "timeout"
    if "schema parse" in text.lower() or "parse" in text.lower():
        return "parse_error"
    if "api" in text.lower() or "connection" in text.lower() or "http" in text.lower():
        return "api_error"
    return "planner_error"


def run_case(case: dict[str, Any], *, model: str, no_llm: bool, timeout_seconds: int = 0) -> dict[str, Any]:
    ag = import_agent_module()
    image_path = resolve_image_path(case)
    image_missing = bool((case.get("setup") or {}).get("has_image")) and not image_path
    step: dict[str, Any] | None = None
    error = ""
    started_at = utc_now()
    start_perf = time.perf_counter()
    llm_end_perf = start_perf
    if not no_llm:
        previous_handler = None
        if timeout_seconds > 0:
            previous_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(timeout_seconds)
        try:
            step = ag.choose_agent_step_with_fallback(
                str(case.get("user_query") or ""),
                image_path or None,
                planner_context={
                    "session_id": f"planner_routing_eval_{case.get('case_id')}",
                    "query_trajectories": [],
                },
                step_index=1,
                max_steps=ag.AGENT_MAX_STEPS,
                model=model or None,
                debug_meta={
                    "session_id": f"planner_routing_eval_{case.get('case_id')}",
                    "run_stamp": "planner_routing_eval",
                    "run_dir": str(DEFAULT_OUT.parent),
                },
            )
        except Exception as exc:
            error = str(exc)
        finally:
            llm_end_perf = time.perf_counter()
            if timeout_seconds > 0:
                signal.alarm(0)
                if previous_handler is not None:
                    signal.signal(signal.SIGALRM, previous_handler)
    elapsed_ms = round((llm_end_perf - start_perf) * 1000, 3)
    finished_at = utc_now()

    metrics = _planner_metrics(step)
    timing = {
        "end_to_end_ms": elapsed_ms,
        "prompt_build_ms": _metric_number(metrics, "prompt_build_ms"),
        "api_call_ms": _metric_number(metrics, "api_call_ms"),
        "first_api_call_ms": _metric_number(metrics, "first_api_call_ms"),
        "retry_api_call_ms": _metric_number(metrics, "retry_api_call_ms"),
        "parse_ms": _metric_number(metrics, "parse_ms"),
        "retry_parse_ms": _metric_number(metrics, "retry_parse_ms"),
        "planner_total_ms": _metric_number(metrics, "planner_total_ms"),
    }
    usage = {
        "input_tokens": _metric_int(metrics, "input_tokens"),
        "output_tokens": _metric_int(metrics, "output_tokens"),
        "total_tokens": _metric_int(metrics, "total_tokens"),
        "first_input_tokens": _metric_int(metrics, "first_input_tokens"),
        "first_output_tokens": _metric_int(metrics, "first_output_tokens"),
        "retry_input_tokens": _metric_int(metrics, "retry_input_tokens"),
        "retry_output_tokens": _metric_int(metrics, "retry_output_tokens"),
        "finish_reason": metrics.get("finish_reason") or metrics.get("first_finish_reason") or "",
        "retry_finish_reason": metrics.get("retry_finish_reason") or "",
    }
    retry = {
        "retry_count": int(metrics.get("retry_count") or 0),
        "retry_reason": str(metrics.get("retry_reason") or ""),
    }
    error_info = {
        "error_type": _error_type_from_message(error)
        or _error_type_from_message(str(metrics.get("error") or metrics.get("first_error") or "")),
        "message": error or str(metrics.get("error") or metrics.get("first_error") or ""),
    }
    if (
        not error_info["error_type"]
        and timeout_seconds > 0
        and elapsed_ms >= max(0, timeout_seconds * 1000 - 1000)
        and timing.get("api_call_ms") is None
    ):
        error_info = {
            "error_type": "timeout_or_slow_fallback",
            "message": "Planner returned a fallback step after a near-timeout interval without API usage metadata.",
        }

    scored = evaluate_case(case, step)
    if image_missing:
        scored["failures"] = [*scored.get("failures", []), "image fixture missing"]
        if scored["passed"] is True:
            scored["passed"] = False
    if error:
        scored["failures"] = [*scored.get("failures", []), f"planner_error: {error}"]
        scored["passed"] = False

    expected = case.get("expected") if isinstance(case.get("expected"), dict) else {}
    return {
        "case_id": str(case.get("case_id") or ""),
        "title": str(case.get("title") or ""),
        "category": str(case.get("category") or ""),
        "user_query": str(case.get("user_query") or ""),
        "image_path": image_path,
        "expected_action": str(expected.get("primary_action") or ""),
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_ms": elapsed_ms,
        "timing": timing,
        "usage": usage,
        "retry": retry,
        "error": error_info,
        **scored,
        "dpo_candidate": bool(scored.get("passed") is False),
    }


def summarize(results: list[dict[str, Any]], *, total_elapsed_ms: float | None = None) -> dict[str, Any]:
    evaluated = [r for r in results if r.get("passed") is not None]
    passed = [r for r in evaluated if r.get("passed") is True]
    elapsed_values = [
        float(((r.get("timing") or {}).get("end_to_end_ms") if isinstance(r.get("timing"), dict) else r.get("elapsed_ms")))
        for r in results
        if isinstance(((r.get("timing") or {}).get("end_to_end_ms") if isinstance(r.get("timing"), dict) else r.get("elapsed_ms")), (int, float))
    ]
    api_values = [
        float(r["timing"]["api_call_ms"])
        for r in results
        if isinstance(r.get("timing"), dict) and isinstance(r["timing"].get("api_call_ms"), (int, float))
    ]
    input_tokens = [
        int(r["usage"]["input_tokens"])
        for r in results
        if isinstance(r.get("usage"), dict) and isinstance(r["usage"].get("input_tokens"), int)
    ]
    output_tokens = [
        int(r["usage"]["output_tokens"])
        for r in results
        if isinstance(r.get("usage"), dict) and isinstance(r["usage"].get("output_tokens"), int)
    ]
    retry_counts = [
        int(r["retry"]["retry_count"])
        for r in results
        if isinstance(r.get("retry"), dict) and isinstance(r["retry"].get("retry_count"), int)
    ]
    slow_case_count = sum(1 for value in elapsed_values if value >= 100000)
    error_types = Counter(
        str(r["error"].get("error_type") or "")
        for r in results
        if isinstance(r.get("error"), dict) and str(r["error"].get("error_type") or "")
    )
    timeout_count = sum(
        1
        for r in results
        if isinstance(r.get("error"), dict) and "timeout" in str(r["error"].get("error_type") or "")
    )
    by_category: dict[str, dict[str, int]] = {}
    for category, group_count in Counter(r.get("category") for r in results).items():
        rows = [r for r in evaluated if r.get("category") == category]
        by_category[str(category)] = {
            "total": int(group_count),
            "evaluated": len(rows),
            "passed": sum(1 for r in rows if r.get("passed") is True),
        }
    return {
        "total": len(results),
        "evaluated": len(evaluated),
        "passed": len(passed),
        "failed": sum(1 for r in evaluated if r.get("passed") is False),
        "accuracy": round(len(passed) / len(evaluated), 4) if evaluated else None,
        "by_category": by_category,
        "actual_actions": dict(Counter(str(r.get("actual_action") or "not_run") for r in results)),
        "dpo_candidates": sum(1 for r in results if r.get("dpo_candidate")),
        "timing": {
            "source": "measured",
            "elapsed_ms": round(total_elapsed_ms, 3) if total_elapsed_ms is not None else None,
            "case_elapsed_ms_total": round(sum(elapsed_values), 3) if elapsed_values else None,
            "case_elapsed_ms_count": len(elapsed_values),
            "case_elapsed_ms_avg": round(sum(elapsed_values) / len(elapsed_values), 3) if elapsed_values else None,
            "case_elapsed_ms_min": round(min(elapsed_values), 3) if elapsed_values else None,
            "case_elapsed_ms_max": round(max(elapsed_values), 3) if elapsed_values else None,
            "api_call_ms_avg": round(sum(api_values) / len(api_values), 3) if api_values else None,
            "api_call_ms_min": round(min(api_values), 3) if api_values else None,
            "api_call_ms_max": round(max(api_values), 3) if api_values else None,
            "slow_case_count": slow_case_count,
            "timeout_count": timeout_count,
        },
        "usage": {
            "input_tokens_avg": round(sum(input_tokens) / len(input_tokens), 3) if input_tokens else None,
            "output_tokens_avg": round(sum(output_tokens) / len(output_tokens), 3) if output_tokens else None,
            "input_tokens_total": sum(input_tokens) if input_tokens else None,
            "output_tokens_total": sum(output_tokens) if output_tokens else None,
        },
        "retry": {
            "retry_count_total": sum(retry_counts) if retry_counts else 0,
            "retry_case_count": sum(1 for value in retry_counts if value > 0),
        },
        "errors": {
            "by_error_type": dict(error_types),
        },
    }


def build_report(
    *,
    cases_path: Path,
    cases: list[dict[str, Any]],
    model: str,
    no_llm: bool,
    out_path: Path | None = None,
    timeout_seconds: int = 0,
    resume: bool = False,
    generation_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ag = import_agent_module()
    results: list[dict[str, Any]] = []
    existing_by_id: dict[str, dict[str, Any]] = {}
    use_model = model or ag.DEMO_ROUTE_MODEL
    report_start = time.perf_counter()
    if resume and out_path and out_path.is_file():
        existing = load_json(out_path)
        if str(existing.get("model") or "") == use_model:
            existing_results = existing.get("results") if isinstance(existing.get("results"), list) else []
            existing_by_id = {
                str(row.get("case_id") or ""): row
                for row in existing_results
                if isinstance(row, dict) and str(row.get("case_id") or "")
            }

    for idx, case in enumerate(cases, start=1):
        cid = str(case.get("case_id") or "")
        if cid in existing_by_id:
            result = existing_by_id[cid]
        else:
            result = run_case(case, model=model, no_llm=no_llm, timeout_seconds=timeout_seconds)
        results.append(result)
        if out_path:
            partial_elapsed = (time.perf_counter() - report_start) * 1000
            partial = {
                "schema_version": "1.1",
                "generated_at": utc_now(),
                "cases_path": str(cases_path),
                "model": use_model,
                "api_base": os.environ.get("DEMO_ROUTE_API_BASE", ""),
                "generation_config": generation_config or {},
                "no_llm": no_llm,
                "partial": idx < len(cases),
                "summary": summarize(results, total_elapsed_ms=partial_elapsed),
                "results": results,
            }
            write_json(out_path, partial)

    total_elapsed = (time.perf_counter() - report_start) * 1000
    return {
        "schema_version": "1.1",
        "generated_at": utc_now(),
        "cases_path": str(cases_path),
        "model": use_model,
        "api_base": os.environ.get("DEMO_ROUTE_API_BASE", ""),
        "generation_config": generation_config or {},
        "no_llm": no_llm,
        "partial": False,
        "summary": summarize(results, total_elapsed_ms=total_elapsed),
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run P0 Planner routing eval.")
    parser.add_argument("--cases", default=str(DEFAULT_CASES), help="Path to planner_routing_eval_90cases.json")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output report JSON path")
    parser.add_argument("--model", default="", help="Override Planner model")
    parser.add_argument("--api-base", default="", help="OpenAI-compatible /v1 API base")
    parser.add_argument("--api-key", default=os.environ.get("DEMO_ROUTE_API_KEY", "token.sdc@2026"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--do-sample", default="false")
    parser.add_argument("--case-id", action="append", default=[], help="Run one case id; may repeat")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of cases")
    parser.add_argument("--no-llm", action="store_true", help="Only validate and list cases; do not call Planner")
    parser.add_argument("--resume", action="store_true", help="Reuse completed case rows already present in --out")
    parser.add_argument("--timeout-seconds", type=int, default=0, help="Per-case alarm timeout; 0 disables")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_environment(args)
    import_agent_module()
    cases_path = Path(args.cases)
    if not cases_path.is_absolute():
        cases_path = ROOT / cases_path
    data = load_json(cases_path)
    cases = selected_cases(data, case_ids=set(args.case_id), limit=max(0, int(args.limit or 0)))
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    generation_config = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "do_sample": args.do_sample,
    }
    report = build_report(
        cases_path=cases_path,
        cases=cases,
        model=args.model,
        no_llm=bool(args.no_llm),
        out_path=out_path,
        timeout_seconds=max(0, int(args.timeout_seconds or 0)),
        resume=bool(args.resume),
        generation_config=generation_config,
    )
    write_json(out_path, report)
    summary = report["summary"]
    timing = summary.get("timing") if isinstance(summary.get("timing"), dict) else {}
    print(
        "Planner routing eval:",
        f"evaluated={summary['evaluated']}/{summary['total']}",
        f"passed={summary['passed']}",
        f"failed={summary['failed']}",
        f"accuracy={summary['accuracy']}",
        f"elapsed_ms={timing.get('elapsed_ms')}",
        f"case_avg_ms={timing.get('case_elapsed_ms_avg')}",
        f"out={out_path}",
    )


if __name__ == "__main__":
    main()
