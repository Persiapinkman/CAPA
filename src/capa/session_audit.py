"""Privacy-preserving aggregate audit of historical demo session ledgers."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .tools import registry as tool_registry


NON_TOOL_ACTIONS = {
    "",
    "clarify",
    "final_answer",
    "migration_advisor_offer",
    "self_intro",
}


def _cohort(path: Path) -> str:
    name = path.name
    if "codex_" in name:
        return "codex_eval"
    if "ma_dataset_" in name:
        return "migration_dataset"
    if "smoke" in name:
        return "smoke"
    if "_sess_" in name:
        return "browser_session"
    return "uuid_session"


def _trajectory_values(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        candidates = value.values()
    elif isinstance(value, list):
        candidates = value
    else:
        candidates = ()
    for candidate in candidates:
        if isinstance(candidate, dict):
            yield candidate


def _normalize_action(value: Any) -> str:
    return tool_registry.normalize_tool_action(str(value or "").strip())


def _success_bucket(payload: dict[str, Any]) -> str:
    for key in ("success", "ok"):
        value = payload.get(key)
        if isinstance(value, bool):
            return "success" if value else "failure"
    if payload.get("error") or payload.get("error_message"):
        return "failure"
    return "unknown"


def audit_sessions(sessions_dir: Path) -> dict[str, Any]:
    root = Path(sessions_dir).resolve()
    files = sorted(root.rglob("*.json")) if root.is_dir() else []
    dates: Counter[str] = Counter()
    cohorts: Counter[str] = Counter()
    event_types: Counter[str] = Counter()
    plan_actions: Counter[str] = Counter()
    raw_plan_actions: Counter[str] = Counter()
    observation_actions: Counter[str] = Counter()
    observation_outcomes: dict[str, Counter[str]] = {}
    trajectory_actions: Counter[str] = Counter()
    trajectory_sequences: Counter[str] = Counter()
    unknown_actions: Counter[str] = Counter()
    pending_status: Counter[str] = Counter()
    parsed_files = 0
    parse_error_count = 0
    thread_count = 0
    query_count = 0
    trajectory_step_count = 0
    image_session_count = 0

    declared = set(tool_registry.get_declared_tool_names())
    for path in files:
        dates[path.parent.name] += 1
        cohorts[_cohort(path)] += 1
        try:
            session = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            parse_error_count += 1
            continue
        if not isinstance(session, dict):
            parse_error_count += 1
            continue
        parsed_files += 1
        pending = session.get("pending_clarification")
        if isinstance(pending, dict):
            pending_status[str(pending.get("status") or "present")] += 1

        threads = session.get("threads") if isinstance(session.get("threads"), dict) else {}
        thread_count += len(threads)
        has_image = False
        for thread in threads.values():
            if not isinstance(thread, dict):
                continue
            has_image = has_image or bool(str(thread.get("last_image_path") or "").strip())
            for trajectory in _trajectory_values(thread.get("query_trajectories")):
                query_count += 1
                sequence: list[str] = []
                steps = trajectory.get("steps") if isinstance(trajectory.get("steps"), list) else []
                for step in steps:
                    if not isinstance(step, dict):
                        continue
                    action = _normalize_action(step.get("action"))
                    trajectory_actions[action] += 1
                    trajectory_step_count += 1
                    sequence.append(action)
                    if action not in declared and action not in NON_TOOL_ACTIONS:
                        unknown_actions[action] += 1
                if sequence:
                    trajectory_sequences[" -> ".join(sequence)] += 1

            ledger = thread.get("raw_ledger") if isinstance(thread.get("raw_ledger"), list) else []
            for event in ledger:
                if not isinstance(event, dict):
                    continue
                event_type = str(event.get("event_type") or "UNKNOWN").strip().upper()
                event_types[event_type] += 1
                payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
                if event_type == "PLAN_DECISION":
                    raw_action = str(payload.get("action") or "").strip()
                    action = _normalize_action(raw_action)
                    raw_plan_actions[raw_action] += 1
                    plan_actions[action] += 1
                    if action not in declared and action not in NON_TOOL_ACTIONS:
                        unknown_actions[action] += 1
                elif event_type == "OBSERVATION":
                    action = _normalize_action(
                        payload.get("_action")
                        or payload.get("action")
                        or event.get("observation")
                    )
                    observation_actions[action] += 1
                    observation_outcomes.setdefault(action, Counter())[
                        _success_bucket(payload)
                    ] += 1
                    if action not in declared and action not in NON_TOOL_ACTIONS | {"unknown"}:
                        unknown_actions[action] += 1
        image_session_count += int(has_image)

    return {
        "privacy": {
            "contains_user_text": False,
            "contains_client_addresses": False,
            "description": "Aggregate counts only; no query, answer, or client address is emitted.",
        },
        "files": {
            "discovered": len(files),
            "parsed": parsed_files,
            "parse_errors": parse_error_count,
        },
        "dates": dict(sorted(dates.items())),
        "cohorts": dict(sorted(cohorts.items())),
        "sessions_with_images": image_session_count,
        "threads": thread_count,
        "queries": query_count,
        "trajectory_steps": trajectory_step_count,
        "event_types": dict(event_types.most_common()),
        "plan_actions": dict(plan_actions.most_common()),
        "raw_plan_actions": dict(raw_plan_actions.most_common()),
        "observation_actions": dict(observation_actions.most_common()),
        "observation_outcomes": {
            action: dict(counts)
            for action, counts in sorted(observation_outcomes.items())
        },
        "trajectory_actions": dict(trajectory_actions.most_common()),
        "top_trajectory_sequences": dict(trajectory_sequences.most_common(20)),
        "unknown_actions": dict(unknown_actions.most_common()),
        "pending_clarification": dict(pending_status),
    }


def audit_llm_debug(debug_dir: Path) -> dict[str, Any]:
    """Aggregate Planner/Answerer debug records without emitting prompts or responses."""
    root = Path(debug_dir).resolve()
    files = sorted(root.rglob("*.json")) if root.is_dir() else []
    dates: Counter[str] = Counter()
    cohorts: Counter[str] = Counter()
    stages: Counter[str] = Counter()
    models: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    decision_types: Counter[str] = Counter()
    parse_errors = 0
    raw_json_parse_failures = 0
    planner_response_count = 0
    for path in files:
        dates[path.parent.name] += 1
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            parse_errors += 1
            continue
        if not isinstance(record, dict):
            parse_errors += 1
            continue
        session_id = str(record.get("session_id") or "")
        if "smoke" in session_id or session_id.startswith("capability_reproduction"):
            cohort = "synthetic_smoke"
        elif session_id.startswith("planner_grpo_"):
            cohort = "grpo_eval"
        else:
            cohort = "runtime"
        cohorts[cohort] += 1
        stage = str(record.get("stage") or "unknown")
        stages[stage] += 1
        payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
        if stage.endswith("request"):
            models[str(payload.get("model") or "unknown")] += 1
        if stage not in {"planner_response", "planner_retry_response"}:
            continue
        planner_response_count += 1
        raw_response = payload.get("raw_response")
        try:
            response = json.loads(raw_response) if isinstance(raw_response, str) else {}
        except json.JSONDecodeError:
            raw_json_parse_failures += 1
            continue
        if not isinstance(response, dict):
            raw_json_parse_failures += 1
            continue
        action = _normalize_action(response.get("action"))
        actions[action] += 1
        decision_types[str(response.get("decision_type") or "unknown")] += 1
    return {
        "privacy": {
            "contains_prompts": False,
            "contains_responses": False,
            "contains_session_ids": False,
            "description": "Aggregate stage, model, action, and parse counts only.",
        },
        "files": {
            "discovered": len(files),
            "parsed": len(files) - parse_errors,
            "parse_errors": parse_errors,
        },
        "dates": dict(sorted(dates.items())),
        "cohorts": dict(sorted(cohorts.items())),
        "stages": dict(stages.most_common()),
        "models": dict(models.most_common()),
        "planner_responses": planner_response_count,
        "raw_json_parse_failures": raw_json_parse_failures,
        "actions": dict(actions.most_common()),
        "decision_types": dict(decision_types.most_common()),
    }
