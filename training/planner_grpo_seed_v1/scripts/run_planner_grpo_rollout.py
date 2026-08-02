#!/usr/bin/env python3
"""Run Planner rollouts for GRPO seed cases without executing real tools."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
for import_root in (ROOT / "src", ROOT, ROOT / "demo"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from capa import agent  # noqa: E402
from capa import memory as ms  # noqa: E402


class PlannerRolloutTimeout(BaseException):
    """Raised when one Planner rollout step exceeds the configured timeout."""


def _alarm_handler(signum: int, frame: Any) -> None:
    raise PlannerRolloutTimeout("planner rollout step timed out")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: row must be object")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def make_session(case: dict[str, Any]) -> dict[str, Any]:
    cid = str(case.get("case_id") or "unknown_case").strip()
    setup = case.get("setup") if isinstance(case.get("setup"), dict) else {}
    sid = f"planner_grpo_{cid}"
    tid = f"thread_{cid}"
    qid = f"query_{cid}"
    session = {
        "session_id": sid,
        "active_thread_id": tid,
        "active_query_id": qid,
        "raw_ledger": [],
        "query_trajectories": [],
        "thread_aux_state": {},
    }

    for index, item in enumerate(setup.get("query_trajectories") if isinstance(setup.get("query_trajectories"), list) else [], start=1):
        if not isinstance(item, dict):
            continue
        prior_qid = str(item.get("query_id") or f"prior_{index}_{cid}")
        session["query_trajectories"].append(
            {
                "session_id": sid,
                "thread_id": tid,
                "query_id": prior_qid,
                "query": str(item.get("query") or "").strip(),
                "result_summary": str(item.get("result_summary") or "").strip(),
                "steps": item.get("steps") if isinstance(item.get("steps"), list) else [],
            }
        )

    session["query_trajectories"].append(
        {
            "session_id": sid,
            "thread_id": tid,
            "query_id": qid,
            "query": str(case.get("user_query") or "").strip(),
            "result_summary": "",
            "steps": [],
        }
    )
    ms.LedgerStore.append_event(
        session,
        event_type="USER_INPUT",
        observation="user_input",
        payload={"query_id": qid, "text": str(case.get("user_query") or "").strip()},
        thread_id=tid,
    )
    ms.LedgerStore.sync_ledger_cursor(session)
    return session


def mock_observation_for_step(case: dict[str, Any], step_index: int) -> dict[str, Any]:
    for item in case.get("mock_observations") if isinstance(case.get("mock_observations"), list) else []:
        if not isinstance(item, dict):
            continue
        try:
            after_step = int(item.get("after_step") or 0)
        except (TypeError, ValueError):
            after_step = 0
        if after_step == step_index and isinstance(item.get("observation"), dict):
            return dict(item["observation"])
    return {"success": True, "summary": "mock observation for offline Planner rollout"}


def persist_mock_step(
    *,
    session: dict[str, Any],
    run_dir: Path,
    step_index: int,
    user_query: str,
    decision: dict[str, Any],
    observation: dict[str, Any],
) -> None:
    ms.MemoryProjector.persist_step(
        session,
        step_index=step_index,
        text=user_query,
        step={
            "action": str(decision.get("action") or "").strip(),
            "action_input": decision.get("action_input") if isinstance(decision.get("action_input"), dict) else {},
        },
        thought=str(decision.get("thought") or "").strip(),
        observation=observation,
        run_dir=run_dir,
    )


def expected_step_count(case: dict[str, Any]) -> int:
    expected = case.get("expected_decisions")
    if isinstance(expected, list) and expected:
        return len(expected)
    return 1


def run_case(
    case: dict[str, Any],
    *,
    model: str,
    out_root: Path,
    max_steps: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    cid = str(case.get("case_id") or "unknown_case").strip()
    user_query = str(case.get("user_query") or "").strip()
    image_path = resolve_image_path(case)
    run_dir = out_root / "runs" / cid
    run_dir.mkdir(parents=True, exist_ok=True)
    session = make_session(case)
    decisions: list[dict[str, Any]] = []
    errors: list[str] = []
    target_steps = min(max_steps, expected_step_count(case))
    # The planner must be told the SAME max_steps that the case setup pins,
    # not the default AGENT_MAX_STEPS=10. Otherwise the model sees
    # "step 2 of 10" at eval time while the SFT / GRPO training data always
    # said "step 2 of 3", and the model's "am I on the last step?" cue
    # (which drives finish_after_tool) becomes systematically wrong.
    setup = case.get("setup") if isinstance(case.get("setup"), dict) else {}
    case_max_steps = int(setup.get("max_steps") or 0) or agent.AGENT_MAX_STEPS
    planner_max_steps = min(case_max_steps, target_steps) if target_steps else case_max_steps

    for step_index in range(1, target_steps + 1):
        planner_context = ms.ContextBuilder.build_prompt_context(
            session,
            text=user_query,
            effective_image_path=image_path,
        )
        previous_handler = None
        if timeout_seconds > 0:
            previous_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(timeout_seconds)
        try:
            decision = agent.choose_agent_step_with_fallback(
                user_query,
                image_path or None,
                planner_context=planner_context,
                step_index=step_index,
                max_steps=planner_max_steps,
                model=model or None,
                debug_meta={
                    "session_id": str(session.get("session_id") or ""),
                    "run_stamp": f"planner_grpo_{cid}",
                    "run_dir": str(run_dir),
                },
            )
        except PlannerRolloutTimeout as exc:
            errors.append(str(exc))
            break
        except Exception as exc:
            errors.append(str(exc))
            break
        finally:
            if timeout_seconds > 0:
                signal.alarm(0)
                if previous_handler is not None:
                    signal.signal(signal.SIGALRM, previous_handler)
        decisions.append(decision)

        decision_type = str(decision.get("decision_type") or "").strip()
        if step_index >= target_steps or decision_type != "tool":
            break
        observation = mock_observation_for_step(case, step_index)
        persist_mock_step(
            session=session,
            run_dir=run_dir,
            step_index=step_index,
            user_query=user_query,
            decision=decision,
            observation=observation,
        )

    return {
        "case_id": cid,
        "category": str(case.get("category") or ""),
        "model": model,
        "user_query": user_query,
        "image_path": image_path,
        "decisions": decisions,
        "errors": errors,
        "run_dir": str(run_dir),
    }


def configure_generation(args: argparse.Namespace) -> None:
    if args.api_base:
        os.environ["DEMO_ROUTE_API_BASE"] = str(args.api_base).rstrip("/")
        agent.DEMO_ROUTE_API_BASE = str(args.api_base).rstrip("/")
    if args.api_key:
        os.environ["DEMO_ROUTE_API_KEY"] = str(args.api_key)
        agent.DEMO_ROUTE_API_KEY = str(args.api_key)
    if args.model:
        os.environ["DEMO_ROUTE_MODEL"] = str(args.model)
        agent.DEMO_ROUTE_MODEL = str(args.model)
    if args.temperature is not None:
        os.environ["DEMO_OPENAI_TEMPERATURE"] = str(args.temperature)
    if args.top_p is not None:
        os.environ["DEMO_OPENAI_TOP_P"] = str(args.top_p)
    if args.seed is not None:
        os.environ["DEMO_OPENAI_SEED"] = str(args.seed)
    if args.do_sample is not None:
        os.environ["DEMO_OPENAI_DO_SAMPLE"] = str(args.do_sample)
    if args.openai_timeout_seconds is not None:
        os.environ["DEMO_OPENAI_TIMEOUT_SECONDS"] = str(args.openai_timeout_seconds)
    if getattr(args, "max_tokens", None) is not None:
        os.environ["DEMO_OPENAI_MAX_TOKENS"] = str(args.max_tokens)


def configure_local_backend(args: argparse.Namespace) -> Any | None:
    if args.local_adapter_path is not None and args.local_model_path is None:
        raise ValueError("--local-adapter-path requires --local-model-path")
    if args.local_model_path is None:
        return None
    from training.planner_grpo_seed_v1.scripts.local_hf_planner_backend import (
        LocalHFPlannerBackend,
    )

    backend = LocalHFPlannerBackend(
        model_path=args.local_model_path,
        adapter_path=args.local_adapter_path,
        device=args.local_device,
        attn_implementation=args.local_attn_implementation,
    )
    agent.VLMService = backend.service_factory
    return backend


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline Planner rollouts for GRPO cases.")
    parser.add_argument("--cases", type=Path, required=True, help="GRPO case JSONL")
    parser.add_argument("--out", type=Path, required=True, help="Prediction JSONL output")
    parser.add_argument("--model", default="", help="Planner model name/endpoint identifier")
    parser.add_argument("--api-base", default="", help="OpenAI-compatible Planner API base")
    parser.add_argument(
        "--api-key",
        default="",
        help="Optional Planner API key override; otherwise use DEMO_ROUTE_API_KEY",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional case limit")
    parser.add_argument("--max-steps", type=int, default=3, help="Maximum Planner decisions per case")
    parser.add_argument("--timeout-seconds", type=int, default=45, help="Hard timeout for each Planner step")
    parser.add_argument("--openai-timeout-seconds", type=int, default=120, help="HTTP client timeout")
    parser.add_argument("--max-tokens", type=int, default=384, help="Maximum Planner completion tokens")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--do-sample", default="false")
    parser.add_argument("--local-model-path", type=Path, default=None)
    parser.add_argument("--local-adapter-path", type=Path, default=None)
    parser.add_argument("--local-device", default="cuda")
    parser.add_argument("--local-attn-implementation", default="sdpa")
    args = parser.parse_args()
    configure_generation(args)
    local_backend = configure_local_backend(args)

    cases = load_jsonl(args.cases)
    if args.limit > 0:
        cases = cases[: args.limit]
    out_root = args.out.parent / f"planner_grpo_rollout_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if args.out.exists():
        args.out.unlink()
    rows: list[dict[str, Any]] = []
    for case in cases:
        row = run_case(
            case,
            model=args.model,
            out_root=out_root,
            max_steps=max(1, int(args.max_steps or 1)),
            timeout_seconds=max(0, int(args.timeout_seconds or 0)),
        )
        rows.append(row)
        append_jsonl(args.out, row)
    summary = {
        "cases": len(rows),
        "out": str(args.out),
        "run_root": str(out_root),
        "model": args.model,
        "api_base": args.api_base,
        "inference_backend": "local_transformers" if local_backend is not None else "openai_compatible",
        "local_model_path": str(args.local_model_path.resolve()) if args.local_model_path else "",
        "local_adapter_path": str(args.local_adapter_path.resolve()) if args.local_adapter_path else "",
        "generation_config": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "do_sample": args.do_sample,
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
