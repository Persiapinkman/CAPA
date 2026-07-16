#!/usr/bin/env python3
"""Run reproducible non-Adela Demo Agent end-to-end checks."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# This harness intentionally validates the current non-Adela product scope.
os.environ["CAPA_ENABLE_ADELA"] = "0"
from capa.tools import registry  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="", help="Use an already running Demo server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument(
        "--image",
        type=Path,
        default=ROOT / "examples/images/fisherman.jpg",
    )
    parser.add_argument("--include-migration", action="store_true")
    parser.add_argument("--include-flux", action="store_true")
    parser.add_argument("--include-pipeline", action="store_true")
    parser.add_argument(
        "--allow-side-effects",
        action="store_true",
        help="Required before Flux or pipeline generation is executed.",
    )
    parser.add_argument("--pipeline-images", type=int, default=1)
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated case names for a focused rerun.",
    )
    parser.add_argument("--timeout", type=float, default=420.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/demo_full_e2e_smoke.json",
    )
    return parser.parse_args()


def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind((host, 0))
        return int(handle.getsockname()[1])


def _wait_for_health(client: requests.Session, base_url: str, timeout: float) -> dict:
    deadline = time.monotonic() + min(timeout, 45.0)
    last_error = ""
    while time.monotonic() < deadline:
        try:
            response = client.get(f"{base_url}/health", timeout=2.0)
            if response.ok:
                payload = response.json()
                if isinstance(payload, dict):
                    return payload
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(0.25)
    raise TimeoutError(f"Demo health did not become ready: {last_error}")


def _case_definitions(args: argparse.Namespace) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = [
        {
            "name": "answerer",
            "text": "不要访问公司知识库，请用一句话解释什么是过拟合。",
            "expected_action": "answerer",
            "required_event": "final_answer",
        },
        {
            "name": "rag",
            "text": "请使用 rag_answer 查询 safety_rope v0.2.1 支持哪些输出结果。",
            "expected_action": "rag_answer",
            "required_event": "direct_reply",
        },
        {
            "name": "qwen",
            "text": "请明确使用 qwen_detection 检测图片中的钓鱼人员，只做单图检测。",
            "expected_action": "qwen_detection",
            "required_event": "annotated",
            "image": args.image,
        },
        {
            "name": "rexomni",
            "text": "请明确使用 rexomni_detection 检测图片中的钓鱼人员，只做单图检测。",
            "expected_action": "rexomni_detection",
            "required_event": "annotated",
            "image": args.image,
        },
    ]
    if args.include_migration:
        cases.append(
            {
                "name": "migration",
                "text": (
                    "请明确使用 migration_advisor，为河道监控中的钓鱼人员检测需求生成迁移方案；"
                    "当前没有样例图，只能使用知识库证据。"
                ),
                "expected_action": "migration_advisor",
                "required_event": "migration_advisor_report",
            }
        )
    if args.include_flux:
        cases.append(
            {
                "name": "flux",
                "text": (
                    "请明确使用 flux-image-generation 生成一张写实河岸远景图："
                    "画面中有两名正在钓鱼的人，不要添加文字。"
                ),
                "expected_action": "flux-image-generation",
                "required_event": "generated_one",
                "side_effecting": True,
            }
        )
    if args.include_pipeline:
        cases.append(
            {
                "name": "pipeline",
                "text": (
                    "请明确使用 pipeline_eval 完成目标检测评测：基于参考图扩增样本，"
                    "用 Qwen 和 Rex-Omni 检测钓鱼人员并生成评测报告。"
                ),
                "expected_action": "pipeline_eval",
                "required_event": "evaluation",
                "image": args.image,
                "side_effecting": True,
            }
        )
    return cases


def _planner_actions(events: list[dict]) -> list[str]:
    actions: list[str] = []
    for event in events:
        decision = event.get("decision") if isinstance(event.get("decision"), dict) else {}
        raw = str(decision.get("action") or "").strip()
        if raw:
            normalized = registry.normalize_tool_action(raw)
            if normalized not in actions:
                actions.append(normalized)
    return actions


def _generation_quality_summary(events: list[dict]) -> dict[str, Any]:
    event = next(
        (item for item in events if item.get("type") == "generation_quality"),
        {},
    )
    data = event.get("data") if isinstance(event.get("data"), dict) else event
    gate = data.get("quality_gate_passed")
    if gate is None:
        gate = data.get("passed")
    return {
        "gate": gate,
        "content_compliance_checked": bool(data.get("content_compliance_checked")),
        "warnings": data.get("warnings") if isinstance(data.get("warnings"), list) else [],
    }


def _run_case(
    client: requests.Session,
    base_url: str,
    case: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    session_id = f"demo_e2e_{case['name']}_{uuid.uuid4().hex[:12]}"
    image_path = case.get("image")
    files: list[tuple[str, tuple[Any, ...]]] = [
        ("text", (None, str(case["text"]))),
        ("session_id", (None, session_id)),
    ]
    image_handle = None
    cleanup_ok = False
    result: dict[str, Any] = {
        "name": case["name"],
        "runtime_status": "failed",
    }
    try:
        if image_path:
            path = Path(image_path).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            image_handle = path.open("rb")
            files.append(("image", (path.name, image_handle, "image/jpeg")))
        response = client.post(
            f"{base_url}/run",
            files=files,
            stream=True,
            timeout=(10.0, timeout),
        )
        response.raise_for_status()
        events: list[dict] = []
        for raw_line in response.iter_lines(decode_unicode=True):
            line = str(raw_line or "").strip()
            if not line:
                continue
            event = json.loads(line)
            if isinstance(event, dict):
                events.append(event)
        event_types = sorted(
            {str(event.get("type") or "") for event in events if event.get("type")}
        )
        actions = _planner_actions(events)
        errors = [
            str(event.get("message") or event.get("error") or "")[:300]
            for event in events
            if event.get("type") == "error"
        ]
        done = next(
            (event for event in reversed(events) if event.get("type") == "done"),
            {},
        )
        expected_action = str(case["expected_action"])
        required_event = str(case["required_event"])
        runtime_passed = bool(
            done.get("ok") is True
            and not errors
            and expected_action in actions
            and required_event in event_types
        )
        quality = _generation_quality_summary(events)
        migration_report = next(
            (event.get("report") for event in events if event.get("type") == "migration_advisor_report"),
            {},
        )
        audit = (
            migration_report.get("evidence_audit")
            if isinstance(migration_report, dict)
            and isinstance(migration_report.get("evidence_audit"), dict)
            else {}
        )
        result = {
            "name": case["name"],
            "runtime_status": "passed" if runtime_passed else "failed",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "expected_action": expected_action,
            "planner_actions": actions,
            "required_event": required_event,
            "event_types": event_types,
            "error_count": len(errors),
            "errors": errors,
            "done_ok": done.get("ok") is True,
            "generated_images": sum(event.get("type") == "generated_one" for event in events),
            "annotated_images": sum(event.get("type") == "annotated" for event in events),
            "generation_quality_gate": quality["gate"],
            "content_compliance_checked": quality["content_compliance_checked"],
            "generation_quality_warnings": quality["warnings"],
            "validated_migration_facts": int(audit.get("validated_fact_count") or 0),
            "migration_grounding": str(audit.get("grounding") or ""),
        }
    except Exception as exc:
        result = {
            "name": case["name"],
            "runtime_status": "failed",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "error": f"{type(exc).__name__}: {str(exc)[:500]}",
        }
    finally:
        if image_handle is not None:
            image_handle.close()
        try:
            response = client.post(
                f"{base_url}/session/delete",
                json={"session_id": session_id},
                timeout=10.0,
            )
            first_delete_ok = response.ok and bool(response.json().get("ok"))
            time.sleep(0.25)
            verify = client.post(
                f"{base_url}/session/delete",
                json={"session_id": session_id},
                timeout=10.0,
            )
            cleanup_ok = first_delete_ok and verify.status_code == 404
        except Exception:
            cleanup_ok = False
        result["session_cleaned"] = cleanup_ok
        if not cleanup_ok:
            result["runtime_status"] = "failed"
    return result


def main() -> None:
    args = parse_args()
    if (args.include_flux or args.include_pipeline) and not args.allow_side_effects:
        raise SystemExit("--allow-side-effects is required for Flux or pipeline execution")
    args.pipeline_images = max(1, min(5, int(args.pipeline_images)))
    cases = _case_definitions(args)
    selected = {name.strip() for name in str(args.only or "").split(",") if name.strip()}
    if selected:
        known = {str(case["name"]) for case in cases}
        unknown = sorted(selected - known)
        if unknown:
            raise SystemExit(f"unknown or disabled smoke cases: {', '.join(unknown)}")
        cases = [case for case in cases if case["name"] in selected]
    process: subprocess.Popen | None = None
    client = requests.Session()
    client.trust_env = False
    started = time.perf_counter()
    try:
        if args.server_url:
            base_url = args.server_url.rstrip("/")
        else:
            port = args.port or _free_port(args.host)
            base_url = f"http://{args.host}:{port}"
            env = dict(os.environ)
            env["CAPA_ENABLE_ADELA"] = "0"
            env["DEMO_PIPELINE_NUM_GENERATED_IMAGES"] = str(args.pipeline_images)
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT / "demo/demo_server.py"),
                    "--host",
                    args.host,
                    "--port",
                    str(port),
                ],
                cwd=ROOT,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
        health = _wait_for_health(client, base_url, args.timeout)
        capability_response = client.get(f"{base_url}/health/capabilities", timeout=10.0)
        capability_response.raise_for_status()
        capability_rows = capability_response.json().get("capabilities") or []
        active_tools = sorted(
            str(item.get("tool_name") or "")
            for item in capability_rows
            if isinstance(item, dict) and item.get("tool_name")
        )
        expected_tools = sorted(registry.get_declared_tool_names())
        capability_contract_ok = bool(
            active_tools == expected_tools and "adela_cli_eval" not in active_tools
        )
        results = [
            _run_case(client, base_url, case, args.timeout)
            for case in cases
        ]
        runtime_passed = capability_contract_ok and all(
            result.get("runtime_status") == "passed" for result in results
        )
        blockers: list[str] = []
        pipeline_result = next(
            (result for result in results if result.get("name") == "pipeline"),
            None,
        )
        if pipeline_result is None:
            blockers.append("pipeline was not executed, so generation quality was not evaluated")
        elif pipeline_result.get("generation_quality_gate") is not True:
            blockers.append("pipeline generation diversity/quality gate did not pass")
        elif pipeline_result.get("content_compliance_checked") is not True:
            blockers.append("pipeline content compliance was not independently checked")
        migration_result = next(
            (result for result in results if result.get("name") == "migration"),
            None,
        )
        if migration_result is None:
            blockers.append("migration advisor was not executed, so evidence grounding was not evaluated")
        elif migration_result.get("migration_grounding") != "validated_quote_and_source_id":
            blockers.append("migration report did not expose validated evidence grounding")
        if not runtime_passed:
            blockers.append("one or more runtime capability checks failed")
        readiness_evaluated = pipeline_result is not None and migration_result is not None
        readiness = (
            "passed"
            if readiness_evaluated and runtime_passed and not blockers
            else "blocked"
            if readiness_evaluated
            else "not_evaluated"
        )
        payload = {
            "schema_version": "demo-full-e2e-v1",
            "base_url": base_url,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "health_ok": bool(health.get("ok")),
            "active_tools": active_tools,
            "expected_tools": expected_tools,
            "adela_excluded": "adela_cli_eval" not in active_tools,
            "capability_contract_ok": capability_contract_ok,
            "runtime_status": "passed" if runtime_passed else "failed",
            "rl_readiness": readiness,
            "rl_blockers": blockers,
            "cases": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, ensure_ascii=False))
        if not runtime_passed:
            raise SystemExit(1)
    finally:
        client.close()
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=8.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)


if __name__ == "__main__":
    main()
