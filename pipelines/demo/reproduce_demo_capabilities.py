#!/usr/bin/env python3
"""Audit and safely reproduce the CAPA demo Agent capability contracts."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capa.capabilities import build_capability_inventory, validate_capability_inventory
from capa.service_health import probe_demo_services, service_summary
from capa.session_audit import audit_llm_debug, audit_sessions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--sessions-dir", type=Path, default=ROOT / "demo/sessions")
    parser.add_argument(
        "--llm-debug-dir", type=Path, default=ROOT / "demo/llm_debug"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "reports/demo_agent_capability_reproduction.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=ROOT / "reports/demo_agent_capability_reproduction.md",
    )
    parser.add_argument("--live", action="store_true", help="Run read-only service probes.")
    parser.add_argument(
        "--include-flux",
        action="store_true",
        help="Include a credentialed, read-only Flux model-list probe.",
    )
    parser.add_argument(
        "--model-smoke",
        action="store_true",
        help="Run one synthetic Planner call and one synthetic Answerer call.",
    )
    parser.add_argument(
        "--http-smoke",
        action="store_true",
        help="Start the Demo and verify health, capability, and NDJSON /run paths.",
    )
    parser.add_argument(
        "--rex-smoke",
        action="store_true",
        help="Run Rex-Omni on the repository banner fixture.",
    )
    parser.add_argument(
        "--rex-image",
        type=Path,
        default=ROOT / "examples/images/banner.jpg",
    )
    parser.add_argument(
        "--rex-prompt",
        type=Path,
        default=(
            ROOT
            / "skills/rexomni-open-set-detection/references/prompt_example.json"
        ),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("/raid/zkq/artifacts/CAPA/capability_reproduction"),
    )
    parser.add_argument(
        "--rag-smoke-report",
        type=Path,
        default=ROOT / "reports/demo_rag_smoke.json",
    )
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--smoke-timeout", type=float, default=120.0)
    return parser.parse_args()


def _model_smoke() -> dict[str, Any]:
    from capa import agent

    result: dict[str, Any] = {}
    planner_started = time.perf_counter()
    try:
        step = agent.choose_agent_step_llm(
            "请用一句话解释什么是过拟合。",
            None,
            planner_context={},
            step_index=1,
            max_steps=3,
            model=os.environ.get("DEMO_SMOKE_ROUTE_MODEL", "Qwen3.5-4B"),
            debug_meta={
                "session_id": "capability_reproduction",
                "run_stamp": "model_smoke",
                "run_dir": "",
            },
        )
        result["planner"] = {
            "status": "passed",
            "decision_type": step.get("decision_type"),
            "action": step.get("action"),
            "elapsed_ms": round((time.perf_counter() - planner_started) * 1000, 3),
            "metrics": step.get("_planner_metrics", {}),
        }
    except Exception as exc:
        result["planner"] = {
            "status": "failed",
            "elapsed_ms": round((time.perf_counter() - planner_started) * 1000, 3),
            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
        }

    answer_started = time.perf_counter()
    try:
        answer = agent.generate_final_answer_llm(
            answerer_input={
                "user_query": "请用一句话解释什么是过拟合。",
                "evidence": {"retrieved_chunks": [], "query_trajectories": []},
            },
            mode="direct",
            model=os.environ.get("DEMO_SMOKE_ANSWER_MODEL", "Qwen3.5-4B"),
            debug_meta={
                "session_id": "capability_reproduction",
                "run_stamp": "model_smoke",
                "step_index": 1,
                "run_dir": "",
            },
        )
        result["answerer"] = {
            "status": "passed" if answer.strip() else "failed",
            "answer_characters": len(answer.strip()),
            "elapsed_ms": round((time.perf_counter() - answer_started) * 1000, 3),
        }
    except Exception as exc:
        result["answerer"] = {
            "status": "failed",
            "elapsed_ms": round((time.perf_counter() - answer_started) * 1000, 3),
            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
        }
    return result


def _load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _http_smoke(*, root: Path, timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    port = _free_loopback_port()
    base_url = f"http://127.0.0.1:{port}"
    session_id = f"capability_http_smoke_{int(time.time())}"
    process = subprocess.Popen(
        [
            sys.executable,
            str(root / "demo/demo_server.py"),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    client = requests.Session()
    client.trust_env = False
    cleanup_ok = False
    try:
        deadline = time.monotonic() + min(30.0, timeout)
        health: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Demo exited during startup with code {process.returncode}")
            try:
                response = client.get(f"{base_url}/health", timeout=1.0)
                if response.ok:
                    health = response.json()
                    break
            except requests.RequestException:
                pass
            time.sleep(0.2)
        if not isinstance(health, dict):
            raise TimeoutError("Demo health endpoint did not become ready")

        capabilities_response = client.get(
            f"{base_url}/health/capabilities", timeout=5.0
        )
        capabilities_response.raise_for_status()
        capabilities = capabilities_response.json().get("capabilities") or []

        run_response = client.post(
            f"{base_url}/run",
            files={
                "text": (
                    None,
                    "不访问公司知识库，请用一句话解释什么是过拟合。",
                ),
                "session_id": (None, session_id),
            },
            stream=True,
            timeout=(5.0, timeout),
        )
        run_response.raise_for_status()
        events: list[dict[str, Any]] = []
        for raw_line in run_response.iter_lines(decode_unicode=True):
            line = str(raw_line or "").strip()
            if not line:
                continue
            event = json.loads(line)
            if isinstance(event, dict):
                events.append(event)
        done = next(
            (event for event in reversed(events) if event.get("type") == "done"),
            {},
        )
        event_types = sorted(
            {str(event.get("type") or "") for event in events if event.get("type")}
        )
        final_characters = sum(
            len(str(event.get("text") or ""))
            for event in events
            if event.get("type") == "final_answer"
        )
        planner_actions = [
            str((event.get("decision") or {}).get("action") or "")
            for event in events
            if event.get("type") == "meta"
            and isinstance(event.get("decision"), dict)
        ]
        cleanup_response = client.post(
            f"{base_url}/session/delete",
            json={"session_id": session_id},
            timeout=5.0,
        )
        cleanup_ok = cleanup_response.ok and bool(
            cleanup_response.json().get("ok")
        )
        passed = bool(
            health.get("ok")
            and len(capabilities) == len(build_capability_inventory(root))
            and done.get("ok") is True
            and final_characters > 0
            and "final_answer" in event_types
        )
        return {
            "status": "passed" if passed else "failed",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "static_health_ok": bool(health.get("ok")),
            "capability_count": len(capabilities),
            "event_types": event_types,
            "planner_actions": planner_actions,
            "final_answer_characters": final_characters,
            "done_ok": done.get("ok") is True,
            "synthetic_session_cleaned": cleanup_ok,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            "synthetic_session_cleaned": cleanup_ok,
        }
    finally:
        client.close()
        if process.poll() is None:
            process.terminate()
        try:
            process.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=5.0)


def _rex_smoke(
    *,
    root: Path,
    image_path: Path,
    prompt_path: Path,
    artifact_root: Path,
    timeout: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = artifact_root.resolve() / f"rex-smoke-{stamp}.json"
    base_url = (
        os.environ.get("DEMO_REX_BASE_URL")
        or os.environ.get("DEMO_LLM_API_BASE")
        or "http://10.111.32.253:8000/v1"
    )
    command = [
        sys.executable,
        str(root / "skills/rexomni-open-set-detection/scripts/run_detection.py"),
        "--images",
        str(image_path.resolve()),
        "--prompt",
        str(prompt_path.resolve()),
        "--base-url",
        base_url,
        "--out",
        str(out_path),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:
        return {
            "status": "failed",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "error": f"{type(exc).__name__}: {str(exc)[:500]}",
        }
    if completed.returncode != 0:
        error = (completed.stderr or completed.stdout or "Rex smoke failed").strip()
        return {
            "status": "failed",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "exit_code": completed.returncode,
            "error": error[-1000:],
        }
    coco = _load_json_object(out_path)
    if coco is None:
        return {
            "status": "failed",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "error": "Rex completed without a valid COCO JSON artifact",
        }
    images = coco.get("images") if isinstance(coco.get("images"), list) else []
    categories = (
        coco.get("categories") if isinstance(coco.get("categories"), list) else []
    )
    annotations = (
        coco.get("annotations")
        if isinstance(coco.get("annotations"), list)
        else []
    )
    image_by_id = {
        item.get("id"): item for item in images if isinstance(item, dict)
    }
    bbox_valid = True
    for annotation in annotations:
        if not isinstance(annotation, dict):
            bbox_valid = False
            break
        bbox = annotation.get("bbox")
        image = image_by_id.get(annotation.get("image_id"), {})
        if not isinstance(bbox, list) or len(bbox) != 4:
            bbox_valid = False
            break
        try:
            x, y, width, height = (float(value) for value in bbox)
            image_width = float(image.get("width"))
            image_height = float(image.get("height"))
        except (TypeError, ValueError):
            bbox_valid = False
            break
        if not (
            x >= 0
            and y >= 0
            and width > 0
            and height > 0
            and x + width <= image_width + 1
            and y + height <= image_height + 1
        ):
            bbox_valid = False
            break
    passed = bool(images and categories and annotations and bbox_valid)
    return {
        "status": "passed" if passed else "failed",
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "fixture": image_path.name,
        "categories": [
            str(item.get("name") or "")
            for item in categories
            if isinstance(item, dict)
        ],
        "annotations": len(annotations),
        "bbox_valid": bbox_valid,
        "artifact": str(out_path),
    }


def _service_status_for_capability(item: dict, probes: dict) -> str:
    if not probes:
        return "not_probed"
    service_statuses: list[str] = []
    for service in item["services"]:
        if service == "rag":
            rag_statuses = {
                str((probes.get("rag_playbook") or {}).get("status") or "offline"),
                str((probes.get("rag_unified") or {}).get("status") or "offline"),
            }
            if rag_statuses == {"online"}:
                service_statuses.append("online")
            elif rag_statuses & {"online", "degraded"}:
                service_statuses.append("degraded")
            else:
                service_statuses.append("offline")
        else:
            service_statuses.append(
                str((probes.get(service) or {}).get("status") or "offline")
            )
    if service_statuses and all(status == "online" for status in service_statuses):
        return "online"
    if any(status in {"not_probed", "unconfigured"} for status in service_statuses):
        return "unverified"
    return "partial_or_offline"


def _render_markdown(payload: dict[str, Any]) -> str:
    static = payload["static_contract"]
    sessions = payload["historical_sessions"]
    llm_debug = payload["historical_llm_debug"]
    lines = [
        "# Demo Agent Capability Reproduction",
        "",
        "## 结论",
        "",
        (
            "静态能力契约已完整复现。"
            if static["passed"]
            else "静态能力契约存在缺口，不能宣称完整复现。"
        ),
        "外部服务状态与代码契约分开记录；有脚本不代表对应服务当前在线。",
        "",
        "## 能力矩阵",
        "",
        "| Tool | Owner | Components | Services | Image | Side effect | Historical observations | Live status |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    observations = sessions.get("observation_actions", {})
    for item in payload["capabilities"]:
        lines.append(
            f"| `{item['tool_name']}` | {item['runtime_owner']} | "
            f"{', '.join(item['components'])} | {', '.join(item['services'])} | "
            f"{str(item['requires_image']).lower()} | {str(item['side_effecting']).lower()} | "
            f"{observations.get(item['tool_name'], 0)} | {item['live_status']} |"
        )
    files = sessions.get("files", {})
    lines.extend(
        [
            "",
            "## 历史证据",
            "",
            f"- Session JSON：发现 {files.get('discovered', 0)}，成功解析 {files.get('parsed', 0)}，错误 {files.get('parse_errors', 0)}。",
            f"- Threads：{sessions.get('threads', 0)}；queries：{sessions.get('queries', 0)}；轨迹 steps：{sessions.get('trajectory_steps', 0)}。",
            f"- 带图片 session：{sessions.get('sessions_with_images', 0)}。",
            f"- Session cohort：`{json.dumps(sessions.get('cohorts', {}), ensure_ascii=False)}`。",
            f"- 未知历史动作：`{json.dumps(sessions.get('unknown_actions', {}), ensure_ascii=False)}`。",
            "- 报告仅包含聚合计数，不包含用户 query、回答或客户端地址。",
            f"- LLM debug：{llm_debug.get('files', {}).get('parsed', 0)} 条；Planner response {llm_debug.get('planner_responses', 0)} 条；原始 JSON 解析失败 {llm_debug.get('raw_json_parse_failures', 0)} 条。",
            f"- LLM debug cohort：`{json.dumps(llm_debug.get('cohorts', {}), ensure_ascii=False)}`；不输出 prompt、response 或 session ID。",
            "",
            "## 服务探活",
            "",
            "| Service | Status | Detail |",
            "|---|---|---|",
        ]
    )
    probes = payload.get("service_probes", {})
    if probes:
        for name, probe in probes.items():
            detail = probe.get("error") or probe.get("url") or probe.get("command") or ""
            lines.append(f"| `{name}` | {probe.get('status', '')} | {str(detail)[:180]} |")
    else:
        lines.append("| _not probed_ | | 使用 `--live` 执行只读探活 |")
    if payload.get("model_smoke"):
        lines.extend(
            [
                "",
                "## Model Smoke",
                "",
                f"- Planner：`{payload['model_smoke'].get('planner', {}).get('status', '')}`。",
                f"- Answerer：`{payload['model_smoke'].get('answerer', {}).get('status', '')}`。",
            ]
        )
    if payload.get("http_smoke"):
        http_smoke = payload["http_smoke"]
        lines.extend(
            [
                "",
                "## Demo HTTP Smoke",
                "",
                f"- 结果：`{http_smoke.get('status', '')}`；静态健康 `{http_smoke.get('static_health_ok', False)}`；工具数 {http_smoke.get('capability_count', 0)}。",
                f"- NDJSON done `{http_smoke.get('done_ok', False)}`；最终回答字符数 {http_smoke.get('final_answer_characters', 0)}；合成会话已清理 `{http_smoke.get('synthetic_session_cleaned', False)}`。",
            ]
        )
    recorded_smokes = payload.get("recorded_smokes", {})
    if recorded_smokes:
        rag = recorded_smokes.get("rag") or {}
        rex = recorded_smokes.get("rex") or {}
        lines.extend(
            [
                "",
                "## Recorded Smokes",
                "",
                f"- RAG：`{rag.get('status', 'not_run')}`；HTTP contract `{rag.get('http_contract_passed', False)}`；content ready `{rag.get('content_ready', False)}`。",
                f"- Rex-Omni：`{rex.get('status', 'not_run')}`；boxes {rex.get('annotations', 0)}；bbox valid `{rex.get('bbox_valid', False)}`。",
            ]
        )
    lines.extend(
        [
            "",
            "## 使用边界",
            "",
            "默认复现不会调用 Flux 或其它有成本/副作用的动作；Adela 已从默认能力范围排除。",
            "这些能力必须在隔离测试资产上显式执行端到端验收后，才能标记为 live reproduced。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    errors = validate_capability_inventory(root)
    inventory = build_capability_inventory(root)
    sessions = audit_sessions(args.sessions_dir)
    probes = (
        probe_demo_services(
            root, timeout=args.timeout, include_flux=args.include_flux
        )
        if args.live
        else {}
    )
    for item in inventory:
        item["live_status"] = _service_status_for_capability(item, probes)
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "static_contract": {"passed": not errors, "errors": errors},
        "capabilities": inventory,
        "historical_sessions": sessions,
        "historical_llm_debug": audit_llm_debug(args.llm_debug_dir),
        "service_probes": probes,
        "service_summary": service_summary(probes) if probes else {},
        "recorded_smokes": {},
    }
    rag_smoke = _load_json_object(args.rag_smoke_report)
    if rag_smoke is not None:
        payload["recorded_smokes"]["rag"] = rag_smoke
    if args.model_smoke:
        payload["model_smoke"] = _model_smoke()
    if args.http_smoke:
        payload["http_smoke"] = _http_smoke(
            root=root,
            timeout=args.smoke_timeout,
        )
    if args.rex_smoke:
        payload["recorded_smokes"]["rex"] = _rex_smoke(
            root=root,
            image_path=args.rex_image,
            prompt_path=args.rex_prompt,
            artifact_root=args.artifact_root,
            timeout=args.smoke_timeout,
        )
    if args.model_smoke or args.http_smoke:
        # Refresh after synthetic calls so cohort counts include, but isolate, them.
        payload["historical_sessions"] = audit_sessions(args.sessions_dir)
        payload["historical_llm_debug"] = audit_llm_debug(args.llm_debug_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(_render_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "passed" if not errors else "failed",
                "tools": len(inventory),
                "sessions": sessions["files"]["parsed"],
                "output_json": str(args.output_json),
                "output_md": str(args.output_md),
            },
            ensure_ascii=False,
        )
    )
    smoke_failed = (
        args.rex_smoke
        and payload["recorded_smokes"].get("rex", {}).get("status") != "passed"
    ) or (
        args.model_smoke
        and any(
            result.get("status") != "passed"
            for result in payload.get("model_smoke", {}).values()
        )
    ) or (
        args.http_smoke and payload.get("http_smoke", {}).get("status") != "passed"
    )
    if errors or smoke_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
