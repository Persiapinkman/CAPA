#!/usr/bin/env python3
"""CAPA reproduction preflight checker.

Validates hardware, repository layout, virtualenvs, model artifacts and external
service connectivity BEFORE any training / evaluation / demo work. Every check
is either ``required`` (blocks reproduction) or ``optional`` (informational).

Usage
-----
    python scripts/reproduce_preflight.py --out reports/preflight_YYYY-MM-DD.json
    python scripts/reproduce_preflight.py --check hardware --check env
    python scripts/reproduce_preflight.py --strict     # exit 1 if any required fails

The script is intentionally dependency-free: only stdlib is used so it can run
before the two venvs exist.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_ROOT = Path(os.environ.get("CAPA_STORAGE_ROOT", "/raid/zkq"))
DEFAULT_MODEL_DIR = ARTIFACT_ROOT / "models" / "Qwen2.5-7B-Instruct"
DEFAULT_OUTPUTS_DIR = ARTIFACT_ROOT / "artifacts" / "CAPA" / "outputs"

CHECK_GROUPS = ("hardware", "repo", "env", "data", "models", "services")


@dataclass
class CheckResult:
    name: str
    group: str
    required: bool
    ok: bool
    message: str
    detail: dict[str, Any] = field(default_factory=dict)


# ---------- primitives ----------


def _run(cmd: list[str], timeout: int = 5) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except FileNotFoundError:
        return 127, "", f"not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout}s"


def _http_ok(url: str, timeout: int = 4) -> tuple[bool, str]:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return 200 <= resp.status < 500, f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        # 4xx still means "server is reachable"
        return exc.code < 500, f"HTTP {exc.code}"
    except (urllib.error.URLError, socket.timeout, OSError) as exc:
        return False, f"unreachable: {exc}"


def _tcp_ok(host: str, port: int, timeout: float = 2.0) -> tuple[bool, str]:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, f"tcp {host}:{port} open"
    except OSError as exc:
        return False, f"tcp {host}:{port} closed ({exc})"


# ---------- checks ----------


def check_python(results: list[CheckResult]) -> None:
    ver = sys.version_info
    ok = (ver.major, ver.minor) == (3, 10)
    results.append(
        CheckResult(
            name="python_version",
            group="env",
            required=True,
            ok=ok,
            message=f"Python {ver.major}.{ver.minor}.{ver.micro} (require 3.10.x)",
            detail={"version": list(ver[:3])},
        )
    )


def check_hardware(results: list[CheckResult]) -> None:
    rc, out, _ = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        timeout=6,
    )
    if rc != 0:
        results.append(
            CheckResult(
                name="nvidia_smi",
                group="hardware",
                required=True,
                ok=False,
                message="nvidia-smi not runnable; GPU driver missing?",
            )
        )
        return
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    results.append(
        CheckResult(
            name="gpu_inventory",
            group="hardware",
            required=True,
            ok=bool(lines),
            message=f"{len(lines)} GPU(s) detected",
            detail={"gpus": lines},
        )
    )
    is_v100 = all("V100" in ln for ln in lines) if lines else False
    results.append(
        CheckResult(
            name="gpu_is_v100_fp16",
            group="hardware",
            required=False,
            ok=is_v100,
            message=(
                "V100 confirmed; keep dtype=fp16 attn=sdpa"
                if is_v100
                else "Non-V100 detected; fp16 lock is optional but bf16 disallowed on V100"
            ),
        )
    )
    rc_c, out_c, _ = _run(["nvcc", "--version"], timeout=4)
    results.append(
        CheckResult(
            name="cuda_toolkit",
            group="hardware",
            required=False,
            ok=(rc_c == 0),
            message=(out_c.splitlines()[-1] if out_c else "nvcc not in PATH"),
        )
    )
    # disk on /raid/zkq
    if ARTIFACT_ROOT.exists():
        stat = shutil.disk_usage(str(ARTIFACT_ROOT))
        free_gb = stat.free / (1024**3)
        results.append(
            CheckResult(
                name="artifact_disk_free",
                group="hardware",
                required=True,
                ok=free_gb > 200,
                message=f"{ARTIFACT_ROOT} free={free_gb:.1f} GiB (need >200)",
                detail={"free_gib": round(free_gb, 1)},
            )
        )
    else:
        results.append(
            CheckResult(
                name="artifact_disk_free",
                group="hardware",
                required=True,
                ok=False,
                message=f"{ARTIFACT_ROOT} does not exist; create it or set CAPA_STORAGE_ROOT",
            )
        )


def check_repo(results: list[CheckResult]) -> None:
    must = [
        "pyproject.toml",
        "init_env.sh",
        "src/capa/agent.py",
        "src/capa/tools/registry.py",
        "demo/demo_server.py",
        "pipelines/data/register_planner_dataset.py",
        "pipelines/eval/run_generation_eval.py",
        "pipelines/eval/compare_generation_runs.py",
        "pipelines/eval/check_runtime_routing_multiseed_gate.py",
        "pipelines/experiments/registry_cli.py",
        "scripts/run_qwen25_7b_trl_sft_lora.sh",
        "scripts/run_qwen25_7b_trl_grpo_lora.sh",
        "scripts/merge_lora_adapter.py",
        "configs/environments/trl-cu124.lock.txt",
        "configs/train/qwen25_grpo_stateful_retrieval_v1.json",
        "experiments/registry.jsonl",
    ]
    missing = [p for p in must if not (ROOT / p).exists()]
    results.append(
        CheckResult(
            name="required_files",
            group="repo",
            required=True,
            ok=not missing,
            message=("all present" if not missing else f"{len(missing)} missing"),
            detail={"missing": missing},
        )
    )
    rc, out, _ = _run(["git", "rev-parse", "HEAD"], timeout=4)
    results.append(
        CheckResult(
            name="git_commit",
            group="repo",
            required=False,
            ok=(rc == 0),
            message=(out[:12] if rc == 0 else "not a git checkout"),
            detail={"sha": out} if rc == 0 else {},
        )
    )
    rc2, out2, _ = _run(["git", "status", "--porcelain"], timeout=4)
    dirty = bool(out2.strip()) if rc2 == 0 else False
    results.append(
        CheckResult(
            name="git_clean",
            group="repo",
            required=False,
            ok=not dirty,
            message=("clean" if not dirty else "dirty working tree; record in run provenance"),
        )
    )


def check_env(results: list[CheckResult]) -> None:
    check_python(results)

    demo_bin = ROOT / ".venv" / "bin" / "python"
    train_bin = ROOT / ".venv-trl-grpo-cu124" / "bin" / "python"
    results.append(
        CheckResult(
            name="venv_demo",
            group="env",
            required=True,
            ok=demo_bin.is_file() and os.access(demo_bin, os.X_OK),
            message=str(demo_bin),
        )
    )
    results.append(
        CheckResult(
            name="venv_train_cu124",
            group="env",
            required=True,
            ok=train_bin.is_file() and os.access(train_bin, os.X_OK),
            message=str(train_bin),
        )
    )

    # If the train venv is present, verify pinned versions.
    if train_bin.is_file():
        rc, out, _ = _run(
            [
                str(train_bin),
                "-c",
                "import importlib.metadata as m,json;print(json.dumps({k:m.version(k) for k in "
                "['torch','transformers','trl','peft','accelerate','datasets']}))",
            ],
            timeout=15,
        )
        detail: dict[str, Any] = {}
        ok = False
        if rc == 0:
            try:
                versions = json.loads(out)
                detail = versions
                expected = {
                    "torch": "2.6.0",
                    "transformers": "4.57.6",
                    "trl": "1.8.0",
                    "peft": "0.19.1",
                    "accelerate": "1.14.0",
                    "datasets": "5.0.0",
                }
                mismatches = {
                    k: (versions.get(k), v)
                    for k, v in expected.items()
                    if not versions.get(k, "").startswith(v)
                }
                ok = not mismatches
                if mismatches:
                    detail["mismatches"] = mismatches
            except json.JSONDecodeError:
                detail = {"raw": out}
        results.append(
            CheckResult(
                name="train_env_pinned_versions",
                group="env",
                required=True,
                ok=ok,
                message=("pinned versions match" if ok else "pinned versions drift"),
                detail=detail,
            )
        )


def check_data(results: list[CheckResult]) -> None:
    important = [
        "training/planner_grpo_seed_v1/cases/planner_grpo_focused_train_v3_cases.jsonl",
        "training/planner_grpo_seed_v1/cases/planner_grpo_focused_val_v3_cases.jsonl",
        "training/planner_grpo_seed_v1/cases/planner_grpo_compound245_eval_cases.jsonl",
        "training/planner_grpo_seed_v1/sft_data_v3_chatml/train.jsonl",
        "training/planner_grpo_seed_v1/sft_data_runtime_probe_curriculum_v1_chatml/train.jsonl",
        "training/planner_grpo_seed_v1/sft_data_runtime_routing_v1_chatml/train.jsonl",
        "data/datasets/planner_focused_v3",
        "data/datasets/planner_runtime_probe_curriculum_v1",
    ]
    missing = [p for p in important if not (ROOT / p).exists()]
    results.append(
        CheckResult(
            name="planner_datasets_present",
            group="data",
            required=True,
            ok=not missing,
            message=("all present" if not missing else f"{len(missing)} missing"),
            detail={"missing": missing},
        )
    )


def check_models(results: list[CheckResult]) -> None:
    for path, label in [
        (DEFAULT_MODEL_DIR, "qwen25_7b_base"),
        (DEFAULT_OUTPUTS_DIR / "merged-qwen25-7b-sft-v3-chatml", "sft_v3_merged"),
    ]:
        exists = path.is_dir() and any(path.iterdir()) if path.exists() else False
        results.append(
            CheckResult(
                name=f"model_{label}",
                group="models",
                required=(label == "qwen25_7b_base"),
                ok=exists,
                message=str(path) + (" ok" if exists else " missing"),
            )
        )


def check_services(results: list[CheckResult]) -> None:
    # SOCKS proxy
    ok, msg = _tcp_ok("127.0.0.1", 8888)
    results.append(
        CheckResult(
            name="socks5_proxy_8888",
            group="services",
            required=False,
            ok=ok,
            message=msg,
        )
    )
    # RAG tunnels
    for port, label in [(6061, "rag_gbrain"), (6062, "rag_playbook")]:
        ok, msg = _tcp_ok("127.0.0.1", port)
        results.append(
            CheckResult(
                name=label,
                group="services",
                required=False,
                ok=ok,
                message=msg,
            )
        )
    # optional demo server
    ok, msg = _http_ok("http://127.0.0.1:18080/health")
    results.append(
        CheckResult(
            name="demo_server_health",
            group="services",
            required=False,
            ok=ok,
            message=msg,
        )
    )


CHECK_FUNCS: dict[str, Callable[[list[CheckResult]], None]] = {
    "hardware": check_hardware,
    "repo": check_repo,
    "env": check_env,
    "data": check_data,
    "models": check_models,
    "services": check_services,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="append",
        choices=CHECK_GROUPS,
        help="Restrict to specific check groups (repeatable). Default: all.",
    )
    parser.add_argument("--out", type=Path, help="Write JSON report to this path.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit code 1 if any required check fails.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    groups = args.check or list(CHECK_GROUPS)
    results: list[CheckResult] = []
    for group in groups:
        CHECK_FUNCS[group](results)

    required_failed = [r for r in results if r.required and not r.ok]
    optional_failed = [r for r in results if not r.required and not r.ok]
    passed = [r for r in results if r.ok]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "groups": groups,
        "summary": {
            "total": len(results),
            "passed": len(passed),
            "required_failed": len(required_failed),
            "optional_failed": len(optional_failed),
        },
        "results": [asdict(r) for r in results],
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    # human summary
    for r in results:
        icon = "ok " if r.ok else ("REQ" if r.required else "opt")
        print(f"[{icon}] {r.group:9s} {r.name:34s} {r.message}")
    s = report["summary"]
    print(
        f"\nsummary: passed={s['passed']} "
        f"required_failed={s['required_failed']} "
        f"optional_failed={s['optional_failed']}"
    )

    if args.strict and required_failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
