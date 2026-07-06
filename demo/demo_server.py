#!/usr/bin/env python3
"""
Demo HTTP server entrypoint.

主要功能：
- 提供 `/run` 流式接口与静态结果文件访问。
- 负责会话文件读写、上传文件处理、运行目录管理与日志落盘。
- 串联 `AgentOrchestrator` 与 `ToolExecutor`，驱动一次完整请求生命周期。

主要模块：
- 会话与文件 I/O：`_load_session_state` / `_save_session_state` / `_session_guard`；会话下按 `thread_*` 分桶保存对话态（账本/摘要/轮次等）
- HTTP 层：`DemoHandler`（GET/POST、NDJSON 流输出）
- 工具执行桥接：`_run_agent_loop` 与各 `_run_*_streaming` 方法
- 启动入口：`main()`
"""
import argparse
import ast
import cgi
import csv
import hashlib
import importlib.util
import json
import mimetypes
import os
import re
import secrets
import socket
import subprocess
import tempfile
import threading
import warnings
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import requests
from requests.exceptions import RequestsDependencyWarning

import agent
import clarification as clarification_state
import gbrain_rag_client as gbrain_rag
import migration_advisor
import memory_system as ms
import prompts
from tools import schemas as tool_schemas
from tools.executor import ToolExecutor
from frontend_page import HTML_PAGE
from util.rex_label_extraction import extract_rex_detection_labels
from util.vlm_service import VLMService

warnings.filterwarnings("ignore", category=RequestsDependencyWarning)

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "demo" / "runs"
RUN_LOG_FILE = RUNS_DIR / "run_log.txt"
SESSIONS_DIR = ROOT / "demo" / "sessions"
SESSION_SUMMARY_LIMIT = int(os.environ.get("DEMO_SESSION_SUMMARY_LIMIT", "12"))
DEFAULT_API_BASE = os.environ.get("DEMO_API_BASE", "https://api.apiyi.com/v1")
# 文本 / VLM skill（方案报告、意图、扩写、评测等），与 Flux 使用的 apiyi 凭证分离
DEMO_LLM_API_BASE = os.environ.get("DEMO_LLM_API_BASE", "http://10.111.32.253:8000/v1")
DEMO_LLM_API_KEY = os.environ.get("DEMO_LLM_API_KEY", "token.sdc@2026")
# 会话 thread 前置分流（OpenAI 兼容 /chat/completions，默认 Qwen 4B 量级小模型）
THREAD_ROUTER_API_BASE = os.environ.get("DEMO_THREAD_ROUTER_API_BASE", "").strip() or DEMO_LLM_API_BASE
THREAD_ROUTER_MODEL = os.environ.get("DEMO_THREAD_ROUTER_MODEL", "Qwen3.5-4B").strip()
THREAD_ROUTER_TIMEOUT_SEC = int(os.environ.get("DEMO_THREAD_ROUTER_TIMEOUT_SEC", "15"))
THREAD_TOPIC_MODEL = os.environ.get("DEMO_THREAD_TOPIC_MODEL", "Qwen3.5-4B").strip()
THREAD_TOPIC_TIMEOUT_SEC = int(os.environ.get("DEMO_THREAD_TOPIC_TIMEOUT_SEC", "20"))
THREAD_TOPIC_MAX_TURNS = int(os.environ.get("DEMO_THREAD_TOPIC_MAX_TURNS", "10"))
QUERY_TRAJ_SUMMARY_MODEL = os.environ.get("DEMO_QUERY_TRAJ_SUMMARY_MODEL", "Qwen3.5-4B").strip()
QUERY_TRAJ_SUMMARY_TIMEOUT_SEC = int(os.environ.get("DEMO_QUERY_TRAJ_SUMMARY_TIMEOUT_SEC", "25"))
API_KEY_FILE = ROOT / "api_key.txt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_UPLOAD_IMAGES = 10
NUM_GENERATED_IMAGES = 3
QWEN_BASE_URL = os.environ.get("DEMO_QWEN_BASE_URL", "http://10.111.32.253:8000/v1")
QWEN_DETECTION_URL = os.environ.get("DEMO_QWEN_DETECTION_URL", "http://127.0.0.1:9012/v1")
REX_BASE_URL = os.environ.get("DEMO_REX_BASE_URL", "http://10.111.32.253:8000/v1")
ADELA_CLI_TIMEOUT_SEC = int(os.environ.get("DEMO_ADELA_CLI_TIMEOUT_SEC", "900"))

# ---------------------------------------------------------------------------
# RAG 配置：可直接改下面常量；未改时仍可用环境变量覆盖（export DEMO_RAG_API_MODE=...）
# 优先级：configure_rag() 显式参数 > 环境变量 > 默认值
# ---------------------------------------------------------------------------
DEMO_RAG_API_MODE = os.environ.get("DEMO_RAG_API_MODE", "unified")  # playbook | unified
DEMO_RAG_PLAYBOOK_QUERY_URL = os.environ.get(
    "RAG_BASE_URL",
    os.environ.get("RAG_QUERY_URL", "http://127.0.0.1:6062/api/v1/playbook/query"),
)
DEMO_RAG_GBRAIN_BASE_URL = os.environ.get(
    "GBRAIN_RAG_BASE_URL",
    os.environ.get("RAG_UNIFIED_BASE_URL", "http://127.0.0.1:6061/api/v1/rag"),
)
DEMO_RAG_INCLUDE_FULL_DOCUMENTS = os.environ.get("DEMO_RAG_INCLUDE_FULL_DOCUMENTS", "1") not in {
    "0",
    "false",
    "off",
    "no",
}
DEMO_RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "12"))
DEMO_RAG_PLAYBOOK_TOP_K = int(os.environ.get("RAG_PLAYBOOK_TOP_K", "8"))
DEMO_RAG_RETRIEVAL_METHOD = os.environ.get("RAG_RETRIEVAL_METHOD", "hybrid")


def _apply_demo_rag_config() -> None:
    """把 demo_server 顶部常量写入 gbrain_rag_client（启动时调用一次即可）。"""
    gbrain_rag.configure_rag(
        api_mode=DEMO_RAG_API_MODE,
        playbook_query_url=DEMO_RAG_PLAYBOOK_QUERY_URL,
        gbrain_base_url=DEMO_RAG_GBRAIN_BASE_URL,
        include_full_documents=DEMO_RAG_INCLUDE_FULL_DOCUMENTS,
        top_k=DEMO_RAG_TOP_K,
        playbook_top_k=DEMO_RAG_PLAYBOOK_TOP_K,
        retrieval_method=DEMO_RAG_RETRIEVAL_METHOD,
    )


_apply_demo_rag_config()
RAG_BASE_URL = gbrain_rag.playbook_query_url()

_pipeline_mod = None
SUBPROCESS_TIMEOUT_SEC = int(os.environ.get("DEMO_SUBPROCESS_TIMEOUT_SEC", "300"))

_session_locks_guard = threading.Lock()
_session_locks: dict[str, threading.Lock] = {}
BEIJING_TZ = timezone(timedelta(hours=8))

def _pipeline_helpers():
    global _pipeline_mod
    if _pipeline_mod is None:
        path = ROOT / "skills/target-detection-evaluation/scripts/run_pipeline.py"
        spec = importlib.util.spec_from_file_location("te_pipeline", path)
        if spec is None or spec.loader is None:
            raise RuntimeError("failed to load run_pipeline.py")
        _pipeline_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_pipeline_mod)
    return _pipeline_mod


def _load_api_key() -> str:
    if not API_KEY_FILE.is_file():
        raise FileNotFoundError(
            f"未找到 API Key 文件: {API_KEY_FILE}（请在仓库根目录放置 api_key.txt）"
        )
    key = API_KEY_FILE.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError("api_key.txt 为空")
    return key


def _safe_upload_suffix(filename: str | None) -> str:
    if not filename:
        return ".jpg"
    name = Path(filename).name
    suf = Path(name).suffix.lower()
    if suf in IMAGE_EXTENSIONS:
        return suf
    return ".jpg"


def _guess_lan_ip() -> str | None:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.2)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return None

def _resolve_input_path(p: str) -> str:
    if not p:
        return ""
    candidate = Path(p)
    if not candidate.is_absolute():
        candidate = (ROOT / candidate).resolve()
    return str(candidate)


def _mask_cmd(cmd: list[str]) -> str:
    out = []
    i = 0
    while i < len(cmd):
        tok = cmd[i]
        if tok == "--api-key":
            out.append("--api-key")
            if i + 1 < len(cmd):
                out.append("***")
                i += 2
                continue
        elif tok.startswith("--api-key="):
            out.append("--api-key=***")
            i += 1
            continue
        out.append(tok)
        i += 1
    return " ".join(out)


def _run_subprocess(cmd: list[str], timeout_sec: int | None = None) -> tuple[int, str]:
    effective_timeout = int(timeout_sec or SUBPROCESS_TIMEOUT_SEC)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            env=os.environ.copy(),
            timeout=effective_timeout,
        )
        merged = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        return proc.returncode, merged.strip()
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        merged = (stdout or "") + ("\n" + stderr if stderr else "")
        merged = (
            merged.strip() + f"\n\n[timeout] 子进程执行超过 {effective_timeout} 秒"
        ).strip()
        return 124, merged


def _form_get_first(form: cgi.FieldStorage, key: str, default: str = "") -> str:
    if key not in form:
        return default
    v = form.getvalue(key)
    if v is None:
        return default
    if isinstance(v, list):
        return str(v[0]).strip() if v else default
    return str(v).strip() or default


def _append_reference_image_path(session: dict, path: str) -> None:
    p = str(path or "").strip()
    if not p or not Path(p).is_file():
        return
    refs = session.get("reference_image_paths")
    if not isinstance(refs, list):
        refs = []
    if p not in refs:
        refs.append(p)
    session["reference_image_paths"] = refs[-20:]


def _collect_reference_image_paths(
    session: dict,
    run_dir: Path,
    *,
    primary_path: str = "",
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def _add(path: str) -> None:
        p = str(path or "").strip()
        if not p or not Path(p).is_file():
            return
        key = str(Path(p).resolve())
        if key in seen:
            return
        seen.add(key)
        out.append(key)

    _add(primary_path)
    refs = session.get("reference_image_paths")
    if isinstance(refs, list):
        for item in refs:
            _add(str(item or ""))
    ref_dir = run_dir / "reference_images"
    if ref_dir.is_dir():
        for p in sorted(ref_dir.iterdir(), key=lambda x: x.name):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                _add(str(p.resolve()))
    for p in sorted(run_dir.glob("uploaded_image*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            _add(str(p.resolve()))
    return out


def _save_one_upload_field(item, run_dir: Path, *, slot: int = 0) -> str:
    filename = getattr(item, "filename", None) or ""
    if not str(filename).strip():
        return ""
    suffix = _safe_upload_suffix(filename)
    dest = run_dir / f"uploaded_image_{slot:02d}{suffix}"
    total = 0
    max_bytes = 20 * 1024 * 1024  # 20MB per file
    with open(dest, "wb") as f:
        while True:
            chunk = item.file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                try:
                    dest.unlink(missing_ok=True)
                except Exception:
                    pass
                raise ValueError("上传图片过大，单张限制 20MB")
            f.write(chunk)
    if total == 0:
        try:
            dest.unlink(missing_ok=True)
        except Exception:
            pass
        return ""
    ref_dir = run_dir / "reference_images"
    ref_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%H%M%S_%f")
    ref_copy = ref_dir / f"{stamp}_{slot:02d}_{Path(filename).name or 'upload.jpg'}"
    try:
        ref_copy.write_bytes(dest.read_bytes())
    except OSError:
        pass
    return str(dest.resolve())


def _upload_image_fields(form: cgi.FieldStorage) -> list:
    if "image" not in form:
        return []
    item = form["image"]
    return item if isinstance(item, list) else [item]


def _save_uploaded_images(
    form: cgi.FieldStorage,
    run_dir: Path,
    *,
    max_files: int = MAX_UPLOAD_IMAGES,
) -> list[str]:
    saved: list[str] = []
    for idx, one in enumerate(_upload_image_fields(form)):
        if len(saved) >= max_files:
            break
        if not getattr(one, "filename", None):
            continue
        path = _save_one_upload_field(one, run_dir, slot=idx)
        if path:
            saved.append(path)
    return saved


def _save_uploaded_image(form: cgi.FieldStorage, run_dir: Path) -> str:
    paths = _save_uploaded_images(form, run_dir)
    return paths[0] if paths else ""


def _intent_summary(intent: dict) -> dict:
    return {
        "task_name": intent.get("task_name", ""),
        "scene": intent.get("scene", ""),
        "target": intent.get("target", ""),
        "camera": intent.get("camera", ""),
    }


def _append_run_log(run_stamp: str, message: str) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(RUN_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] [run {run_stamp}]\n")
        f.write((message or "").rstrip() + "\n\n")


def _terminal_log(run_stamp: str, message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [run {run_stamp}] {message}", flush=True)


def _evaluation_for_demo(ev: dict) -> dict:
    out = dict(ev)
    out.pop("per_image_evaluation", None)
    return out


def _trim_text(value: str, limit: int | None = 400) -> str:
    text = str(value or "").strip()
    if limit is None or limit <= 0:
        return text
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def _normalize_adela_candidate_names(values, *, limit: int = 5) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _extract_adela_model_from_evidence(item: dict) -> dict | None:
    if not isinstance(item, dict):
        return None
    payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
    entity = payload.get("entity") if isinstance(payload.get("entity"), dict) else {}
    entity_model_name = str(
        entity.get("model_name")
        or entity.get("name")
        or payload.get("model_name")
        or payload.get("name")
        or ""
    ).strip()
    model_id = 0
    model_info = str(entity.get("model_info") or payload.get("model_info") or "")
    if model_info:
        try:
            parts = model_info.split("\n", 1)
            deploy_info = ast.literal_eval(parts[1] if len(parts) > 1 else model_info)
            model_id = int(deploy_info.get("model_id") or 0)
        except Exception:
            model_id = 0
    if model_id <= 0:
        for key in ("model_id", "rawmodel_id", "rid"):
            for src in (entity, payload):
                try:
                    parsed = int(src.get(key))
                except (TypeError, ValueError):
                    parsed = 0
                if parsed > 0:
                    model_id = parsed
                    break
            if model_id > 0:
                break
    if model_id <= 0:
        return None
    return {"rawmodel_id": model_id, "model_name": entity_model_name}


def _collect_adela_models_from_rag_packet(raw_last_packet: dict) -> list[dict]:
    evidence_groups: list[list] = []
    fused = raw_last_packet.get("fused_evidences")
    if isinstance(fused, list) and fused:
        evidence_groups.append(fused)
    direct = raw_last_packet.get("evidences")
    if isinstance(direct, list) and direct:
        evidence_groups.append(direct)
    models_ordered: list[dict] = []
    seen_ids: set[int] = set()
    for evidence_list in evidence_groups:
        for item in evidence_list:
            extracted = _extract_adela_model_from_evidence(item)
            if not extracted:
                continue
            rid = int(extracted["rawmodel_id"])
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
            models_ordered.append(extracted)
    return models_ordered


def _adela_error_text_blob(*values) -> str:
    parts: list[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, dict):
            try:
                parts.append(json.dumps(value, ensure_ascii=False))
            except (TypeError, ValueError):
                parts.append(str(value))
        else:
            parts.append(str(value))
    return " ".join(parts)


def _parse_json_object_text(raw: str) -> dict | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _adela_unsupported_platform_user_message(*values) -> str | None:
    blob = _adela_error_text_blob(*values).lower()
    if "unsupported platform" not in blob and "unsupported_platform" not in blob:
        return None
    return (
        "模型部署失败：当前填写的部署平台不被 Adela 支持。"
        "请提供规范的部署平台，例如 cuda11.0-trt7.1-fp16-T4、"
        "cuda11.0-trt7.1-int8-T4、cuda11.0-trt7.1-fp32-T4。"
        "格式通常为 cuda<版本>-trt<版本>-<精度>-<GPU型号>，"
        "也可使用 acl-、cpu-、rknn- 前缀的平台标识。"
    )


def _infer_adela_patch_from_text(
    *,
    text: str,
    known_tool_args: dict,
) -> dict:
    src = known_tool_args if isinstance(known_tool_args, dict) else {}
    user_text = str(text or "").strip()
    if not user_text:
        return {}
    patch: dict = {}
    explicit_id = clarification_state._extract_explicit_adela_rawmodel_id(user_text)
    if explicit_id is not None:
        patch["rawmodel_id"] = explicit_id
        patch["model_name"] = ""
    eval_type = clarification_state.adela_eval_type_from_text(user_text)
    if eval_type in (0, 1):
        patch["eval_type"] = eval_type
    platform_match = re.search(
        r"(cuda\d+\.\d+-trt[A-Za-z0-9.\-]+|acl-[A-Za-z0-9.\-]+|cpu-[A-Za-z0-9.\-]+|rknn-[A-Za-z0-9.\-]+)",
        user_text,
        flags=re.I,
    )
    if platform_match:
        patch["platform"] = platform_match.group(1)
    if not patch and (user_text.startswith("换成") or user_text.startswith("改成") or "平台" in user_text or "性能" in user_text or "精度" in user_text):
        if not explicit_id and eval_type is None and "平台" not in patch:
            patch["model_name"] = user_text
    return patch


def _adela_model_name_guess_from_query(user_text: str, platform: str) -> str:
    """从整句问话里抠出可能的 Adela 模型名（供 RAG 解析），去掉平台串与常见问法词。"""
    t = str(user_text or "").strip()
    if not t:
        return ""
    if platform:
        t = re.sub(re.escape(platform), " ", t, flags=re.I)
    t = re.sub(
        r"(部署到|部署在|部署至|在…平台|在平台|目标平台|平台|请问|多少|是什么|哪个|如何|怎样|帮我查|查一下|评测|指标|结果|的|吗|呢|嘛|啊|精度|准确率|准度|性能|速度|mAP|AP\b|accuracy|precision|latency|fps|FPS|吞吐|延迟)",
        " ",
        t,
        flags=re.I,
    )
    t = re.sub(r"[,，.。!！?？;；:：\s]+", " ", t).strip()
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_\-]{5,}", t)
    skip_prefixes = ("cuda", "trt", "acl", "cpu", "rknn", "rawmodel", "modelid")
    cleaned: list[str] = []
    for tok in tokens:
        low = tok.lower()
        if any(low.startswith(p) for p in skip_prefixes):
            continue
        cleaned.append(tok)
    if not cleaned:
        return ""
    with_us = [x for x in cleaned if "_" in x]
    pool = with_us if with_us else cleaned
    return max(pool, key=len)


def _heuristic_forced_adela_cli_step(user_text: str) -> dict | None:
    """
    当问句同时包含 Adela 类平台串 + 精度/性能类意图 + 可猜模型名或显式 rawmodel_id 时，
    强制第一步走 adela_cli_eval（模型名在 executor 内仍经 RAG 解析为 ID），避免 Planner 误走 rag_answer。
    """
    raw = str(user_text or "").strip()
    if len(raw) < 10:
        return None
    platform_m = re.search(
        r"(cuda\d+\.\d+-trt[A-Za-z0-9.\-]+|acl-[A-Za-z0-9.\-]+|cpu-[A-Za-z0-9.\-]+|rknn-[A-Za-z0-9.\-]+)",
        raw,
        flags=re.I,
    )
    if not platform_m:
        return None
    platform = platform_m.group(1)
    low = raw.lower()
    perf_hit = any(
        k in low
        for k in ("性能", "速度", "fps", "吞吐", "延迟", "latency", "推理速度", "inference speed")
    )
    acc_hit = any(
        k in raw for k in ("精度", "准确率", "准度", "准确度", "accuracy", "precision")
    ) or re.search(r"\bmap\b", low) or re.search(r"\bap\b", low)
    bench_hit = "benchmark" in low or "评测" in raw
    if perf_hit and acc_hit:
        eval_type = 0
    elif perf_hit:
        eval_type = 1
    elif acc_hit or bench_hit:
        eval_type = 0
    elif "部署" in raw and re.search(r"(多少|怎样|如何|怎么样|是什么)", raw):
        eval_type = 0
    else:
        return None

    explicit_id = clarification_state._extract_explicit_adela_rawmodel_id(raw)
    action_input: dict = {
        "platform": platform,
        "eval_type": int(eval_type),
        "finish_after_tool": True,
    }
    if explicit_id is not None:
        action_input["rawmodel_id"] = int(explicit_id)
        action_input["model_name"] = ""
    else:
        model_guess = _adela_model_name_guess_from_query(raw, platform)
        if not model_guess:
            return None
        action_input["rawmodel_id"] = ""
        action_input["model_name"] = model_guess

    return {
        "thought": "问句含 Adela 部署平台标识及精度/性能类诉求，强制 adela_cli_eval；模型名由工具内 RAG 解析为 rawmodel_id 后调用 Adela CLI。",
        "decision_type": agent.DECISION_TYPE_TOOL,
        "action": agent.TOOL_ADELA_CLI_EVAL,
        "action_input": action_input,
        "final_answer": "",
    }


def _sanitize_adela_tool_args_for_user_query(
    *,
    user_text: str,
    tool_args: dict,
) -> dict:
    args = dict(tool_args) if isinstance(tool_args, dict) else {}
    inferred = clarification_state.adela_eval_type_from_text(user_text)
    existing = clarification_state.normalize_adela_eval_type_arg(args.get("eval_type"))
    if inferred in (0, 1):
        args["eval_type"] = inferred
    elif existing in (0, 1):
        args["eval_type"] = existing
    else:
        args["eval_type"] = ""
    explicit_id = clarification_state._extract_explicit_adela_rawmodel_id(
        args.get("rawmodel_id"),
        user_text,
    )
    model_name = str(args.get("model_name") or "").strip()
    if explicit_id is None and model_name and not model_name.isdigit() and clarification_state._extract_explicit_adela_rawmodel_id(user_text) is None:
        args["rawmodel_id"] = ""
    return args


def _latest_adela_tool_args_for_thread(session: dict, *, thread_id: str) -> dict:
    tid = str(thread_id or "").strip()
    qt = session.get("query_trajectories")
    if not isinstance(qt, list):
        return {}
    ledger = session.get("raw_ledger") if isinstance(session.get("raw_ledger"), list) else []
    by_id = {str(ev.get("event_id") or ""): ev for ev in ledger if isinstance(ev, dict)}
    for tr in reversed(qt):
        if not isinstance(tr, dict):
            continue
        if tid and str(tr.get("thread_id") or "").strip() != tid:
            continue
        steps = tr.get("steps") if isinstance(tr.get("steps"), list) else []
        for st in reversed(steps):
            if not isinstance(st, dict):
                continue
            if agent.normalize_agent_action(str(st.get("action") or "").strip()) != agent.TOOL_ADELA_CLI_EVAL:
                continue
            ptr = str(st.get("observation_event_id") or "").strip()
            ev = by_id.get(ptr)
            if not isinstance(ev, dict):
                continue
            payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
            args = payload.get("_action_input") if isinstance(payload.get("_action_input"), dict) else {}
            return dict(args)
    return {}


def _build_observation(action: str, **kwargs) -> dict:
    data = dict(kwargs)
    data["action"] = action
    return data


def _playbook_feedback_url() -> str:
    return gbrain_rag.playbook_feedback_url()


def _playbook_retrieve_url() -> str:
    return gbrain_rag.playbook_retrieve_url()


def _normalize_rag_refs(maybe_refs) -> list[dict]:
    out: list[dict] = []
    if not isinstance(maybe_refs, list):
        return out
    seen: set[str] = set()
    for item in maybe_refs:
        if not isinstance(item, dict):
            continue
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        entity = payload.get("entity") if isinstance(payload.get("entity"), dict) else {}
        url = str(
            item.get("url")
            or payload.get("url")
            or payload.get("reference")
            or payload.get("link")
            or entity.get("ones_release_link")
            or item.get("source_path")
            or ""
        ).strip()
        if not url or url in seen:
            continue
        seen.add(url)
        doc_name = str(
            item.get("doc_name")
            or item.get("title")
            or payload.get("doc_name")
            or payload.get("title")
            or item.get("doc_name")
            or ""
        ).strip()
        out.append({"doc_name": doc_name, "url": url})
    return out


def _rag_evidence_ids(maybe_evidences) -> list[str]:
    if not isinstance(maybe_evidences, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in maybe_evidences:
        if not isinstance(item, dict):
            continue
        eid = str(item.get("evidence_id") or item.get("legacy_evidence_id") or "").strip()
        if eid and eid not in seen:
            seen.add(eid)
            out.append(eid)
    return out


def _new_session_id() -> str:
    return f"sess_{secrets.token_hex(8)}"


def _normalize_session_id(raw: str) -> str:
    sid = str(raw or "").strip()
    if not sid:
        return _new_session_id()
    sid = re.sub(r"[^a-zA-Z0-9_-]", "_", sid)
    if len(sid) < 6:
        sid = f"{sid}_{secrets.token_hex(4)}"
    if len(sid) > 64:
        sid = sid[:64]
    return sid


THREAD_PAYLOAD_KEYS = (
    "raw_ledger",
    "query_trajectories",
    "thread_aux_state",
    "summary_history",
    "chat_turns",
    "last_image_path",
)


def _new_thread_id() -> str:
    return f"thread_{secrets.token_hex(8)}"


def _normalize_thread_id(raw: str) -> str:
    tid = str(raw or "").strip()
    if not tid:
        return ""
    tid = re.sub(r"[^a-zA-Z0-9_-]", "_", tid)
    if len(tid) < 8:
        tid = f"thread_{tid}_{secrets.token_hex(3)}"
    if len(tid) > 80:
        tid = tid[:80]
    return tid


def _empty_thread_payload(session_id: str) -> dict:
    return {
        "raw_ledger": [],
        "query_trajectories": [],
        "thread_aux_state": {"session_id": session_id, "ledger_cursor": 0},
        "summary_history": [],
        "chat_turns": [],
        "last_image_path": "",
        "topic": "",
    }


def _strip_ledger_event_for_storage(event: dict) -> dict:
    if not isinstance(event, dict):
        return {}
    return {
        "event_id": str(event.get("event_id") or "").strip(),
        "seq": int(event.get("seq") or 0),
        "event_type": str(event.get("event_type") or "").strip(),
        "observation": str(event.get("observation") or ""),
        "payload": event.get("payload") if isinstance(event.get("payload"), dict) else {},
        "external_ref": str(event.get("external_ref") or "").strip(),
    }


def _inflate_query_trajectories_from_storage(values) -> list[dict]:
    if isinstance(values, list):
        return list(values)
    if not isinstance(values, dict):
        return []
    out: list[dict] = []
    for query_id, raw in values.items():
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        if not str(item.get("query_id") or "").strip():
            item["query_id"] = str(query_id or "").strip()
        out.append(item)
    return out


def _serialize_query_trajectories_for_storage(values) -> dict:
    out: dict[str, dict] = {}
    if not isinstance(values, list):
        return out
    for raw in values:
        if not isinstance(raw, dict):
            continue
        query_id = str(raw.get("query_id") or "").strip()
        if not query_id:
            continue
        item = {
            "query": str(raw.get("query") or raw.get("user_text") or ""),
            "result_summary": str(raw.get("result_summary") or ""),
            "steps": list(raw.get("steps")) if isinstance(raw.get("steps"), list) else [],
        }
        out[query_id] = item
    return out


def _pull_thread_payload_from_session(session: dict) -> dict:
    sid = str(session.get("session_id") or "")
    rl = session.get("raw_ledger")
    if not isinstance(rl, list):
        rl = []
    qt = session.get("query_trajectories")
    if not isinstance(qt, list):
        qt = []
    wt_legacy = session.get("working_trajectory")
    if not isinstance(wt_legacy, list):
        wt_legacy = []
    aux = session.get("thread_aux_state")
    if not isinstance(aux, dict):
        aux = {}
    legacy_ss = session.get("session_state")
    if isinstance(legacy_ss, dict) and legacy_ss and not aux:
        aux = dict(legacy_ss)
    sh = session.get("summary_history")
    if not isinstance(sh, list):
        sh = []
    ct = session.get("chat_turns")
    if not isinstance(ct, list):
        ct = []
    return {
        "raw_ledger": list(rl),
        "query_trajectories": list(qt),
        "thread_aux_state": dict(aux),
        "summary_history": list(sh),
        "chat_turns": list(ct),
        "last_image_path": str(session.get("last_image_path") or "").strip(),
        "topic": "",
        # 仅用于从旧根字段迁入 thread 桶后由 migrate_schema 消费
        "working_trajectory": list(wt_legacy),
    }


def _migrate_session_threads(session: dict, *, from_disk: bool) -> None:
    """确保存在 threads / active_thread_id；磁盘上旧格式（字段在根上）迁入 thread_default。"""
    sid = str(session.get("session_id") or "")
    threads = session.get("threads")
    if not isinstance(threads, dict):
        threads = {}
        session["threads"] = threads

    if not threads:
        if from_disk:
            tid = "thread_default"
            threads[tid] = _pull_thread_payload_from_session(session)
        else:
            tid = _new_thread_id()
            threads[tid] = _empty_thread_payload(sid)
        session["active_thread_id"] = tid

    active = str(session.get("active_thread_id") or "").strip()
    if active not in threads:
        session["active_thread_id"] = sorted(threads.keys())[0]


def _hydrate_thread_into_session(session: dict) -> None:
    """把当前 active thread 的负载拷到 session 根上，供 LedgerStore / Agent 沿用原有字段。"""
    threads = session.get("threads")
    if not isinstance(threads, dict):
        return
    tid = str(session.get("active_thread_id") or "").strip()
    bucket = threads.get(tid)
    if not isinstance(bucket, dict):
        return
    sid = str(session.get("session_id") or "")
    for key in THREAD_PAYLOAD_KEYS:
        if key == "thread_aux_state":
            aux = bucket.get("thread_aux_state")
            if not isinstance(aux, dict):
                aux = bucket.get("session_state")
            session["thread_aux_state"] = dict(aux) if isinstance(aux, dict) else {"session_id": sid, "ledger_cursor": 0}
            session["thread_aux_state"]["session_id"] = sid
            continue
        val = bucket.get(key)
        if key == "raw_ledger":
            session[key] = list(val) if isinstance(val, list) else []
        elif key == "query_trajectories":
            session[key] = _inflate_query_trajectories_from_storage(val)
        elif key in ("summary_history", "chat_turns"):
            session[key] = list(val) if isinstance(val, list) else []
        elif key == "last_image_path":
            session[key] = str(val or "").strip()
        else:
            session[key] = val
    session["topic"] = str(bucket.get("topic") or "").strip()
    wt = bucket.get("working_trajectory")
    if isinstance(wt, list) and wt and not (session.get("query_trajectories") or []):
        session["working_trajectory"] = list(wt)


def _persist_thread_from_session(session: dict) -> None:
    """将根上的线程负载写回 threads[active_thread_id]（保存前调用）。"""
    threads = session.setdefault("threads", {})
    if not isinstance(threads, dict):
        threads = {}
        session["threads"] = threads
    tid = str(session.get("active_thread_id") or "").strip()
    if not tid:
        tid = _new_thread_id()
        session["active_thread_id"] = tid
    bucket = threads.setdefault(tid, _empty_thread_payload(str(session.get("session_id") or "")))
    for key in THREAD_PAYLOAD_KEYS:
        if key == "thread_aux_state":
            aux = session.get("thread_aux_state")
            bucket["thread_aux_state"] = dict(aux) if isinstance(aux, dict) else {}
            continue
        if key == "raw_ledger":
            v = session.get(key)
            if isinstance(v, list):
                bucket[key] = [_strip_ledger_event_for_storage(item) for item in v if isinstance(item, dict)]
            else:
                bucket[key] = []
        elif key == "query_trajectories":
            bucket[key] = _serialize_query_trajectories_for_storage(session.get(key))
        elif key in ("summary_history", "chat_turns"):
            v = session.get(key)
            bucket[key] = list(v) if isinstance(v, list) else []
        elif key == "last_image_path":
            bucket[key] = str(session.get("last_image_path") or "").strip()
        else:
            bucket[key] = session.get(key)
    bucket["topic"] = str(session.get("topic") or bucket.get("topic") or "").strip()
    bucket.pop("working_trajectory", None)
    bucket.pop("session_state", None)


def _active_thread_bucket_from_file(data: dict) -> dict:
    """未走完整 hydrate 的磁盘 JSON：取当前 active 对应的数据桶（兼容旧根字段）。"""
    threads = data.get("threads")
    if isinstance(threads, dict) and threads:
        tid = str(data.get("active_thread_id") or "").strip()
        if tid in threads:
            b = threads.get(tid)
            if isinstance(b, dict):
                return b
        first = sorted(threads.keys())[0]
        b = threads.get(first)
        if isinstance(b, dict):
            return b
    return data if isinstance(data, dict) else {}


def _summary_history_for_session_file(data: dict) -> list:
    """列表接口：从已落盘 JSON 取当前用于预览的 summary_history。"""
    bucket = _active_thread_bucket_from_file(data)
    h = bucket.get("summary_history")
    if isinstance(h, list):
        return h
    h = data.get("summary_history")
    return h if isinstance(h, list) else []


def _sort_records_by_updated_at(values) -> list[dict]:
    if not isinstance(values, list):
        return []
    items = [item for item in values if isinstance(item, dict)]
    items.sort(
        key=lambda item: (
            str(item.get("updated_at") or ""),
            str(item.get("run_stamp") or ""),
        )
    )
    return items


def _aggregate_thread_payload_list(threads_map: dict, key: str) -> list[dict]:
    if not isinstance(threads_map, dict):
        return []
    merged: list[dict] = []
    for tid in sorted(threads_map.keys()):
        bucket = threads_map.get(tid)
        if not isinstance(bucket, dict):
            continue
        values = bucket.get(key)
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, dict):
                continue
            row = dict(item)
            row.setdefault("thread_id", tid)
            merged.append(row)
    return _sort_records_by_updated_at(merged)


def _thread_router_enabled() -> bool:
    return str(os.environ.get("DEMO_THREAD_ROUTER_ENABLED", "1")).strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _thread_router_candidates(session: dict) -> list[dict]:
    """供分流器阅读的各 thread 摘要（规范输入：active_threads）。"""
    threads = session.get("threads")
    if not isinstance(threads, dict):
        return []
    out: list[dict] = []
    for tid in sorted(threads.keys()):
        b = threads.get(tid)
        if not isinstance(b, dict):
            continue
        sh = b.get("summary_history")
        if not isinstance(sh, list):
            sh = []
        last_q, last_a = "", ""
        if sh:
            last = sh[-1] if isinstance(sh[-1], dict) else {}
            last_q = str(last.get("query") or "")
            last_a = str(last.get("final_answer") or "")
        topic = str(b.get("topic") or "").strip()
        if not topic:
            topic = _trim_text(last_q or last_a or "未命名话题", 120)
        out.append(
            {
                "thread_id": tid,
                "topic": _trim_text(topic, 120),
            }
        )
    return out


def _parse_json_object_from_llm(text: str) -> dict | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9]*\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw).strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(raw[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _chat_response_format_json_schema(name: str, schema: dict) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": schema,
        },
    }


def _extract_chat_message_content(data: dict) -> str:
    msg = ""
    if not isinstance(data, dict):
        return msg
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        c0 = choices[0]
        if isinstance(c0, dict):
            msg_obj = c0.get("message")
            if isinstance(msg_obj, dict):
                msg = str(msg_obj.get("content") or "")
    return msg


def _repair_json_once(
    *,
    raw_text: str,
    schema_name: str,
    schema: dict,
    model: str,
    url: str,
    timeout_sec: int,
    max_tokens: int = 180,
) -> tuple[dict | None, dict]:
    """
    解析失败时做一次 JSON 修复重试：
    - 强制 response_format 为同一份 json_schema
    - 输入仅包含“待修复文本”
    """
    repair_body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是 JSON 修复器。请将输入文本修复为严格符合 schema 的 JSON。"
                    "只输出 JSON，不要解释。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps({"raw_text": str(raw_text or "")}, ensure_ascii=False),
            },
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": _chat_response_format_json_schema(schema_name, schema),
    }
    info: dict = {"repair_request": repair_body}
    try:
        resp = requests.post(
            url,
            json=repair_body,
            headers={
                "Authorization": f"Bearer {DEMO_LLM_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=timeout_sec,
        )
        info["repair_status_code"] = resp.status_code
        info["repair_response_text"] = (resp.text or "")[:8000]
        if resp.status_code != 200:
            return None, info
        data = resp.json()
        msg = _extract_chat_message_content(data)
        info["repair_message_content"] = msg
        parsed = _parse_json_object_from_llm(msg)
        info["repair_parsed"] = parsed
        return (parsed if isinstance(parsed, dict) else None), info
    except requests.RequestException as exc:
        info["repair_error"] = str(exc)
        return None, info


def _call_thread_router_llm(*, user_text: str, session: dict, run_dir: Path) -> dict:
    """
    调用小模型决定 thread 路由。
    规范输出：{"reason": "...", "action": "CONTINUE|NEW", "target_thread_id": "..."}
    """
    sid = str(session.get("session_id") or "")
    candidates = _thread_router_candidates(session)
    payload_user = {
        "active_threads": candidates,
        "current_query": _trim_text(user_text, 800),
    }
    system = (
        "你是一个智能对话路由专家。你的任务是判断用户的【当前问题】是属于【现有任务线程】的后续追问，"
        "还是一个完全无关的【全新话题】。\n\n"
        "判断标准：\n\n"
        "如果当前问题与某个现有线程的主题（Topic）在语义、任务目的或指代关系上具有连贯性，请选择 CONTINUE 并输出对应的 thread_id。\n\n"
        "如果当前问题与所有现有线程的主题毫无关联（例如从“代码编写”跳跃到“日常闲聊”，或从“工业检测”跳跃到“水果常识”），请务必选择 NEW 开启新线程。\n\n"
        "只输出 JSON，不要 markdown，不要额外文本。格式固定为：\n"
        '{"reason":"...","action":"CONTINUE|NEW","target_thread_id":"thread_xxx"}'
    )
    url = str(THREAD_ROUTER_API_BASE or "").rstrip("/") + "/chat/completions"
    router_schema = {
        "type": "object",
        "properties": {
            "reason": {"type": "string"},
            "action": {"type": "string", "enum": ["CONTINUE", "NEW"]},
            "target_thread_id": {"type": "string"},
        },
        "required": ["reason", "action", "target_thread_id"],
        "additionalProperties": False,
    }
    body = {
        "model": THREAD_ROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload_user, ensure_ascii=False)},
        ],
        "temperature": 0,
        "max_tokens": 180,
        "response_format": _chat_response_format_json_schema("thread_router", router_schema),
    }
    record: dict = {"session_id": sid, "request": body, "url": url}

    def _finalize(result: dict) -> dict:
        return result
    try:
        resp = requests.post(
            url,
            json=body,
            headers={
                "Authorization": f"Bearer {DEMO_LLM_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=THREAD_ROUTER_TIMEOUT_SEC,
        )
        record["status_code"] = resp.status_code
        record["response_text"] = (resp.text or "")[:8000]
        if resp.status_code != 200:
            record["ok"] = False
            record["error"] = f"HTTP {resp.status_code}"
            try:
                (run_dir / "thread_router.json").write_text(
                    json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except OSError:
                pass
            return _finalize({"ok": False, "error": record["error"]})
        data = resp.json()
        if not isinstance(data, dict):
            record["ok"] = False
            record["error"] = "invalid json"
            try:
                (run_dir / "thread_router.json").write_text(
                    json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except OSError:
                pass
            return _finalize({"ok": False, "error": "invalid response json"})
        msg = _extract_chat_message_content(data)
        record["message_content"] = msg
        parsed = _parse_json_object_from_llm(msg)
        record["parsed"] = parsed
        if not isinstance(parsed, dict):
            repaired, repair_info = _repair_json_once(
                raw_text=msg,
                schema_name="thread_router",
                schema=router_schema,
                model=THREAD_ROUTER_MODEL,
                url=url,
                timeout_sec=THREAD_ROUTER_TIMEOUT_SEC,
                max_tokens=180,
            )
            record.update(repair_info)
            if isinstance(repaired, dict):
                parsed = repaired
                record["parsed"] = repaired
            else:
                record["ok"] = False
                record["error"] = "unparseable model output"
                try:
                    (run_dir / "thread_router.json").write_text(
                        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                except OSError:
                    pass
                return _finalize({"ok": False, "error": "unparseable model output"})
        action = str(parsed.get("action") or "").strip().upper()
        tid_raw = str(parsed.get("target_thread_id") or "").strip()
        reason = str(parsed.get("reason") or "").strip()
        if action not in ("CONTINUE", "NEW"):
            record["ok"] = False
            record["error"] = "invalid action"
            try:
                (run_dir / "thread_router.json").write_text(
                    json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except OSError:
                pass
            return _finalize({"ok": False, "error": "invalid action"})
        record["ok"] = True
        record["action"] = action
        record["target_thread_id"] = tid_raw
        record["reason"] = reason
        try:
            (run_dir / "thread_router.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            pass
        return _finalize({"ok": True, "action": action, "target_thread_id": tid_raw, "reason": reason})
    except requests.RequestException as exc:
        record["ok"] = False
        record["error"] = str(exc)
        try:
            (run_dir / "thread_router.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            pass
        return _finalize({"ok": False, "error": str(exc)})


def _apply_thread_router_decision(session: dict, llm_result: dict) -> str:
    """根据分流结果切换 active thread 并 hydrate；返回最终 thread_id。"""
    threads = session.setdefault("threads", {})
    if not isinstance(threads, dict):
        threads = {}
        session["threads"] = threads
    sid = str(session.get("session_id") or "")

    def _fallback_tid() -> str:
        tid0 = str(session.get("active_thread_id") or "").strip()
        if tid0 and tid0 in threads:
            return tid0
        if threads:
            return sorted(threads.keys())[0]
        nid = _new_thread_id()
        threads[nid] = _empty_thread_payload(sid)
        return nid

    if not isinstance(llm_result, dict) or not llm_result.get("ok"):
        tid = _fallback_tid()
        session["active_thread_id"] = tid
        _hydrate_thread_into_session(session)
        return tid

    action = str(llm_result.get("action") or "").strip().upper()
    raw_id = str(llm_result.get("target_thread_id") or "").strip()
    if action == "NEW":
        nid = _new_thread_id()
        threads[nid] = _empty_thread_payload(sid)
        session["active_thread_id"] = nid
        _hydrate_thread_into_session(session)
        return nid

    tid = _normalize_thread_id(raw_id)
    if action == "CONTINUE" and tid and tid in threads:
        session["active_thread_id"] = tid
        _hydrate_thread_into_session(session)
        return tid

    tid = _fallback_tid()
    session["active_thread_id"] = tid
    _hydrate_thread_into_session(session)
    return tid


def _call_thread_topic_summarizer(*, session_id: str, thread_id: str, qa_pairs: list[dict], run_dir: Path) -> dict:
    """调用 Qwen-4B 摘要 thread 主题。"""
    prompt = {
        "thread_id": thread_id,
        "qa_pairs": qa_pairs[-max(1, THREAD_TOPIC_MAX_TURNS):],
    }
    system = (
        "你是对话线程主题摘要器。请根据用户问答，输出该线程稳定主题。\n"
        "仅输出 JSON：{\"topic\":\"...\"}。\n"
        "topic 要简短（8-24字），任务导向，避免口水话。"
    )
    url = str(THREAD_ROUTER_API_BASE or "").rstrip("/") + "/chat/completions"
    topic_schema = {
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
        },
        "required": ["topic"],
        "additionalProperties": False,
    }
    body = {
        "model": THREAD_TOPIC_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        "temperature": 0,
        "max_tokens": 120,
        "response_format": _chat_response_format_json_schema("thread_topic", topic_schema),
    }
    rec: dict = {"session_id": session_id, "thread_id": thread_id, "request": body, "url": url}

    def _finalize(result: dict) -> dict:
        return result
    try:
        resp = requests.post(
            url,
            json=body,
            headers={
                "Authorization": f"Bearer {DEMO_LLM_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=THREAD_TOPIC_TIMEOUT_SEC,
        )
        rec["status_code"] = resp.status_code
        rec["response_text"] = (resp.text or "")[:8000]
        if resp.status_code != 200:
            rec["ok"] = False
            rec["error"] = f"HTTP {resp.status_code}"
            try:
                (run_dir / "thread_topic.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
            except OSError:
                pass
            return _finalize({"ok": False, "error": rec["error"]})
        data = resp.json()
        msg = _extract_chat_message_content(data)
        rec["message_content"] = msg
        parsed = _parse_json_object_from_llm(msg)
        rec["parsed"] = parsed
        if not isinstance(parsed, dict):
            repaired, repair_info = _repair_json_once(
                raw_text=msg,
                schema_name="thread_topic",
                schema=topic_schema,
                model=THREAD_TOPIC_MODEL,
                url=url,
                timeout_sec=THREAD_TOPIC_TIMEOUT_SEC,
                max_tokens=120,
            )
            rec.update(repair_info)
            if isinstance(repaired, dict):
                parsed = repaired
                rec["parsed"] = repaired
        topic = ""
        if isinstance(parsed, dict):
            topic = _trim_text(str(parsed.get("topic") or "").strip(), 120)
        rec["ok"] = bool(topic)
        rec["topic"] = topic
        try:
            (run_dir / "thread_topic.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass
        if not topic:
            return _finalize({"ok": False, "error": "empty topic"})
        return _finalize({"ok": True, "topic": topic})
    except requests.RequestException as exc:
        rec["ok"] = False
        rec["error"] = str(exc)
        try:
            (run_dir / "thread_topic.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass
        return _finalize({"ok": False, "error": str(exc)})


def _schedule_thread_topic_refresh(*, session_id: str, thread_id: str, run_dir: Path) -> None:
    """后台异步更新 thread topic。"""

    def _worker() -> None:
        try:
            with _session_guard(session_id):
                session = _load_session_state(session_id)
                threads = session.get("threads")
                if not isinstance(threads, dict):
                    return
                bucket = threads.get(thread_id)
                if not isinstance(bucket, dict):
                    return
                history = bucket.get("summary_history")
                if not isinstance(history, list) or not history:
                    return
                qa_pairs: list[dict] = []
                for item in history[-max(1, THREAD_TOPIC_MAX_TURNS):]:
                    if not isinstance(item, dict):
                        continue
                    q = _trim_text(str(item.get("query") or "").strip(), 240)
                    a = _trim_text(str(item.get("final_answer") or "").strip(), 320)
                    if not q and not a:
                        continue
                    qa_pairs.append({"q": q, "a": a})
                if not qa_pairs:
                    return
                result = _call_thread_topic_summarizer(
                    session_id=session_id, thread_id=thread_id, qa_pairs=qa_pairs, run_dir=run_dir
                )
                if not result.get("ok"):
                    return
                bucket["topic"] = _trim_text(str(result.get("topic") or "").strip(), 120)
                _save_session_state(session)
        except Exception:
            return

    t = threading.Thread(target=_worker, name=f"thread-topic-{session_id}-{thread_id}", daemon=True)
    t.start()


def _call_query_trajectory_summarizer(
    *,
    session_id: str,
    thread_id: str,
    query_id: str,
    query: str,
    steps: list[dict],
    run_dir: Path,
) -> dict:
    """小模型总结单条 query 轨迹（供后续 Planner 老 query 仅用 result_summary）。"""
    system = prompts.build_query_trajectory_summary_system_prompt()
    user = prompts.build_query_trajectory_summary_user_prompt(
        query=query,
        steps=steps,
    )
    url = str(THREAD_ROUTER_API_BASE or "").rstrip("/") + "/chat/completions"
    summary_schema = tool_schemas.QUERY_TRAJECTORY_SUMMARY_SCHEMA
    body = {
        "model": QUERY_TRAJ_SUMMARY_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": 500,
        "response_format": tool_schemas.QUERY_TRAJECTORY_SUMMARY_RESPONSE_FORMAT,
    }
    rec: dict = {"session_id": session_id, "thread_id": thread_id, "query_id": query_id, "request": body, "url": url}

    def _finalize(result: dict) -> dict:
        return result

    try:
        resp = requests.post(
            url,
            json=body,
            headers={
                "Authorization": f"Bearer {DEMO_LLM_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=QUERY_TRAJ_SUMMARY_TIMEOUT_SEC,
        )
        rec["status_code"] = resp.status_code
        rec["response_text"] = (resp.text or "")[:8000]
        if resp.status_code != 200:
            rec["ok"] = False
            rec["error"] = f"HTTP {resp.status_code}"
            try:
                (run_dir / "query_traj_summary.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
            except OSError:
                pass
            return _finalize({"ok": False, "error": rec["error"]})
        data = resp.json()
        msg = _extract_chat_message_content(data)
        rec["message_content"] = msg
        parsed = _parse_json_object_from_llm(msg)
        rec["parsed"] = parsed
        if not isinstance(parsed, dict):
            repaired, repair_info = _repair_json_once(
                raw_text=msg,
                schema_name="query_traj_summary",
                schema=summary_schema,
                model=QUERY_TRAJ_SUMMARY_MODEL,
                url=url,
                timeout_sec=QUERY_TRAJ_SUMMARY_TIMEOUT_SEC,
                max_tokens=500,
            )
            rec.update(repair_info)
            if isinstance(repaired, dict):
                parsed = repaired
                rec["parsed"] = repaired
        summary = ""
        if isinstance(parsed, dict):
            summary = _trim_text(str(parsed.get("result_summary") or "").strip(), 2000)
        rec["ok"] = bool(summary)
        rec["result_summary"] = summary
        try:
            (run_dir / "query_traj_summary.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass
        if not summary:
            return _finalize({"ok": False, "error": "empty summary"})
        return _finalize({"ok": True, "result_summary": summary})
    except requests.RequestException as exc:
        rec["ok"] = False
        rec["error"] = str(exc)
        try:
            (run_dir / "query_traj_summary.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass
        return _finalize({"ok": False, "error": str(exc)})


def _schedule_query_trajectory_summary(*, session_id: str, thread_id: str, query_id: str, run_dir: Path) -> None:
    """异步写入 query_trajectories[].result_summary。"""

    def _worker() -> None:
        try:
            with _session_guard(session_id):
                session = _load_session_state(session_id)
                threads = session.get("threads")
                if not isinstance(threads, dict):
                    return
                if thread_id not in threads:
                    return
                session["active_thread_id"] = thread_id
                _hydrate_thread_into_session(session)
                ledger = session.get("raw_ledger")
                if not isinstance(ledger, list):
                    ledger = []
                query = ms.QueryTrajectoryStore.user_text_for_query(
                    session,
                    query_id=query_id,
                    ledger=ledger if isinstance(ledger, list) else [],
                )
                steps = ms.QueryTrajectoryStore.resolve_query_steps(
                    session,
                    query_id=query_id,
                    ledger=ledger if isinstance(ledger, list) else [],
                )
                result = _call_query_trajectory_summarizer(
                    session_id=session_id,
                    thread_id=thread_id,
                    query_id=query_id,
                    query=query,
                    steps=steps,
                    run_dir=run_dir,
                )
                if not result.get("ok"):
                    return
                ms.QueryTrajectoryStore.set_result_summary(
                    session, query_id=query_id, summary=str(result.get("result_summary") or "")
                )
                _save_session_state(session)
        except Exception:
            return

    t = threading.Thread(
        target=_worker,
        name=f"query-traj-summary-{session_id}-{query_id}",
        daemon=True,
    )
    t.start()


def _get_session_lock(session_id: str) -> threading.Lock:
    sid = _normalize_session_id(session_id)
    with _session_locks_guard:
        lock = _session_locks.get(sid)
        if lock is None:
            lock = threading.Lock()
            _session_locks[sid] = lock
        return lock


@contextmanager
def _session_guard(session_id: str):
    lock = _get_session_lock(session_id)
    lock.acquire()
    try:
        yield
    finally:
        lock.release()


def _resolve_effective_image_path(current_image_path: str, session: dict) -> str:
    now_path = str(current_image_path or "").strip()
    if now_path and Path(now_path).is_file():
        return str(Path(now_path).resolve())

    last_path = str(session.get("last_image_path") or "").strip()
    if last_path and Path(last_path).is_file():
        return str(Path(last_path).resolve())

    return ""


def _parse_bool_flag(raw, default: bool = False) -> bool:
    if raw is None:
        return default
    text = str(raw).strip().lower()
    if not text:
        return default
    return text in {"1", "true", "yes", "on", "y"}


def _parse_index_selection(raw) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, int):
        return [raw] if raw >= 0 else []
    if isinstance(raw, list):
        out: list[int] = []
        for item in raw:
            out.extend(_parse_index_selection(item))
        return out
    text = str(raw).strip()
    if not text:
        return []
    out: list[int] = []
    for part in re.split(r"[\s,]+", text):
        token = str(part).strip()
        if not token:
            continue
        try:
            idx = int(token)
        except ValueError:
            continue
        if idx >= 0:
            out.append(idx)
    return out


def _load_queries_from_csv(csv_path: str) -> list[dict]:
    path = Path(str(csv_path or "").strip()).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"csv file not found: {path}")
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        except csv.Error:
            dialect = csv.get_dialect("excel-tab") if "\t" in sample and "," not in sample else csv.get_dialect("excel")
        reader = csv.DictReader(f, dialect=dialect)
        if not reader.fieldnames:
            raise ValueError("csv header is required")
        fieldnames = [str(name or "").strip() for name in reader.fieldnames if str(name or "").strip()]
        query_col = ""
        for name in fieldnames:
            if name.lower() in {"query", "text", "question", "prompt", "示例题目", "题目", "问题"}:
                query_col = name
                break
        records = [row if isinstance(row, dict) else {} for row in reader]
        if not query_col and fieldnames:
            candidate_scores: dict[str, tuple[int, int]] = {}
            for name in fieldnames:
                values = [str((rec.get(name) or "")).strip() for rec in records[:5]]
                values = [v for v in values if v]
                if not values:
                    continue
                numeric_like = sum(1 for v in values if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", v))
                avg_len = sum(len(v) for v in values) // max(1, len(values))
                candidate_scores[name] = (avg_len, -numeric_like)
            if candidate_scores:
                query_col = sorted(candidate_scores.items(), key=lambda item: (item[1][0], item[1][1]), reverse=True)[0][0]
            else:
                query_col = fieldnames[-1]
        for idx, row in enumerate(records, start=0):
            record = row if isinstance(row, dict) else {}
            query = str(record.get(query_col) or "").strip()
            if not query:
                continue
            rows.append({"index": idx, "query": query, "row": record})
    return rows


def _beijing_date_prefix() -> str:
    return datetime.now(BEIJING_TZ).strftime("%Y%m%d")


def _session_path_by_date(session_id: str, date_prefix: str) -> Path:
    return SESSIONS_DIR / date_prefix / f"{date_prefix}_{session_id}.json"


def _iter_all_session_files() -> list[Path]:
    if not SESSIONS_DIR.exists():
        return []
    out: list[Path] = []
    for p in SESSIONS_DIR.rglob("*.json"):
        if p.is_file():
            out.append(p)
    return out


def _match_session_file(path: Path, session_id: str) -> bool:
    name = path.name
    if name == f"{session_id}.json":
        return True
    return bool(re.match(rf"^\d{{8}}_{re.escape(session_id)}\.json$", name))


def _find_existing_session_file(session_id: str) -> Path | None:
    candidates = [p for p in _iter_all_session_files() if _match_session_file(p, session_id)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _session_file_for_load(session_id: str) -> Path:
    existed = _find_existing_session_file(session_id)
    if existed is not None:
        return existed
    return _session_path_by_date(session_id, _beijing_date_prefix())


def _session_file_for_save(session_id: str) -> Path:
    return _session_path_by_date(session_id, _beijing_date_prefix())


def _delete_session_files(session_id: str) -> None:
    for p in [x for x in _iter_all_session_files() if _match_session_file(x, session_id)]:
        p.unlink(missing_ok=True)


def _client_ip_from_request(handler: BaseHTTPRequestHandler) -> str:
    forwarded = str(handler.headers.get("X-Forwarded-For") or "").strip()
    if forwarded:
        return forwarded.split(",")[0].strip()
    client = handler.client_address[0] if handler.client_address else ""
    return str(client or "").strip() or "unknown"


def _client_scope_from_request(handler: BaseHTTPRequestHandler) -> str:
    ip = _client_ip_from_request(handler)
    browser_id = str(handler.headers.get("X-Client-Id") or "").strip()
    if not browser_id:
        browser_id = "unknown_browser"
    browser_id = re.sub(r"[^a-zA-Z0-9_-]", "_", browser_id)[:80] or "unknown_browser"
    return f"{ip}::{browser_id}"


def _list_sessions_for_client_scope(client_scope: str) -> list[dict]:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []
    for path in sorted(_iter_all_session_files(), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        if str(data.get("client_scope") or "").strip() != client_scope:
            continue
        sid = str(data.get("session_id") or "").strip()
        if not sid:
            continue
        history = _summary_history_for_session_file(data)
        latest = history[-1] if history else {}
        latest = latest if isinstance(latest, dict) else {}
        title = str(latest.get("query") or "").strip() or "新对话"
        preview = str(latest.get("final_answer") or "").strip()
        items.append(
            {
                "session_id": sid,
                "title": _trim_text(title, 60),
                "preview": _trim_text(preview, 100),
                "updated_at": str(data.get("updated_at") or ""),
                "turns": len(history),
            }
        )
    return items


def _load_session_state(session_id: str) -> dict:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = _session_file_for_load(session_id)

    def _blank_shell() -> dict:
        return {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "client_ip": "",
            "client_scope": "",
            "pending_clarification": {},
        }

    if not path.is_file():
        data = _blank_shell()
        _migrate_session_threads(data, from_disk=False)
        _hydrate_thread_into_session(data)
        ms.LedgerStore.migrate_schema(data)
        return data
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("invalid session file")
    except (OSError, json.JSONDecodeError, ValueError):
        data = _blank_shell()
        _migrate_session_threads(data, from_disk=False)
        _hydrate_thread_into_session(data)
        ms.LedgerStore.migrate_schema(data)
        return data
    data["session_id"] = session_id
    data.setdefault("client_ip", "")
    data.setdefault("client_scope", "")
    data["pending_clarification"] = clarification_state.normalize_pending_clarification(
        data.get("pending_clarification"),
        normalize_thread_id=_normalize_thread_id,
        normalize_action=agent.normalize_agent_action,
    )
    _migrate_session_threads(data, from_disk=True)
    _hydrate_thread_into_session(data)
    ms.LedgerStore.migrate_schema(data)
    return data


def _save_session_state(session: dict) -> None:
    session_id = _normalize_session_id(str(session.get("session_id") or ""))
    session["session_id"] = session_id
    session["updated_at"] = datetime.now().isoformat(timespec="seconds")

    if not isinstance(session.get("threads"), dict) or not session.get("threads"):
        _migrate_session_threads(session, from_disk=False)
        _hydrate_thread_into_session(session)

    ms.LedgerStore.migrate_schema(session)

    session["raw_ledger"] = ms.LedgerStore.normalize_raw_ledger(
        session_id, session.get("raw_ledger"), default_thread_id=str(session.get("active_thread_id") or "")
    )
    tid = str(session.get("active_thread_id") or "").strip()
    session["query_trajectories"] = ms.QueryTrajectoryStore.normalize_list(
        session.get("query_trajectories"), session_id=session_id, thread_id_default=tid
    )
    session["thread_aux_state"] = ms.LedgerStore.normalize_thread_aux_state(
        session_id, session.get("thread_aux_state")
    )
    session["thread_aux_state"]["session_id"] = session_id
    session["thread_aux_state"]["ledger_cursor"] = ms.LedgerStore.ledger_max_seq(session)
    session.pop("working_memory", None)
    session.pop("session_memory", None)
    session["pending_clarification"] = clarification_state.normalize_pending_clarification(
        session.get("pending_clarification"),
        normalize_thread_id=_normalize_thread_id,
        normalize_action=agent.normalize_agent_action,
    )

    sh = session.get("summary_history", [])
    if isinstance(sh, list):
        session["summary_history"] = sh[-SESSION_SUMMARY_LIMIT:]
    else:
        session["summary_history"] = []

    _persist_thread_from_session(session)

    out: dict = {}
    for key, val in session.items():
        if key in THREAD_PAYLOAD_KEYS:
            continue
        out[key] = val
    threads = session.get("threads")
    out["threads"] = threads if isinstance(threads, dict) else {}
    out["active_thread_id"] = str(session.get("active_thread_id") or "").strip()

    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    dst = _session_file_for_save(session_id)
    dst.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(dst.parent),
        delete=False,
        suffix=".tmp",
    ) as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        tmp_path = Path(f.name)

    tmp_path.replace(dst)


class DemoHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        return

    def _emit_stream(self, obj: dict) -> None:
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        self.wfile.write(line.encode("utf-8"))
        self.wfile.flush()

    def _start_ndjson(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

    def _send_json(self, obj: dict, status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str, status: int = 200) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        ctype, _ = cgi.parse_header(self.headers.get("Content-Type", ""))
        if ctype != "application/json":
            raise ValueError("Content-Type must be application/json")
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(max(0, length))
        body = json.loads(raw.decode("utf-8") if raw else "{}")
        return body if isinstance(body, dict) else {}

    def _run_migration_advisor_test_case(
        self,
        *,
        query: str,
        session_id: str,
        case_index: int,
        csv_row: dict | None = None,
        force_migration_advisor: bool = True,
    ) -> dict:
        text = str(query or "").strip()
        if not text:
            raise ValueError("query is required")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(4)
        run_dir = RUNS_DIR / stamp
        run_dir.mkdir(parents=True, exist_ok=True)
        events: list[dict] = []
        old_emit = self._emit_stream

        def _capture_emit(obj: dict) -> None:
            if isinstance(obj, dict):
                events.append(dict(obj))

        self._emit_stream = _capture_emit
        try:
            with _session_guard(session_id):
                session = _load_session_state(session_id)
                session["client_ip"] = _client_ip_from_request(self)
                session["client_scope"] = _client_scope_from_request(self)
                _save_session_state(session)
            if not force_migration_advisor:
                raise ValueError("force_migration_advisor=false is not supported in test route")
            obs = self._run_migration_advisor_streaming(
                text=text,
                rag_trace=[],
                run_dir=run_dir,
                run_stamp=stamp,
                session_id=session_id,
                session={"session_id": session_id},
                emit_done=False,
            )
            markdown = str(obs.get("summary") or "").strip()
            result = {
                "ok": bool(obs.get("success")),
                "query": text,
                "case_index": case_index,
                "session_id": session_id,
                "run_stamp": stamp,
                "run_dir": str(run_dir),
                "markdown_path": str((run_dir / "migration_advisor_report.md").resolve()),
                "json_path": str((run_dir / "migration_advisor_report.json").resolve()),
                "report_summary": markdown,
                "events_count": len(events),
            }
            if isinstance(csv_row, dict) and csv_row:
                result["csv_row"] = csv_row
            return result
        finally:
            self._emit_stream = old_emit

    def _handle_migration_advisor_test(self, body: dict) -> None:
        payload = body if isinstance(body, dict) else {}
        csv_path = str(payload.get("csv_path") or "").strip()
        session_prefix = _normalize_session_id(str(payload.get("session_id_prefix") or "migration_test"))
        force_migration = _parse_bool_flag(payload.get("force_migration_advisor"), default=True)
        selected = _parse_index_selection(payload.get("indices"))
        if not selected:
            selected = _parse_index_selection(payload.get("index"))
        cases: list[dict] = []
        query = str(payload.get("query") or "").strip()
        if query:
            cases = [{"index": 0, "query": query, "row": {}}]
        elif csv_path:
            rows = _load_queries_from_csv(csv_path)
            if selected:
                wanted = set(selected)
                cases = []
                for row in rows:
                    raw_index = row.get("index")
                    try:
                        row_index = int(raw_index)
                    except (TypeError, ValueError):
                        row_index = -1
                    if row_index in wanted:
                        cases.append(row)
            else:
                cases = rows
        else:
            self._send_json({"ok": False, "error": "query or csv_path is required"}, status=400)
            return
        if not cases:
            self._send_json({"ok": False, "error": "no query selected"}, status=400)
            return
        results: list[dict] = []
        for row in cases:
            idx = int(row.get("index") or 0)
            session_id = _normalize_session_id(f"{session_prefix}_{idx}")
            result = self._run_migration_advisor_test_case(
                query=str(row.get("query") or ""),
                session_id=session_id,
                case_index=idx,
                csv_row=row.get("row") if isinstance(row.get("row"), dict) else {},
                force_migration_advisor=force_migration,
            )
            results.append(result)
        self._send_json(
            {
                "ok": True,
                "mode": "migration_advisor_test",
                "count": len(results),
                "force_migration_advisor": force_migration,
                "results": results,
            }
        )

    def _serve_demo_run_file(self, path: str) -> None:
        parts = [p for p in path.split("/") if p]
        if len(parts) < 3 or parts[0] != "demo-run":
            self._send_json({"error": "not found"}, status=404)
            return
        stamp = parts[1]
        rel = "/".join(parts[2:])
        if ".." in rel or rel.startswith("/"):
            self._send_json({"error": "forbidden"}, status=403)
            return
        base = (RUNS_DIR / stamp).resolve()
        target = (base / rel).resolve()
        try:
            target.relative_to(RUNS_DIR.resolve())
        except ValueError:
            self._send_json({"error": "forbidden"}, status=403)
            return
        if not target.is_file():
            self._send_json({"error": "not found"}, status=404)
            return
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(HTML_PAGE)
            return
        if parsed.path == "/sessions":
            client_ip = _client_ip_from_request(self)
            client_scope = _client_scope_from_request(self)
            sessions = _list_sessions_for_client_scope(client_scope)
            self._send_json({"ok": True, "client_ip": client_ip, "sessions": sessions})
            return
        if parsed.path == "/session":
            qs = parse_qs(parsed.query or "", keep_blank_values=False)
            sid = _normalize_session_id((qs.get("session_id") or [""])[0])
            req_thread_raw = (qs.get("thread_id") or [""])[0].strip()
            client_scope = _client_scope_from_request(self)
            session = _load_session_state(sid)
            if str(session.get("client_scope") or "").strip() != client_scope:
                self._send_json({"ok": False, "error": "session not found"}, status=404)
                return
            threads_map = session.get("threads")
            if not isinstance(threads_map, dict):
                threads_map = {}
            active_tid = str(session.get("active_thread_id") or "").strip()
            if req_thread_raw:
                view_tid = _normalize_thread_id(req_thread_raw)
                if view_tid not in threads_map:
                    self._send_json({"ok": False, "error": "thread not found"}, status=404)
                    return
                bucket = threads_map[view_tid]
            elif active_tid and active_tid in threads_map:
                bucket = threads_map[active_tid]
            else:
                bucket = {
                    "summary_history": session.get("summary_history"),
                    "chat_turns": session.get("chat_turns"),
                    "raw_ledger": session.get("raw_ledger"),
                }
            if not isinstance(bucket, dict):
                bucket = {}
            history = _aggregate_thread_payload_list(threads_map, "summary_history")
            if not history:
                history = bucket.get("summary_history")
                if not isinstance(history, list):
                    history = []
            chat_turns = _aggregate_thread_payload_list(threads_map, "chat_turns")
            if not chat_turns:
                chat_turns = bucket.get("chat_turns")
                if not isinstance(chat_turns, list):
                    chat_turns = []
            thread_ids = sorted(threads_map.keys()) if threads_map else []
            thread_topics: dict[str, str] = {}
            for tid in thread_ids:
                tb = threads_map.get(tid)
                if not isinstance(tb, dict):
                    continue
                topic = _trim_text(str(tb.get("topic") or "").strip(), 120)
                if topic:
                    thread_topics[tid] = topic
            history_refs: dict[str, list[dict]] = {}
            raw_ledgers: list[dict] = []
            if isinstance(threads_map, dict) and threads_map:
                for tid in thread_ids:
                    tb = threads_map.get(tid)
                    if not isinstance(tb, dict):
                        continue
                    vals = tb.get("raw_ledger")
                    if not isinstance(vals, list):
                        continue
                    raw_ledgers.extend([item for item in vals if isinstance(item, dict)])
            if not raw_ledgers:
                raw_ledger = bucket.get("raw_ledger")
                if not isinstance(raw_ledger, list):
                    raw_ledger = session.get("raw_ledger") if isinstance(session.get("raw_ledger"), list) else []
                raw_ledgers = [item for item in raw_ledger if isinstance(item, dict)]
            if raw_ledgers:
                for item in raw_ledgers:
                    if not isinstance(item, dict):
                        continue
                    if str(item.get("event_type") or "").upper() != "OBSERVATION":
                        continue
                    payload = item.get("payload")
                    if not isinstance(payload, dict):
                        continue
                    action = str(payload.get("action") or payload.get("_action") or "").strip()
                    if action != agent.ACTION_RAG_ANSWER:
                        continue
                    refs = payload.get("references")
                    if not isinstance(refs, list):
                        continue
                    query = str((payload.get("_action_input") or {}).get("query") or "").strip()
                    if not query:
                        continue
                    cleaned: list[dict] = []
                    seen: set[str] = set()
                    for ref in refs:
                        if not isinstance(ref, dict):
                            continue
                        url = str(ref.get("url") or "").strip()
                        if not url or url in seen:
                            continue
                        seen.add(url)
                        cleaned.append(
                            {
                                "doc_name": str(ref.get("doc_name") or "").strip(),
                                "url": url,
                            }
                        )
                    if cleaned:
                        history_refs[query] = cleaned
            self._send_json(
                {
                    "ok": True,
                    "session_id": sid,
                    "active_thread_id": active_tid,
                    "thread_ids": thread_ids,
                    "thread_topics": thread_topics,
                    "history": history,
                    "chat_turns": chat_turns,
                    "history_refs": history_refs,
                }
            )
            return
        up = unquote(parsed.path).lstrip("/")
        if up.startswith("demo-run/"):
            self._serve_demo_run_file(up)
            return
        self._send_json({"error": "not found"}, status=404)

    def _failure_observation(self, action: str, message: str) -> dict:
        return _build_observation(action, success=False, summary=str(message or "").strip(), error=str(message or "").strip())

    def _submit_playbook_feedback(self, body: dict) -> None:
        session_id = _normalize_session_id(str(body.get("session_id") or ""))
        if not session_id:
            self._send_json({"ok": False, "error": "session_id is required"}, status=400)
            return
        run_id = str(body.get("run_id") or "").strip()
        if not run_id:
            self._send_json({"ok": False, "error": "run_id is required"}, status=400)
            return
        with _session_guard(session_id):
            client_scope = _client_scope_from_request(self)
            session = _load_session_state(session_id)
            if str(session.get("client_scope") or "").strip() != client_scope:
                self._send_json({"ok": False, "error": "session not found"}, status=404)
                return
            feedback_type = str(body.get("feedback_type") or "other").strip() or "other"
            if feedback_type not in {"helpful", "harmful", "correction", "missing_evidence", "other"}:
                feedback_type = "other"
            payload = {
                "run_id": run_id,
                "feedback_type": feedback_type,
                "rating": body.get("rating") if isinstance(body.get("rating"), int) else None,
                "corrected_answer": str(body.get("corrected_answer") or "").strip() or None,
                "expected_evidence_ids": [
                    str(x).strip()
                    for x in (body.get("expected_evidence_ids") if isinstance(body.get("expected_evidence_ids"), list) else [])
                    if str(x).strip()
                ],
                "comment": str(body.get("comment") or "").strip() or None,
            }
            local_feedback_id = f"fb_{secrets.token_hex(8)}"
            feedbacks = session.get("playbook_feedbacks")
            if not isinstance(feedbacks, list):
                feedbacks = []
            feedbacks.append(
                {
                    "local_feedback_id": local_feedback_id,
                    "run_id": run_id,
                    "feedback_type": feedback_type,
                    "rating": payload.get("rating"),
                    "corrected_answer": payload.get("corrected_answer") or "",
                    "expected_evidence_ids": payload.get("expected_evidence_ids") or [],
                    "comment": payload.get("comment") or "",
                    "status": "queued",
                    "response": {},
                    "error": "",
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            session["playbook_feedbacks"] = feedbacks[-100:]
            _save_session_state(session)

        def _worker() -> None:
            result: dict = {}
            error = ""
            status = "submitted"
            try:
                resp = requests.post(
                    _playbook_feedback_url(),
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=(10, 60),
                )
            except requests.RequestException as exc:
                status = "failed"
                error = f"Playbook feedback request failed: {exc}"
            else:
                if resp.status_code != 200:
                    status = "failed"
                    error = f"Playbook feedback HTTP {resp.status_code}: {(resp.text or '')[:1000]}"
                else:
                    try:
                        parsed = resp.json()
                    except json.JSONDecodeError:
                        parsed = {"raw": (resp.text or "")[:1000]}
                    result = parsed if isinstance(parsed, dict) else {}
            try:
                with _session_guard(session_id):
                    session2 = _load_session_state(session_id)
                    feedbacks2 = session2.get("playbook_feedbacks")
                    if not isinstance(feedbacks2, list):
                        feedbacks2 = []
                    found = False
                    for item in reversed(feedbacks2):
                        if not isinstance(item, dict):
                            continue
                        if str(item.get("local_feedback_id") or "") != local_feedback_id:
                            continue
                        item["status"] = status
                        item["response"] = result
                        item["error"] = error
                        item["updated_at"] = datetime.now().isoformat(timespec="seconds")
                        found = True
                        break
                    if not found:
                        feedbacks2.append(
                            {
                                "local_feedback_id": local_feedback_id,
                                "run_id": run_id,
                                "feedback_type": feedback_type,
                                "status": status,
                                "response": result,
                                "error": error,
                                "updated_at": datetime.now().isoformat(timespec="seconds"),
                            }
                        )
                    session2["playbook_feedbacks"] = feedbacks2[-100:]
                    _save_session_state(session2)
            except Exception:
                return

        t = threading.Thread(
            target=_worker,
            name=f"playbook-feedback-{session_id}-{local_feedback_id}",
            daemon=True,
        )
        t.start()
        self._send_json({"ok": True, "status": "queued", "local_feedback_id": local_feedback_id})

    def _run_agent_loop(
        self,
        *,
        text: str,
        image_path: str,
        api_key: str,
        api_base: str,
        run_dir: Path,
        run_stamp: str,
        session: dict,
        forced_first_step: dict | None = None,
    ) -> dict:
        tool_executor = ToolExecutor(
            emit=self._emit_stream,
            failure_observation=self._failure_observation,
            run_rag_streaming=self._run_rag_streaming,
            run_flux_only_streaming=self._run_flux_only_streaming,
            run_detection_only_streaming=self._run_detection_only_streaming,
            run_rex_detection_only_streaming=self._run_rex_detection_only_streaming,
            run_pipeline_streaming=self._run_pipeline_streaming,
            run_migration_advisor_streaming=self._run_migration_advisor_streaming,
            run_adela_cli_streaming=self._run_adela_cli_streaming,
            resolve_adela_model_reference=self._resolve_adela_model_reference_via_rag,
        )
        orchestrator = agent.AgentOrchestrator(
            emit=self._emit_stream, execute_tool=tool_executor.execute
        )
        return orchestrator.run(
            text=text,
            image_path=image_path,
            api_key=api_key,
            api_base=api_base,
            run_dir=run_dir,
            run_stamp=run_stamp,
            session=session,
            forced_first_step=forced_first_step,
        )

    def _run_rag_streaming(
        self, text: str, run_dir: Path, run_stamp: str, *, emit_done: bool = True
    ) -> dict | None:
        emit = self._emit_stream
        t0 = datetime.now()
        query = (text or "").strip()
        if not query:
            emit({"type": "error", "message": "缺少查询内容"})
            if emit_done:
                emit({"type": "done", "ok": False})
            return None
        rag_out = run_dir / "rag_response.json"
        api_mode = gbrain_rag.rag_api_mode()
        use_unified = api_mode == gbrain_rag.RAG_API_MODE_UNIFIED
        if use_unified:
            url = gbrain_rag.unified_query_url()
            payload = gbrain_rag.build_unified_query_payload(query, stream=False)
        else:
            url = gbrain_rag.playbook_query_url()
            payload = gbrain_rag.build_playbook_query_payload(query, stream=False)
        _append_run_log(
            run_stamp,
            "rag_query_request: "
            + json.dumps(
                gbrain_rag.debug_request_blob(for_query=True, query=query),
                ensure_ascii=False,
            ),
        )
        rag_label = "Unified RAG" if use_unified else "Playbook RAG"

        try:
            final_packet = gbrain_rag.post_rag_json(url, payload, stream=False, timeout=(10, 300))
        except requests.HTTPError as exc:
            body = ""
            if getattr(exc, "response", None) is not None:
                body = (exc.response.text or "")[:2000]
            emit({"type": "error", "message": f"{rag_label} HTTP 错误: {body or exc}"})
            if emit_done:
                emit({"type": "done", "ok": False})
            return None
        except requests.RequestException as exc:
            emit({"type": "error", "message": f"{rag_label} 问答请求失败: {exc}"})
            if emit_done:
                emit({"type": "done", "ok": False})
            return None
        except ValueError as exc:
            emit({"type": "error", "message": f"{rag_label} 问答响应解析失败: {exc}"})
            if emit_done:
                emit({"type": "done", "ok": False})
            return None

        answer = gbrain_rag.extract_answer(final_packet)
        refs = _normalize_rag_refs(final_packet.get("reference"))
        if not refs:
            refs = _normalize_rag_refs(final_packet.get("evidences"))
        if not refs:
            refs = _normalize_rag_refs(final_packet.get("fused_evidences"))
        retrieved_chunks = gbrain_rag.extract_evidences(final_packet)
        run_id = str(final_packet.get("run_id") or "").strip()
        evidence_ids = _rag_evidence_ids(final_packet.get("evidences"))
        knowledge_base_fully_answered = agent.normalize_knowledge_base_score(
            final_packet.get("knowledge_base_fully_answered")
        )

        try:
            rag_out.write_text(
                json.dumps(
                    {
                        "success": bool(answer),
                        "answer": answer,
                        "reference": refs,
                        "retrieved_chunks": retrieved_chunks,
                        "run_id": run_id,
                        "evidence_ids": evidence_ids,
                        "knowledge_base_fully_answered": knowledge_base_fully_answered,
                        "raw_last_packet": final_packet,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass

        if not answer:
            emit({"type": "error", "message": "RAG 未返回有效 answer，请检查流式服务响应"})
            if emit_done:
                emit({"type": "done", "ok": False})
            return None
        emit({"type": "direct_reply", "text": answer})
        elapsed_ms = int((datetime.now() - t0).total_seconds() * 1000)
        emit({"type": "step_timing", "step": "rag", "elapsed_ms": elapsed_ms})
        if run_id:
            emit({"type": "rag_run", "run_id": run_id, "evidence_ids": evidence_ids})
        if refs:
            emit({"type": "rag_references", "references": refs})
        if emit_done:
            emit({"type": "done", "ok": True})
        observation_kwargs = {
            "answer": answer,
            "elapsed_ms": elapsed_ms,
            "references": refs,
            "run_id": run_id,
            "evidence_ids": evidence_ids,
            "knowledge_base_fully_answered": knowledge_base_fully_answered,
        }
        if agent.is_rag_miss(
            {"knowledge_base_fully_answered": knowledge_base_fully_answered}
        ):
            observation_kwargs["retrieved_chunks"] = retrieved_chunks
        return _build_observation(
            agent.ACTION_RAG_ANSWER,
            **observation_kwargs,
        )

    def _run_playbook_retrieve(
        self,
        *,
        query: str,
        field: str,
        run_dir: Path,
        run_stamp: str,
    ) -> dict:
        q = str(query or "").strip()
        f = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(field or "field").strip()).strip("._") or "field"
        api_mode = gbrain_rag.rag_api_mode()
        use_unified = api_mode == gbrain_rag.RAG_API_MODE_UNIFIED
        if use_unified:
            url = gbrain_rag.unified_retrieve_url()
            payload = gbrain_rag.build_unified_retrieve_payload(q)
        else:
            url = gbrain_rag.playbook_retrieve_url()
            payload = gbrain_rag.build_playbook_retrieve_payload(q)
        req_blob = gbrain_rag.debug_request_blob(for_query=False, query=q, extra={"field": f})
        _append_run_log(
            run_stamp,
            "rag_retrieve_request: " + json.dumps(req_blob, ensure_ascii=False),
        )
        result: dict = {
            "success": False,
            "field": f,
            "query": q,
            "api_mode": api_mode,
            "retrieved_chunks": [],
            "full_documents": [],
            "raw_response": {},
            "error": "",
        }
        try:
            data = gbrain_rag.post_rag_json(url, payload, timeout=(10, 120))
        except requests.HTTPError as exc:
            body = ""
            if getattr(exc, "response", None) is not None:
                body = (exc.response.text or "")[:1000]
            result["error"] = body or str(exc)
            return result
        except (requests.RequestException, ValueError) as exc:
            result["error"] = str(exc)
            return result

        chunks = gbrain_rag.extract_evidences(data)
        result.update(
            {
                "success": bool(chunks),
                "retrieved_chunks": chunks,
                "full_documents": gbrain_rag.extract_full_documents(data),
                "raw_response": data,
                "references": _normalize_rag_refs(data.get("reference"))
                or _normalize_rag_refs(data.get("evidences"))
                or _normalize_rag_refs(data.get("fused_evidences")),
            }
        )
        try:
            query_key = hashlib.sha1(q.encode("utf-8")).hexdigest()[:10]
            prefix = "unified_retrieve" if use_unified else "playbook_retrieve"
            out = run_dir / f"{prefix}_{f}_{query_key}.json"
            out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass
        return result

    def _run_migration_rex_annotation(
        self,
        image_paths: list[str],
        detection: dict,
        run_dir: Path,
        run_stamp: str,
    ) -> dict:
        from util.rex_label_extraction import summarize_label_hits

        ph = _pipeline_helpers()
        det = detection if isinstance(detection, dict) else {}
        lab = str(det.get("display_label") or det.get("target_label") or "").strip()
        classes = det.get("classes") if isinstance(det.get("classes"), list) else []
        if not lab and not classes and not det.get("tokens"):
            return {"success": False, "error": "未解析到检测目标描述"}

        imgs: list[Path] = []
        for raw in image_paths if isinstance(image_paths, list) else []:
            p = Path(str(raw or "")).resolve()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                imgs.append(p)
        if not imgs:
            return {"success": False, "error": "没有可用的参考图片"}

        def url(rel: str) -> str:
            return f"/demo-run/{run_stamp}/{rel}"

        input_dir = run_dir / "migration_rex_input"
        input_dir.mkdir(parents=True, exist_ok=True)
        image_map: dict[str, Path] = {}
        for idx, src in enumerate(imgs):
            dest = input_dir / f"{idx:02d}_{src.name}"
            if dest.resolve() != src.resolve():
                try:
                    dest.write_bytes(src.read_bytes())
                except OSError:
                    dest = src
            image_map[dest.name] = dest.resolve()

        rex_prompt_path = run_dir / ".rex_prompt_migration.json"
        rex_coco_out = run_dir / "migration_rex_detect_coco.json"
        result: dict = {
            "success": False,
            "label": lab,
            "task_mode": str(det.get("task_mode") or ""),
            "object": str(det.get("object") or ""),
            "classes": classes,
            "num_boxes": 0,
            "pred_bboxes": [],
            "annotated_urls": [],
            "annotated_image_path": "",
            "per_image": [],
            "label_hit_summary": {},
            "error": "",
        }
        try:
            ph.write_rex_prompt_json(rex_prompt_path, det)
            cmd = [
                "python3",
                "skills/rexomni-open-set-detection/scripts/run_detection.py",
                "--images",
                str(input_dir),
                "--prompt",
                str(rex_prompt_path),
                "--base-url",
                REX_BASE_URL,
                "--out",
                str(rex_coco_out),
            ]
            _append_run_log(run_stamp, f"command: {_mask_cmd(cmd)}")
            code, logs = _run_subprocess(cmd)
            if logs:
                _append_run_log(run_stamp, logs)
            if code != 0:
                result["error"] = logs or "Rex-Omni 检测失败"
                return result

            with open(rex_coco_out, "r", encoding="utf-8") as f:
                coco = json.load(f)
        finally:
            try:
                rex_prompt_path.unlink()
            except FileNotFoundError:
                pass

        by_id = ph.coco_to_pred_bboxes_by_image_id(coco)
        id_to_name = {
            int(im["id"]): str(im.get("file_name") or "")
            for im in coco.get("images", [])
            if isinstance(im, dict) and im.get("id") is not None
        }
        pred_records: list[dict] = []
        total_boxes = 0
        for iid in sorted(by_id.keys()):
            fname = id_to_name.get(iid) or image_map.get(iid, Path()).name
            src = image_map.get(fname)
            if not src or not src.is_file():
                for k, v in image_map.items():
                    if k == fname or fname.endswith(k):
                        src = v
                        fname = k
                        break
            boxes = by_id.get(iid, [])
            total_boxes += len(boxes)
            pred_records.append(
                {
                    "image": fname,
                    "source": "original",
                    "image_idx": iid,
                    "models": [{"model": "rex-omni", "pred_bboxes": boxes}],
                }
            )

        pred_path = run_dir / "migration_rex_prediction.json"
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(pred_records, f, indent=2, ensure_ascii=False)

        viz_dir = run_dir / "migration_rex_annotated"
        ph.draw_detection_overlays(image_map, pred_path, viz_dir)
        ann_urls: list[str] = []
        annotated_paths: list[str] = []
        annotated_path = ""
        url_by_name: dict[str, str] = {}
        for p in sorted(viz_dir.glob("*"), key=lambda x: x.name):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                rel = f"migration_rex_annotated/{p.name}"
                ann_urls.append(url(rel))
                url_by_name[p.name] = url(rel)
                annotated_paths.append(str(p.resolve()))
                if not annotated_path:
                    annotated_path = annotated_paths[-1]

        hit_summary = summarize_label_hits(coco, classes=classes)
        per_image_out: list[dict] = []
        for row in hit_summary.get("per_image") if isinstance(hit_summary.get("per_image"), list) else []:
            if not isinstance(row, dict):
                continue
            fname = str(row.get("file_name") or "")
            per_image_out.append(
                {
                    "file_name": fname,
                    "annotated_url": url_by_name.get(fname, ""),
                    "num_boxes": int(row.get("num_boxes") or 0),
                    "hit_labels": row.get("hit_labels") if isinstance(row.get("hit_labels"), list) else [],
                    "miss_labels": row.get("miss_labels") if isinstance(row.get("miss_labels"), list) else [],
                    "hits": row.get("hits") if isinstance(row.get("hits"), list) else [],
                    "misses": row.get("misses") if isinstance(row.get("misses"), list) else [],
                }
            )

        result.update(
            {
                "success": True,
                "num_boxes": total_boxes,
                "pred_bboxes": by_id.get(0, []),
                "pred_reports": pred_records,
                "annotated_urls": ann_urls,
                "annotated_image_path": annotated_path,
                "annotated_image_paths": annotated_paths[:MAX_UPLOAD_IMAGES],
                "per_image": per_image_out,
                "label_hit_summary": hit_summary,
            }
        )
        return result

    def _run_migration_advisor_streaming(
        self,
        *,
        text: str,
        rag_trace: list[dict] | None,
        run_dir: Path,
        run_stamp: str,
        session_id: str = "",
        image_path: str = "",
        image_paths: list[str] | None = None,
        session: dict | None = None,
        emit_done: bool = True,
    ) -> dict:
        emit = self._emit_stream
        user_query = str(text or "").strip()
        t0 = datetime.now()
        if not user_query:
            msg = "缺少迁移顾问原始需求"
            emit({"type": "error", "message": msg})
            if emit_done:
                emit({"type": "done", "ok": False})
            return _build_observation("migration_advisor", success=False, summary=msg, error=msg)

        emit(
            {
                "type": "meta",
                "flow": "migration_advisor",
                "decision": {
                    "action": "migration_advisor",
                    "reason": "基于历史资产、相似模型和可选样例图生成迁移顾问报告。",
                    "direct_reply": "",
                },
                "run_stamp": run_stamp,
                "step_index": 1,
            }
        )

        def _retrieve(field: str, query: str) -> dict:
            return self._run_playbook_retrieve(
                query=query,
                field=field,
                run_dir=run_dir,
                run_stamp=run_stamp,
            )

        effective_image = str(image_path or "").strip()
        if effective_image and Path(effective_image).is_file():
            effective_image = str(Path(effective_image).resolve())
        ref_paths = _collect_reference_image_paths(
            session if isinstance(session, dict) else {},
            run_dir,
            primary_path=effective_image,
        )
        if not ref_paths and isinstance(image_paths, list):
            ref_paths = [str(p) for p in image_paths if str(p).strip() and Path(p).is_file()]

        def _rex_annotate(img_paths: list[str], detection_targets: dict) -> dict:
            return self._run_migration_rex_annotation(
                img_paths,
                detection_targets,
                run_dir,
                run_stamp,
            )

        try:
            output = migration_advisor.run_workflow(
                user_query=user_query,
                rag_trace=rag_trace if isinstance(rag_trace, list) else [],
                retrieve=_retrieve,
                run_dir=run_dir,
                image_paths=ref_paths,
                run_rex_annotation=_rex_annotate if ref_paths else None,
                debug_meta={
                    "session_id": str(session_id or "").strip(),
                    "run_stamp": run_stamp,
                    "run_dir": str(run_dir),
                },
                emit=emit,
            )
        except Exception as exc:
            msg = f"迁移顾问报告生成失败: {exc}"
            emit({"type": "error", "message": msg})
            if emit_done:
                emit({"type": "done", "ok": False})
            return _build_observation("migration_advisor", success=False, summary=msg, error=msg)

        markdown = str(output.get("markdown") or "").strip()
        if markdown:
            emit({"type": "final_answer", "text": markdown})
        elapsed_ms = int((datetime.now() - t0).total_seconds() * 1000)
        emit({"type": "step_timing", "step": "migration_advisor", "elapsed_ms": elapsed_ms})
        if emit_done:
            emit({"type": "done", "ok": True})
        return _build_observation(
            "migration_advisor",
            success=True,
            summary=markdown,
            elapsed_ms=elapsed_ms,
            report=output.get("report") if isinstance(output.get("report"), dict) else {},
            plan=output.get("plan") if isinstance(output.get("plan"), dict) else {},
        )

    def _run_flux_only_streaming(
        self,
        text: str,
        image_path: str,
        flux_prompt: str,
        num_images: int,
        api_key: str,
        api_base: str,
        run_dir: Path,
        run_stamp: str,
        *,
        emit_done: bool = True,
    ) -> dict | None:
        emit = self._emit_stream
        ph = _pipeline_helpers()
        t_gen = datetime.now()

        def url(rel: str) -> str:
            return f"/demo-run/{run_stamp}/{rel}"

        task_text = (text or "").strip()
        hint = (flux_prompt or "").strip()
        if hint and hint not in task_text:
            task_text = f"{task_text}\n补充：{hint}" if task_text else hint
        if not task_text:
            emit({"type": "error", "message": "缺少图像任务描述"})
            if emit_done:
                emit({"type": "done", "ok": False})
            return None

        has_source_image = bool(image_path) and Path(image_path).is_file()
        intent_json = run_dir / "intent.json"
        cmd_intent = [
            "python3",
            "skills/user-intent-understanding/scripts/run_intent.py",
            "--text",
            task_text,
            "--api-base",
            DEMO_LLM_API_BASE,
            "--api-key",
            DEMO_LLM_API_KEY,
            "--out",
            str(intent_json),
        ]
        if has_source_image:
            cmd_intent.extend(["--image", image_path])
        _append_run_log(run_stamp, f"command: {_mask_cmd(cmd_intent)}")
        code, logs = _run_subprocess(cmd_intent)
        if logs:
            _append_run_log(run_stamp, logs)
        if code != 0:
            emit({"type": "error", "message": logs or "run_intent failed（Flux 流程）"})
            if emit_done:
                emit({"type": "done", "ok": False})
            return None

        prompts_json = run_dir / "prompts.json"
        cmd_prompts = [
            "python3",
            "skills/llm-prompts-generation/scripts/run_prompt_generation.py",
            "--intent",
            str(intent_json),
            "--task-text",
            task_text,
            "--api-base",
            DEMO_LLM_API_BASE,
            "--api-key",
            DEMO_LLM_API_KEY,
            "--out",
            str(prompts_json),
        ]
        _append_run_log(run_stamp, f"command: {_mask_cmd(cmd_prompts)}")
        code, logs = _run_subprocess(cmd_prompts)
        if logs:
            _append_run_log(run_stamp, logs)
        if code != 0:
            emit({"type": "error", "message": logs or "run_prompt_generation failed（Flux 流程）"})
            if emit_done:
                emit({"type": "done", "ok": False})
            return None

        try:
            all_prompts = ph.load_prompts_for_generation(prompts_json)
        except Exception as exc:
            emit({"type": "error", "message": f"读取 prompts 失败: {exc}"})
            if emit_done:
                emit({"type": "done", "ok": False})
            return None

        tail = all_prompts[1:] if len(all_prompts) > 1 else []
        if not tail:
            emit(
                {
                    "type": "error",
                    "message": "扩写 prompts 不足：跳过第 1 条后无可用的 prompt",
                }
            )
            if emit_done:
                emit({"type": "done", "ok": False})
            return None

        n = max(1, min(5, int(num_images)))
        gen_dir = run_dir / "generated_images"
        gen_dir.mkdir(parents=True, exist_ok=True)
        records: list[dict] = []
        for i in range(n):
            prompt = tail[i % len(tail)]
            out_image = gen_dir / f"generated_{i}.jpg"
            cmd_flux = [
                "python3",
                "skills/flux-image-generation/scripts/run_generation.py",
                "--prompt",
                prompt,
                "--api-base",
                api_base,
                "--api-key",
                api_key,
                "--out",
                str(out_image),
            ]
            if has_source_image:
                cmd_flux[2:2] = ["--source-image", image_path]
            _append_run_log(run_stamp, f"command: {_mask_cmd(cmd_flux)}")
            code, logs = _run_subprocess(cmd_flux)
            if logs:
                _append_run_log(run_stamp, logs)
            if code != 0:
                emit({"type": "error", "message": logs or f"flux generation failed（第 {i + 1} 张）"})
                if emit_done:
                    emit({"type": "done", "ok": False})
                return None
            records.append({"prompt": prompt, "saved_path": str(out_image.resolve())})
            rel = f"generated_images/generated_{i}.jpg"
            emit({"type": "generated_one", "index": i, "url": url(rel)})

        gen_meta = run_dir / "generated_images.json"
        with open(gen_meta, "w", encoding="utf-8") as f:
            json.dump({"generated_images": records}, f, indent=2, ensure_ascii=False)

        elapsed_ms = int((datetime.now() - t_gen).total_seconds() * 1000)
        emit({"type": "step_timing", "step": "flux_gen", "elapsed_ms": elapsed_ms})
        if emit_done:
            emit({"type": "done", "ok": True})
        return _build_observation(
            agent.ACTION_FLUX_IMAGE_GENERATION,
            summary=f"已生成 {len(records)} 张图片",
            num_images=len(records),
            generated_images=[
                {
                    "prompt": _trim_text(item.get("prompt", ""), 180),
                    "saved_path": item.get("saved_path", ""),
                }
                for item in records
            ],
            elapsed_ms=elapsed_ms,
        )

    def _run_detection_only_streaming(
        self,
        image_path: str,
        label: str,
        run_dir: Path,
        run_stamp: str,
        *,
        emit_done: bool = True,
    ) -> dict | None:
        emit = self._emit_stream
        ph = _pipeline_helpers()
        t_det = datetime.now()

        def url(rel: str) -> str:
            return f"/demo-run/{run_stamp}/{rel}"

        lab = (label or "").strip()
        if not lab:
            emit({"type": "error", "message": "未解析到检测目标描述（detection_label）"})
            if emit_done:
                emit({"type": "done", "ok": False})
            return None

        emit({"type": "detection_info", "label": lab})
        qwen_out = run_dir / "qwen_detect.json"
        cmd = [
            "python3",
            "skills/qwen-vlm-open-set-delection/scripts/run_detection.py",
            "--images",
            image_path,
            "--label",
            lab,
            "--base-url",
            QWEN_DETECTION_URL,
            "--out",
            str(qwen_out),
        ]
        _append_run_log(run_stamp, f"command: {_mask_cmd(cmd)}")
        code, logs = _run_subprocess(cmd)
        if logs:
            _append_run_log(run_stamp, logs)
        if code != 0:
            emit({"type": "error", "message": logs or "detection skill failed"})
            if emit_done:
                emit({"type": "done", "ok": False})
            return None

        with open(qwen_out, "r", encoding="utf-8") as f:
            pack = json.load(f)
        results = pack.get("results", [])
        if not results:
            emit({"type": "error", "message": "检测无结果"})
            if emit_done:
                emit({"type": "done", "ok": False})
            return None

        item = results[0]
        name = Path(item.get("image") or image_path).name
        src = Path(image_path).resolve()
        image_map = {name: src}
        boxes = [
            x["bbox"]
            for x in item.get("bboxes", [])
            if isinstance(x, dict) and "bbox" in x
        ]
        pred_records = [
            {
                "image": name,
                "source": "original",
                "image_idx": 0,
                "models": [{"model": "qwen3-vl-8b", "pred_bboxes": boxes}],
            }
        ]
        pred_path = run_dir / "prediction.json"
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(pred_records, f, indent=2, ensure_ascii=False)

        viz_dir = run_dir / "annotated_images"
        ph.draw_detection_overlays(image_map, pred_path, viz_dir)
        ann_urls = []
        for p in sorted(viz_dir.glob("*"), key=lambda x: x.name):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                ann_urls.append(url(f"annotated_images/{p.name}"))
        emit({"type": "annotated", "urls": ann_urls})
        elapsed_ms = int((datetime.now() - t_det).total_seconds() * 1000)
        emit({"type": "step_timing", "step": "detect", "elapsed_ms": elapsed_ms})
        if emit_done:
            emit({"type": "done", "ok": True})
        return _build_observation(
            agent.ACTION_QWEN_OPEN_SET_DETECTION,
            summary=f"Qwen 检测目标“{lab}”，得到 {len(boxes)} 个候选框",
            label=lab,
            num_boxes=len(boxes),
            annotated_urls=ann_urls,
            elapsed_ms=elapsed_ms,
        )

    def _run_rex_detection_only_streaming(
        self,
        image_path: str,
        label: str,
        run_dir: Path,
        run_stamp: str,
        *,
        emit_done: bool = True,
    ) -> dict | None:
        emit = self._emit_stream
        ph = _pipeline_helpers()
        t_det = datetime.now()

        def url(rel: str) -> str:
            return f"/demo-run/{run_stamp}/{rel}"

        lab = (label or "").strip()
        if not lab:
            emit({"type": "error", "message": "未解析到检测目标描述（detection_label）"})
            if emit_done:
                emit({"type": "done", "ok": False})
            return None

        try:
            detection = extract_rex_detection_labels(
                lab,
                api_key=_load_api_key(),
                base_url=os.environ.get("DEMO_ANSWER_API_BASE")
                or os.environ.get("DEMO_LLM_API_BASE")
                or "http://10.111.32.253:8000/v1",
            )
        except Exception as exc:
            emit({"type": "error", "message": f"检测标签抽取失败: {exc}"})
            if emit_done:
                emit({"type": "done", "ok": False})
            return None

        display_label = str(detection.get("display_label") or lab).strip()
        emit({"type": "detection_info", "label": display_label})
        rex_prompt_path = run_dir / ".rex_prompt_demo.json"
        rex_coco_out = run_dir / "rex_detect_coco.json"
        try:
            ph.write_rex_prompt_json(rex_prompt_path, detection)
            cmd = [
                "python3",
                "skills/rexomni-open-set-detection/scripts/run_detection.py",
                "--images",
                image_path,
                "--prompt",
                str(rex_prompt_path),
                "--base-url",
                REX_BASE_URL,
                "--out",
                str(rex_coco_out),
            ]
            _append_run_log(run_stamp, f"command: {_mask_cmd(cmd)}")
            code, logs = _run_subprocess(cmd)
            if logs:
                _append_run_log(run_stamp, logs)
            if code != 0:
                emit({"type": "error", "message": logs or "Rex-Omni 检测失败"})
                if emit_done:
                    emit({"type": "done", "ok": False})
                return None

            with open(rex_coco_out, "r", encoding="utf-8") as f:
                coco = json.load(f)
        finally:
            try:
                rex_prompt_path.unlink()
            except FileNotFoundError:
                pass

        by_id = ph.coco_to_pred_bboxes_by_image_id(coco)
        boxes = by_id.get(0, [])
        name = Path(image_path).name
        src = Path(image_path).resolve()
        image_map = {name: src}
        pred_records = [
            {
                "image": name,
                "source": "original",
                "image_idx": 0,
                "models": [{"model": "rex-omni", "pred_bboxes": boxes}],
            }
        ]
        pred_path = run_dir / "prediction.json"
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(pred_records, f, indent=2, ensure_ascii=False)

        viz_dir = run_dir / "annotated_images"
        ph.draw_detection_overlays(image_map, pred_path, viz_dir)
        ann_urls = []
        for p in sorted(viz_dir.glob("*"), key=lambda x: x.name):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                ann_urls.append(url(f"annotated_images/{p.name}"))
        emit({"type": "annotated", "urls": ann_urls})
        elapsed_ms = int((datetime.now() - t_det).total_seconds() * 1000)
        emit({"type": "step_timing", "step": "detect", "elapsed_ms": elapsed_ms})
        if emit_done:
            emit({"type": "done", "ok": True})
        return _build_observation(
            agent.ACTION_REXOMNI_OPEN_SET_DETECTION,
            summary=f"Rex-Omni 检测目标“{display_label}”，得到 {len(boxes)} 个候选框",
            label=display_label,
            num_boxes=len(boxes),
            annotated_urls=ann_urls,
            elapsed_ms=elapsed_ms,
        )

    def _run_pipeline_streaming(
        self,
        text: str,
        image_path: str,
        api_key: str,
        api_base: str,
        run_dir: Path,
        run_stamp: str,
        *,
        emit_done: bool = True,
    ) -> dict | None:
        emit = self._emit_stream
        ph = _pipeline_helpers()
        t_intent = datetime.now()

        def url(rel: str) -> str:
            return f"/demo-run/{run_stamp}/{rel}"

        ws = run_dir
        intent_json = ws / "intent.json"
        prompts_json = ws / "prompts.json"
        gen_json = ws / "generated_images.json"
        prediction_json = ws / "prediction.json"
        evaluation_json = ws / "evaluation.json"

        cmd_intent = [
            "python3",
            "skills/user-intent-understanding/scripts/run_intent.py",
            "--text",
            text,
            "--image",
            image_path,
            "--api-base",
            DEMO_LLM_API_BASE,
            "--api-key",
            DEMO_LLM_API_KEY,
            "--out",
            str(intent_json),
        ]
        _append_run_log(run_stamp, f"command: {_mask_cmd(cmd_intent)}")
        code, logs = _run_subprocess(cmd_intent)
        if logs:
            _append_run_log(run_stamp, logs)
        if code != 0:
            emit({"type": "error", "message": logs or "run_intent failed"})
            if emit_done:
                emit({"type": "done", "ok": False})
            return None
        with open(intent_json, "r", encoding="utf-8") as f:
            intent = json.load(f)
        emit({"type": "intent_summary", "summary": _intent_summary(intent)})
        emit({"type": "step_timing", "step": "intent", "elapsed_ms": int((datetime.now() - t_intent).total_seconds() * 1000)})

        t_gen = datetime.now()
        cmd_prompts = [
            "python3",
            "skills/llm-prompts-generation/scripts/run_prompt_generation.py",
            "--intent",
            str(intent_json),
            "--task-text",
            text,
            "--api-base",
            DEMO_LLM_API_BASE,
            "--api-key",
            DEMO_LLM_API_KEY,
            "--out",
            str(prompts_json),
        ]
        _append_run_log(run_stamp, f"command: {_mask_cmd(cmd_prompts)}")
        code, logs = _run_subprocess(cmd_prompts)
        if logs:
            _append_run_log(run_stamp, logs)
        if code != 0:
            emit({"type": "error", "message": logs or "run_prompt_generation failed"})
            if emit_done:
                emit({"type": "done", "ok": False})
            return None

        prompts_for_generation = ph.load_prompts_for_generation(prompts_json)
        if not prompts_for_generation:
            emit({"type": "error", "message": "No prompts found in prompts.json"})
            if emit_done:
                emit({"type": "done", "ok": False})
            return None

        generated_dir = ws / "generated_images"
        generated_dir.mkdir(parents=True, exist_ok=True)
        generated_records: list[dict] = []
        for i in range(NUM_GENERATED_IMAGES):
            prompt = prompts_for_generation[i % len(prompts_for_generation)]
            out_image = generated_dir / f"generated_{i}.jpg"
            cmd_flux = [
                "python3",
                "skills/flux-image-generation/scripts/run_generation.py",
                "--source-image",
                image_path,
                "--prompt",
                prompt,
                "--api-base",
                api_base,
                "--api-key",
                api_key,
                "--out",
                str(out_image),
            ]
            _append_run_log(run_stamp, f"command: {_mask_cmd(cmd_flux)}")
            code, logs = _run_subprocess(cmd_flux)
            if logs:
                _append_run_log(run_stamp, logs)
            if code != 0:
                emit({"type": "error", "message": logs or "flux generation failed"})
                if emit_done:
                    emit({"type": "done", "ok": False})
                return None
            generated_records.append({"prompt": prompt, "saved_path": str(out_image.resolve())})
            rel = f"generated_images/generated_{i}.jpg"
            emit({"type": "generated_one", "index": i, "url": url(rel)})

        with open(gen_json, "w", encoding="utf-8") as f:
            json.dump({"generated_images": generated_records}, f, indent=2, ensure_ascii=False)
        emit({"type": "step_timing", "step": "gen", "elapsed_ms": int((datetime.now() - t_gen).total_seconds() * 1000)})

        target_label = str(intent["target_label"]).replace("_", " ")
        gen_dir = ws / "generated_images"
        orig_image = Path(image_path).resolve()
        gen_files = sorted(
            (p for p in gen_dir.iterdir() if p.is_file() and p.suffix.lower() in ph.IMAGE_EXTENSIONS),
            key=lambda p: p.name,
        )
        image_paths = [orig_image] + gen_files
        images_abs = [str(p.resolve()) for p in image_paths]
        image_map = {p.name: p for p in image_paths}

        qwen_orig_tmp = ws / ".qwen_orig_tmp.json"
        qwen_gen_tmp = ws / ".qwen_gen_tmp.json"
        rex_orig_tmp = ws / ".rex_orig_tmp.json"
        rex_gen_tmp = ws / ".rex_gen_tmp.json"
        rex_prompt_path = ws / ".rex_prompt_tmp.json"
        t_det = datetime.now()
        try:
            ph.write_rex_prompt_json(rex_prompt_path, intent)

            cmd_q_o = [
                "python3",
                "skills/qwen-vlm-open-set-delection/scripts/run_detection.py",
                "--images",
                str(orig_image),
                "--label",
                target_label,
                "--base-url",
                QWEN_DETECTION_URL,
                "--out",
                str(qwen_orig_tmp),
            ]
            _append_run_log(run_stamp, f"command: {_mask_cmd(cmd_q_o)}")
            code, logs = _run_subprocess(cmd_q_o)
            if logs:
                _append_run_log(run_stamp, logs)
            if code != 0:
                emit({"type": "error", "message": logs or "qwen orig detection failed"})
                if emit_done:
                    emit({"type": "done", "ok": False})
                return None

            cmd_q_g = [
                "python3",
                "skills/qwen-vlm-open-set-delection/scripts/run_detection.py",
                "--images",
                str(gen_dir.resolve()),
                "--label",
                target_label,
                "--base-url",
                QWEN_DETECTION_URL,
                "--out",
                str(qwen_gen_tmp),
            ]
            _append_run_log(run_stamp, f"command: {_mask_cmd(cmd_q_g)}")
            code, logs = _run_subprocess(cmd_q_g)
            if logs:
                _append_run_log(run_stamp, logs)
            if code != 0:
                emit({"type": "error", "message": logs or "qwen gen detection failed"})
                if emit_done:
                    emit({"type": "done", "ok": False})
                return None

            with open(qwen_orig_tmp, "r", encoding="utf-8") as f:
                qwen_orig_result = json.load(f)
            with open(qwen_gen_tmp, "r", encoding="utf-8") as f:
                qwen_gen_result = json.load(f)

            qwen_results = qwen_orig_result.get("results", []) + qwen_gen_result.get("results", [])
            if len(qwen_results) != len(image_paths):
                emit(
                    {
                        "type": "error",
                        "message": f"Qwen 结果条数 ({len(qwen_results)}) 与图片数 ({len(image_paths)}) 不一致",
                    }
                )
                if emit_done:
                    emit({"type": "done", "ok": False})
                return None

            cmd_r_o = [
                "python3",
                "skills/rexomni-open-set-detection/scripts/run_detection.py",
                "--images",
                str(orig_image),
                "--prompt",
                str(rex_prompt_path),
                "--base-url",
                REX_BASE_URL,
                "--out",
                str(rex_orig_tmp),
            ]
            _append_run_log(run_stamp, f"command: {_mask_cmd(cmd_r_o)}")
            code, logs = _run_subprocess(cmd_r_o)
            if logs:
                _append_run_log(run_stamp, logs)
            if code != 0:
                emit({"type": "error", "message": logs or "rexomni orig detection failed"})
                if emit_done:
                    emit({"type": "done", "ok": False})
                return None

            cmd_r_g = [
                "python3",
                "skills/rexomni-open-set-detection/scripts/run_detection.py",
                "--images",
                str(gen_dir.resolve()),
                "--prompt",
                str(rex_prompt_path),
                "--base-url",
                REX_BASE_URL,
                "--out",
                str(rex_gen_tmp),
            ]
            _append_run_log(run_stamp, f"command: {_mask_cmd(cmd_r_g)}")
            code, logs = _run_subprocess(cmd_r_g)
            if logs:
                _append_run_log(run_stamp, logs)
            if code != 0:
                emit({"type": "error", "message": logs or "rexomni gen detection failed"})
                if emit_done:
                    emit({"type": "done", "ok": False})
                return None

            with open(rex_orig_tmp, "r", encoding="utf-8") as f:
                rex_orig_coco = json.load(f)
            with open(rex_gen_tmp, "r", encoding="utf-8") as f:
                rex_gen_coco = json.load(f)

            rex_orig_by_id = ph.coco_to_pred_bboxes_by_image_id(rex_orig_coco)
            rex_gen_by_id = ph.coco_to_pred_bboxes_by_image_id(rex_gen_coco)

            rex_boxes_by_idx: list[list[list[float]]] = []
            rex_boxes_by_idx.append(rex_orig_by_id.get(0, []))
            for idx in range(len(gen_files)):
                rex_boxes_by_idx.append(rex_gen_by_id.get(idx, []))
        finally:
            for p in (qwen_orig_tmp, qwen_gen_tmp, rex_orig_tmp, rex_gen_tmp, rex_prompt_path):
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass

        pred_records: list[dict] = []
        for i, item in enumerate(qwen_results):
            name = Path(item["image"]).name
            qwen_boxes = [
                x["bbox"] for x in item.get("bboxes", []) if isinstance(x, dict) and "bbox" in x
            ]
            rex_boxes = rex_boxes_by_idx[i] if i < len(rex_boxes_by_idx) else []
            pred_records.append(
                {
                    "image": name,
                    "source": "original" if i == 0 else "generated",
                    "image_idx": i,
                    "models": [
                        {"model": "qwen3-vl-8b", "pred_bboxes": qwen_boxes},
                        {"model": "rex-omni", "pred_bboxes": rex_boxes},
                    ],
                }
            )

        with open(prediction_json, "w", encoding="utf-8") as f:
            json.dump(pred_records, f, indent=2, ensure_ascii=False)

        viz_dir = ws / "annotated_images"
        ph.draw_detection_overlays(image_map, prediction_json, viz_dir)
        ann_urls = []
        for p in sorted(viz_dir.glob("*"), key=lambda x: x.name):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                ann_urls.append(url(f"annotated_images/{p.name}"))
        emit({"type": "annotated", "urls": ann_urls})
        emit({"type": "step_timing", "step": "anno", "elapsed_ms": int((datetime.now() - t_det).total_seconds() * 1000)})

        t_eval = datetime.now()
        cmd_rep = [
            "python3",
            "skills/eval-reports-generation/scripts/run_eval_report_generation.py",
            "--images",
            json.dumps(images_abs, ensure_ascii=False),
            "--prediction",
            str(prediction_json),
            "--task-text",
            text,
            "--target-label",
            target_label,
            "--api-base",
            DEMO_LLM_API_BASE,
            "--api-key",
            DEMO_LLM_API_KEY,
            "--out",
            str(evaluation_json),
        ]
        _append_run_log(run_stamp, f"command: {_mask_cmd(cmd_rep)}")
        code, logs = _run_subprocess(cmd_rep)
        if logs:
            _append_run_log(run_stamp, logs)
        if code != 0:
            emit({"type": "error", "message": logs or "report generation failed"})
            if emit_done:
                emit({"type": "done", "ok": False})
            return None
        with open(evaluation_json, "r", encoding="utf-8") as f:
            ev = json.load(f)
        emit({"type": "evaluation", "data": _evaluation_for_demo(ev)})
        elapsed_eval_ms = int((datetime.now() - t_eval).total_seconds() * 1000)
        emit({"type": "step_timing", "step": "eval", "elapsed_ms": elapsed_eval_ms})
        if emit_done:
            emit({"type": "done", "ok": True})
        ev_demo = _evaluation_for_demo(ev)
        return _build_observation(
            agent.ACTION_TARGET_DETECTION_EVALUATION,
            summary=_trim_text(
                str(ev_demo.get("overall_conclusion") or "")
                or f"已完成目标检测评测，共处理 {len(images_abs)} 张图片",
                600,
            ),
            intent_summary=_intent_summary(intent),
            num_images=len(images_abs),
            annotated_count=len(ann_urls),
            evaluation=ev_demo,
            elapsed_ms={
                "intent": int((datetime.now() - t_intent).total_seconds() * 1000),
                "gen": int((datetime.now() - t_gen).total_seconds() * 1000),
                "anno": int((datetime.now() - t_det).total_seconds() * 1000),
                "eval": elapsed_eval_ms,
            },
        )

    def _run_adela_cli_streaming(
        self,
        rawmodel_id: int,
        platform: str,
        eval_type: int,
        run_dir: Path,
        run_stamp: str,
        *,
        emit_done: bool = True,
    ) -> dict | None:
        emit = self._emit_stream
        t0 = datetime.now()
        cmd = [
            "python3",
            "-u",
            "skills/adela-cli/scripts/run_pipeline.py",
            "--rawmodel_id",
            str(rawmodel_id),
            "--platform",
            platform,
            "--eval_type",
            str(eval_type),
        ]
        cmd_disp = _mask_cmd(cmd)
        _terminal_log(run_stamp, f"adela-cli CMD (cwd={ROOT}): {cmd_disp}")
        _append_run_log(run_stamp, f"command: {cmd_disp}")

        event_trace: list[dict] = []
        final_payload: dict = {}
        log_lines: list[str] = []
        proc = None

        def compact(packet: dict) -> dict:
            result = packet.get("result")
            result_preview = ""
            if result is not None:
                try:
                    result_preview = _trim_text(json.dumps(result, ensure_ascii=False), 400000)
                except (TypeError, ValueError):
                    result_preview = _trim_text(str(result), 400000)
            rid = packet.get("rawmodel_id") if packet.get("rawmodel_id") is not None else rawmodel_id
            url = str(packet.get("model_url") or "").strip()
            if not url and rid:
                url = f"http://scg-adela.sensetime.com/dashboard#/mainpage/project/3/release?rid={rid}"
            cp: dict = {
                "event": str(packet.get("event") or "").strip(),
                "message": _trim_text(str(packet.get("message") or ""), 240),
                "status": _trim_text(str(packet.get("status") or ""), 64),
                "rawmodel_id": rid,
                "platform": _trim_text(str(packet.get("platform") or platform), 200),
                "model_url": _trim_text(url, 500),
                "dataset_id": packet.get("dataset_id"),
                "deployment_id": packet.get("deployment_id"),
                "benchmark_id": packet.get("benchmark_id"),
                "result_preview": result_preview,
            }
            if str(packet.get("event") or "").strip() == "deployment_list_result":
                cp["target_deployment_id"] = packet.get("target_deployment_id")
                cp["deployment_count"] = packet.get("deployment_count")
                cp["is_quant_platform"] = packet.get("is_quant_platform")
                cp["quant_dataset_step_needed"] = packet.get("quant_dataset_step_needed")
            err_detail = str(packet.get("error_detail") or "").strip()
            if err_detail:
                cp["error_detail"] = _trim_text(err_detail, 1200)
            return cp

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
                bufsize=1,
            )
            assert proc.stdout is not None
            while True:
                raw_line = proc.stdout.readline()
                if raw_line == "" and proc.poll() is not None:
                    break
                if raw_line == "":
                    continue
                line = raw_line.rstrip("\n")
                if line:
                    log_lines.append(line)
                stripped = line.strip()
                if stripped.startswith("[CMD]"):
                    _terminal_log(run_stamp, stripped)
                    continue
                if not stripped.startswith("data:"):
                    continue
                payload_line = stripped[5:].strip()
                if not payload_line or payload_line == "[DONE]":
                    continue
                try:
                    packet = json.loads(payload_line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(packet, dict):
                    continue
                compact_packet = compact(packet)
                event_trace.append(compact_packet)
                event_name = compact_packet["event"]
                if event_name == "adela_final_result":
                    final_payload = packet
                if event_name == "adela_pipeline_error":
                    err_detail = str(packet.get("error_detail") or "").strip()
                    platform_hint = _adela_unsupported_platform_user_message(
                        err_detail,
                        packet.get("message"),
                    )
                    if platform_hint:
                        compact_packet["message"] = platform_hint
                        emit_msg = platform_hint
                    else:
                        base_msg = compact_packet["message"] or "Adela CLI 执行失败"
                        emit_msg = base_msg
                        if err_detail:
                            emit_msg = f"{base_msg} 原因：{_trim_text(err_detail, 600)}"
                    emit({"type": "error", "message": emit_msg})
                elif event_name == "model_deployment_result":
                    platform_hint = _adela_unsupported_platform_user_message(
                        packet.get("result"),
                        packet.get("message"),
                        compact_packet.get("result_preview"),
                    )
                    if platform_hint:
                        compact_packet["message"] = platform_hint
                    emit({"type": "adela_event", **compact_packet})
                else:
                    emit({"type": "adela_event", **compact_packet})
            try:
                proc.wait(timeout=ADELA_CLI_TIMEOUT_SEC)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
                emit({"type": "error", "message": f"Adela CLI 执行超过 {ADELA_CLI_TIMEOUT_SEC} 秒"})
                if emit_done:
                    emit({"type": "done", "ok": False})
                return self._failure_observation(
                    agent.ACTION_ADELA_CLI_EVAL,
                    f"Adela CLI 执行超过 {ADELA_CLI_TIMEOUT_SEC} 秒",
                )
        except OSError as exc:
            emit({"type": "error", "message": f"启动 Adela CLI 失败: {exc}"})
            if emit_done:
                emit({"type": "done", "ok": False})
            return None
        finally:
            if log_lines:
                _append_run_log(run_stamp, "\n".join(log_lines))

        return_code = int(proc.returncode if proc is not None and proc.returncode is not None else 0)
        elapsed_ms = int((datetime.now() - t0).total_seconds() * 1000)
        emit({"type": "step_timing", "step": "adela_cli", "elapsed_ms": elapsed_ms})

        final_result = final_payload.get("result") if isinstance(final_payload.get("result"), dict) else {}
        final_message = str((final_result or {}).get("message") or "").strip()
        if not final_message:
            final_message = str(final_payload.get("message") or "").strip()

        success = return_code == 0
        summary = final_message or "Adela CLI 执行完成"
        if event_trace and event_trace[-1].get("event") == "adela_pipeline_error":
            success = False
            summary = (
                str(event_trace[-1].get("message") or "").strip()
                or "Adela CLI 执行失败，暂未获取到评测结果。"
            )
            err_detail = str(event_trace[-1].get("error_detail") or "").strip()
            if err_detail:
                summary = f"{summary} 详细原因：{_trim_text(err_detail, 1000)}"
            platform_hint = _adela_unsupported_platform_user_message(
                err_detail,
                summary,
                final_result.get("detail") if isinstance(final_result, dict) else "",
            )
            if platform_hint:
                summary = platform_hint
        # 缺少数据集等短路返回时，final_result 里可能仍带有历史 benchmark 的 result 嵌套；
        # 不得再按嵌套 info.status=SUCCESS 把 success 改回 True，否则最终答复会误带评测指标。
        _adela_terminal_messages = {
            "缺少量化数据集",
            "缺少评测数据集",
            "部署失败",
            "评测失败",
            "部署超时",
            "评测超时",
        }
        terminal_short = False
        if final_result:
            message = str(final_result.get("message") or "").strip()
            if message:
                summary = message
            if message in _adela_terminal_messages:
                success = False
                terminal_short = True
                if message == "部署失败":
                    pipeline_err = ""
                    if event_trace and event_trace[-1].get("event") == "adela_pipeline_error":
                        pipeline_err = str(event_trace[-1].get("error_detail") or "").strip()
                    platform_hint = _adela_unsupported_platform_user_message(
                        final_result.get("detail"),
                        pipeline_err,
                        summary,
                    )
                    if platform_hint:
                        summary = platform_hint
            if not terminal_short:
                result_info = final_result.get("result") if isinstance(final_result.get("result"), dict) else {}
                final_status = str(
                    ((result_info.get("info") or {}).get("status"))
                    if isinstance(result_info.get("info"), dict)
                    else final_result.get("status") or ""
                ).upper()
                if final_status == "FAILURE":
                    success = False
                elif final_status == "SUCCESS":
                    success = True

        if return_code != 0:
            success = False
            if not event_trace and log_lines:
                summary = _trim_text(log_lines[-1], 240) or summary
            emit({"type": "error", "message": summary or f"Adela CLI 执行失败（退出码 {return_code}）"})

        if emit_done:
            emit({"type": "done", "ok": success})
        err_detail_out = ""
        if event_trace and event_trace[-1].get("event") == "adela_pipeline_error":
            err_detail_out = str(event_trace[-1].get("error_detail") or "").strip()
        adela_result_out = final_result if isinstance(final_result, dict) else {}
        if terminal_short and isinstance(adela_result_out, dict):
            adela_result_out = {"message": str(summary or "").strip() or str(adela_result_out.get("message") or "")}
        return _build_observation(
            agent.ACTION_ADELA_CLI_EVAL,
            success=success,
            summary=summary or "Adela CLI 执行完成",
            rawmodel_id=rawmodel_id,
            platform=platform,
            eval_type=eval_type,
            event_trace=event_trace[-12:],
            adela_result=adela_result_out,
            elapsed_ms=elapsed_ms,
            error_detail=_trim_text(err_detail_out, 1200) if err_detail_out else "",
        )

    def _resolve_adela_model_reference_via_rag(
        self,
        *,
        rawmodel_id: int | None,
        model_name: str,
        run_dir: Path,
        run_stamp: str,
    ) -> dict:
        if rawmodel_id is not None and int(rawmodel_id) > 0:
            query = f"查询 Adela 模型 rawmodel_id {int(rawmodel_id)} 的模型信息"
        else:
            query = f"查询 Adela 模型 {model_name} 的 model_id 或 rawmodel_id"
        obs = self._run_rag_streaming(query, run_dir, run_stamp, emit_done=False)
        if not isinstance(obs, dict):
            return {
                "status": "not_found",
                "rawmodel_id": None,
                "matched_name": "",
                "candidate_model_names": [],
                "message": "当前 adela 平台上没有这个模型，如有需要，请到 monolith 平台上训练模型，并上传到 adela 平台。",
            }

        if agent.is_rag_miss(
            {
                "knowledge_base_fully_answered": obs.get(
                    "knowledge_base_fully_answered"
                )
            }
        ):
            return {
                "status": "not_found",
                "rawmodel_id": None,
                "matched_name": "",
                "candidate_model_names": [],
                "message": "当前 adela 平台上没有这个模型，如有需要，请到 monolith 平台上训练模型，并上传到 adela 平台。",
            }

        rag_json = run_dir / "rag_response.json"
        raw_last_packet = {}
        try:
            if rag_json.is_file():
                pack = json.loads(rag_json.read_text(encoding="utf-8"))
                if isinstance(pack, dict):
                    raw_last_packet = pack.get("raw_last_packet") if isinstance(pack.get("raw_last_packet"), dict) else {}
        except Exception:
            raw_last_packet = {}

        models_ordered = _collect_adela_models_from_rag_packet(raw_last_packet)
        if not models_ordered and agent.is_rag_miss(
            {
                "knowledge_base_fully_answered": obs.get(
                    "knowledge_base_fully_answered"
                )
            }
        ):
            return {
                "status": "not_found",
                "rawmodel_id": None,
                "matched_name": "",
                "candidate_model_names": [],
                "message": "当前 adela 平台上没有这个模型，如有需要，请到 monolith 平台上训练模型，并上传到 adela 平台。",
            }
        candidate_names = [
            str(m.get("model_name") or "").strip()
            for m in models_ordered
            if str(m.get("model_name") or "").strip()
        ]
        if not models_ordered:
            return {
                "status": "not_found",
                "rawmodel_id": None,
                "matched_name": "",
                "candidate_model_names": [],
                "message": "当前 adela 平台上没有这个模型，如有需要，请到 monolith 平台上训练模型，并上传到 adela 平台。",
            }

        if rawmodel_id is not None and int(rawmodel_id) > 0:
            id_matches = [m for m in models_ordered if int(m["rawmodel_id"]) == int(rawmodel_id)]
            if id_matches:
                chosen = max(id_matches, key=lambda m: int(m["rawmodel_id"]))
            else:
                chosen = None
        else:
            target = str(model_name or "").strip().lower()
            exact_matches = [
                m
                for m in models_ordered
                if str(m.get("model_name") or "").strip().lower() == target
            ]
            if exact_matches:
                chosen = max(exact_matches, key=lambda m: int(m["rawmodel_id"]))
            else:
                chosen = models_ordered[0]

        if not chosen:
            return {
                "status": "not_found",
                "rawmodel_id": None,
                "matched_name": "",
                "candidate_model_names": _normalize_adela_candidate_names(candidate_names),
                "message": "当前 adela 平台上没有这个模型，如有需要，请到 monolith 平台上训练模型，并上传到 adela 平台。",
            }
        return {
            "status": "resolved",
            "rawmodel_id": int(chosen["rawmodel_id"]),
            "matched_name": str(chosen.get("model_name") or "").strip(),
            "candidate_model_names": _normalize_adela_candidate_names(candidate_names),
            "message": "",
        }

    def _resolve_adela_model_id_via_rag(
        self,
        *,
        model_name: str,
        run_dir: Path,
        run_stamp: str,
    ) -> dict:
        return self._resolve_adela_model_reference_via_rag(
            rawmodel_id=None,
            model_name=model_name,
            run_dir=run_dir,
            run_stamp=run_stamp,
        )

    def _resolve_adela_direct_tool_before_execute(
        self,
        *,
        session: dict,
        pending_action: dict,
        forced_first_step: dict,
        qid: str,
        effective_text: str,
        run_dir: Path,
        run_stamp: str,
    ) -> dict:
        action_input = (
            dict(forced_first_step.get("action_input"))
            if isinstance(forced_first_step.get("action_input"), dict)
            else {}
        )
        explicit_id = clarification_state._extract_explicit_adela_rawmodel_id(
            action_input.get("rawmodel_id"),
            action_input.get("model_name"),
        )
        model_name = str(action_input.get("model_name") or "").strip()
        if explicit_id is None and not model_name:
            task_state = clarification_state.build_adela_clarification_task_state(
                task_state={
                    "candidate_tool": agent.TOOL_ADELA_CLI_EVAL,
                    "original_user_text": effective_text,
                },
                known_slots={
                    key: str(value).strip()
                    for key, value in action_input.items()
                    if key != "finish_after_tool" and str(value).strip()
                },
                missing_slots=["model_name"],
                tool_args=action_input,
            )
            return {
                "action": clarification_state.ACTION_STILL_PENDING,
                "pending_clarification": {
                    "status": "pending",
                    "source": "tool_precondition",
                    "query_id": qid,
                    "thread_id": str(pending_action.get("thread_id") or "").strip(),
                    "clarification_question": "要继续做 Adela 部署评测，请补充模型名称或 rawmodel_id。",
                    "task_state": task_state,
                },
            }

        old_emit = self._emit_stream
        try:
            self._emit_stream = lambda _obj: None
            resolved = self._resolve_adela_model_reference_via_rag(
                rawmodel_id=explicit_id,
                model_name=model_name,
                run_dir=run_dir,
                run_stamp=run_stamp,
            )
        finally:
            self._emit_stream = old_emit

        status = str(resolved.get("status") or "").strip() if isinstance(resolved, dict) else ""
        if status == "resolved":
            rawmodel_id = int(resolved.get("rawmodel_id") or explicit_id or 0)
            matched_name = str(resolved.get("matched_name") or model_name).strip()
            if rawmodel_id > 0:
                action_input["rawmodel_id"] = rawmodel_id
            if matched_name:
                action_input["model_name"] = matched_name
            updated_step = dict(forced_first_step)
            updated_step["action_input"] = action_input
            return {"action": "ready", "forced_first_step": updated_step}

        if status == "ambiguous":
            candidates = resolved.get("candidate_model_names") if isinstance(resolved.get("candidate_model_names"), list) else []
            question = (
                f"没有找到与“{model_name}”完全一致的模型名称。请确认你要的是下面哪个模型：\n"
                + "\n".join(f"{idx + 1}. {name}" for idx, name in enumerate(candidates))
                + "\n请直接回复完整模型名称。"
            ) if candidates else f"没有找到与“{model_name}”完全一致的模型名称，请提供更准确的模型名称。"
            task_state = clarification_state.build_adela_clarification_task_state(
                task_state={
                    "candidate_tool": agent.TOOL_ADELA_CLI_EVAL,
                    "original_user_text": str(pending_action.get("effective_text") or effective_text).strip(),
                },
                known_slots={
                    "model_name": model_name,
                    "platform": str(action_input.get("platform") or "").strip(),
                    "eval_type": str(action_input.get("eval_type") or "").strip(),
                },
                missing_slots=["model_name"],
                tool_args={
                    **action_input,
                    "model_name": model_name,
                    "rawmodel_id": "",
                },
                model_resolution_status="ambiguous",
                candidate_model_names=candidates,
            )
            return {
                "action": clarification_state.ACTION_STILL_PENDING,
                "pending_clarification": {
                    "status": "pending",
                    "source": "tool_precondition",
                    "query_id": qid,
                    "thread_id": str(pending_action.get("thread_id") or "").strip(),
                    "clarification_question": question,
                    "task_state": task_state,
                },
            }

        message = (
            str(resolved.get("message") or "").strip()
            if isinstance(resolved, dict)
            else ""
        ) or "当前 adela 平台上没有这个模型，如有需要，请到 monolith 平台上训练模型，并上传到 adela 平台。"
        return {"action": "final", "message": message}

    def do_POST(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/feedback":
            try:
                self._submit_playbook_feedback(self._read_json_body())
                return
            except ValueError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=415)
                return
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=500)
                return

        if parsed.path == "/session/delete":
            try:
                body = self._read_json_body()
                session_id = _normalize_session_id(str(body.get("session_id") or ""))
                if not session_id:
                    self._send_json({"ok": False, "error": "session_id is required"}, status=400)
                    return
                client_scope = _client_scope_from_request(self)
                session = _load_session_state(session_id)
                if str(session.get("client_scope") or "").strip() != client_scope:
                    self._send_json({"ok": False, "error": "session not found"}, status=404)
                    return
                _delete_session_files(session_id)
                self._send_json({"ok": True, "session_id": session_id})
                return
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=500)
                return

        if parsed.path == "/test/migration-advisor":
            try:
                self._handle_migration_advisor_test(self._read_json_body())
                return
            except ValueError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
                return
            except FileNotFoundError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=404)
                return
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=500)
                return

        if parsed.path != "/run":
            self._send_json({"error": "not found"}, status=404)
            return

        stream_started = False
        try:
            ctype, pdict = cgi.parse_header(self.headers.get("Content-Type", ""))
            if ctype != "multipart/form-data":
                self._send_json(
                    {"ok": False, "error": "Content-Type must be multipart/form-data"},
                    status=415,
                )
                return

            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                },
            )

            text = _form_get_first(form, "text", "")
            session_id = _normalize_session_id(_form_get_first(form, "session_id", ""))

            api_key = _load_api_key()
            api_base = DEFAULT_API_BASE

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(4)
            run_dir = RUNS_DIR / stamp
            run_dir.mkdir(parents=True, exist_ok=True)

            upload_fields = _upload_image_fields(form)
            upload_count = sum(1 for one in upload_fields if getattr(one, "filename", None))
            if upload_count > MAX_UPLOAD_IMAGES:
                self._send_json(
                    {
                        "ok": False,
                        "error": f"最多上传 {MAX_UPLOAD_IMAGES} 张图片，当前选择了 {upload_count} 张",
                    },
                    status=400,
                )
                return
            uploaded_image_paths = _save_uploaded_images(form, run_dir)
            uploaded_image_path = uploaded_image_paths[0] if uploaded_image_paths else ""
            image_for_route = uploaded_image_path

            with _session_guard(session_id):
                session = _load_session_state(session_id)
                session["client_ip"] = _client_ip_from_request(self)
                session["client_scope"] = _client_scope_from_request(self)
                pending = clarification_state.get_pending_clarification(
                    session,
                    normalize_thread_id=_normalize_thread_id,
                    normalize_action=agent.normalize_agent_action,
                )
                structured_text = _parse_json_object_text(text)
                pending_migration = (
                    session.get("pending_migration_advisor")
                    if isinstance(session.get("pending_migration_advisor"), dict)
                    else {}
                )
                migration_choice = ""
                if (
                    isinstance(structured_text, dict)
                    and str(structured_text.get("_structured_type") or "").strip()
                    == "migration_advisor_choice"
                ):
                    migration_choice = str(structured_text.get("choice") or "").strip()
                if not text and not uploaded_image_paths:
                    self._send_json({"ok": False, "error": "text is required"}, status=400)
                    return
                if not text and not pending:
                    self._send_json({"ok": False, "error": "text is required unless this is a clarification reply with image"}, status=400)
                    return
                rr: dict = {}
                routed_reason = ""
                pending_action = {"action": clarification_state.ACTION_NONE}
                if migration_choice and pending_migration:
                    mig_tid = str(pending_migration.get("thread_id") or "").strip()
                    threads_map = session.get("threads")
                    if isinstance(threads_map, dict) and mig_tid and mig_tid in threads_map:
                        session["active_thread_id"] = mig_tid
                        _hydrate_thread_into_session(session)
                    routed_reason = "resume_pending_migration_advisor"
                    session.pop("pending_migration_advisor", None)
                elif pending:
                    pending_tid = str(pending.get("thread_id") or "").strip()
                    threads_map = session.get("threads")
                    if isinstance(threads_map, dict) and pending_tid and pending_tid in threads_map:
                        session["active_thread_id"] = pending_tid
                        _hydrate_thread_into_session(session)
                    pending_action = clarification_state.handle_pending_reply(
                        pending=pending,
                        user_text=text,
                        image_path=image_for_route,
                        normalize_action=agent.normalize_agent_action,
                        decision_type_tool=agent.DECISION_TYPE_TOOL,
                        qwen_detection_action=agent.TOOL_QWEN_DETECTION,
                        rex_detection_action=agent.TOOL_REXOMNI_DETECTION,
                        pipeline_eval_action=agent.TOOL_PIPELINE_EVAL,
                        flux_action=agent.TOOL_FLUX_IMAGE_GENERATION,
                        adela_cli_action=agent.TOOL_ADELA_CLI_EVAL,
                    )
                    routed_reason = "resume_pending_clarification"
                    clarification_state.clear_pending_clarification(session)
                elif _thread_router_enabled():
                    rr = _call_thread_router_llm(user_text=text, session=session, run_dir=run_dir)
                    _apply_thread_router_decision(session, rr)
                    if rr.get("ok"):
                        routed_reason = str(rr.get("reason") or "").strip()
                    else:
                        routed_reason = str(rr.get("error") or "router_failed").strip()
                else:
                    # 未启用分流器时，保留当前活动线程，不接受外部 thread_id 覆盖。
                    _apply_thread_router_decision(session, {"ok": False, "error": "router_disabled"})
                captured_events: list[dict] = []

                old_emit = self._emit_stream

                def _emit_with_capture(obj: dict) -> None:
                    old_emit(obj)
                    if (
                        isinstance(obj, dict)
                        and obj.get("type") not in ("direct_reply", "final_answer")
                    ):
                        captured_events.append(dict(obj))

                self._emit_stream = _emit_with_capture

                if uploaded_image_paths:
                    resolved_paths = [
                        str(Path(p).resolve())
                        for p in uploaded_image_paths
                        if str(p).strip() and Path(p).is_file()
                    ]
                    if resolved_paths:
                        image_for_route = resolved_paths[0]
                        session["last_image_path"] = image_for_route
                        for p in resolved_paths:
                            _append_reference_image_path(session, p)

                effective_image_path = _resolve_effective_image_path(image_for_route, session)
                effective_image_paths = _collect_reference_image_paths(
                    session,
                    run_dir,
                    primary_path=effective_image_path,
                )

                ms.LedgerStore.migrate_schema(session)
                active_tid = str(session.get("active_thread_id") or "").strip()
                effective_text = str(text or "").strip()
                forced_first_step = None
                resumed_pending = bool(pending)
                pending_action_type = str(pending_action.get("action") or clarification_state.ACTION_NONE).strip()
                if not pending and active_tid:
                    latest_adela_args = _latest_adela_tool_args_for_thread(session, thread_id=active_tid)
                    if latest_adela_args:
                        patch = _infer_adela_patch_from_text(
                            text=effective_text,
                            known_tool_args=latest_adela_args,
                        )
                        if patch:
                            merged_args = dict(latest_adela_args)
                            merged_args.update({k: v for k, v in patch.items() if v is not None and str(v) != ""})
                            if "model_name" in patch and "rawmodel_id" not in patch:
                                merged_args["rawmodel_id"] = ""
                            if "rawmodel_id" in patch and "model_name" in merged_args and patch.get("rawmodel_id"):
                                merged_args["model_name"] = ""
                            latest_known_slots = {
                                k: str(v).strip()
                                for k, v in merged_args.items()
                                if k != "finish_after_tool" and str(v).strip()
                            }
                            missing_slots: list[str] = []
                            if not str(merged_args.get("rawmodel_id") or "").strip() and not str(merged_args.get("model_name") or "").strip():
                                missing_slots.append("model_name")
                            if not str(merged_args.get("platform") or "").strip():
                                missing_slots.append("platform")
                            if clarification_state.normalize_adela_eval_type_arg(merged_args.get("eval_type")) is None:
                                missing_slots.append("eval_type")
                            session["pending_clarification"] = clarification_state.normalize_pending_clarification(
                                {
                                    "status": "pending",
                                    "source": "tool_precondition",
                                    "query_id": "",
                                    "thread_id": active_tid,
                                    "clarification_question": "",
                                    "task_state": clarification_state.build_adela_clarification_task_state(
                                        task_state={
                                            "candidate_tool": agent.TOOL_ADELA_CLI_EVAL,
                                            "original_user_text": effective_text,
                                        },
                                        known_slots=latest_known_slots,
                                        missing_slots=missing_slots,
                                        tool_args=merged_args,
                                    ),
                                },
                                normalize_thread_id=_normalize_thread_id,
                                normalize_action=agent.normalize_agent_action,
                            )
                            pending = clarification_state.get_pending_clarification(
                                session,
                                normalize_thread_id=_normalize_thread_id,
                                normalize_action=agent.normalize_agent_action,
                            )
                            pending_action = clarification_state.handle_pending_reply(
                                pending=pending,
                                user_text=effective_text,
                                image_path=image_for_route,
                                normalize_action=agent.normalize_agent_action,
                                decision_type_tool=agent.DECISION_TYPE_TOOL,
                                qwen_detection_action=agent.TOOL_QWEN_DETECTION,
                                rex_detection_action=agent.TOOL_REXOMNI_DETECTION,
                                pipeline_eval_action=agent.TOOL_PIPELINE_EVAL,
                                flux_action=agent.TOOL_FLUX_IMAGE_GENERATION,
                                adela_cli_action=agent.TOOL_ADELA_CLI_EVAL,
                            )
                            resumed_pending = True
                            pending_action_type = str(pending_action.get("action") or clarification_state.ACTION_NONE).strip()
                if not pending and not resumed_pending:
                    latest_adela_args = _latest_adela_tool_args_for_thread(session, thread_id=active_tid) if active_tid else {}
                    if latest_adela_args:
                        latest_adela_args = _sanitize_adela_tool_args_for_user_query(
                            user_text=effective_text,
                            tool_args=latest_adela_args,
                        )
                pending_resolved_task_state = (
                    pending_action.get("resolved_task_state")
                    if isinstance(pending_action.get("resolved_task_state"), dict)
                    else {}
                )
                pending_resolved_tool = agent.normalize_agent_action(
                    str(pending_resolved_task_state.get("candidate_tool") or "").strip()
                )
                is_adela_resolved_direct_tool = (
                    pending_action_type == clarification_state.ACTION_DIRECT_TOOL
                    and pending_resolved_tool == agent.TOOL_ADELA_CLI_EVAL
                )
                if migration_choice and pending_migration:
                    qid = str(pending_migration.get("query_id") or "").strip()
                    if qid:
                        session["active_query_id"] = qid
                    original_user_text = str(
                        pending_migration.get("original_user_text") or ""
                    ).strip()
                    effective_text = original_user_text or effective_text
                    if qid:
                        qt = session.get("query_trajectories")
                        if isinstance(qt, list):
                            for tr in reversed(qt):
                                if not isinstance(tr, dict):
                                    continue
                                if str(tr.get("query_id") or "") != qid:
                                    continue
                                tr["query"] = effective_text
                                break
                elif is_adela_resolved_direct_tool and str(pending_action.get("query_id") or "").strip():
                    qid = str(pending_action.get("query_id") or "").strip()
                    tool_args = (
                        dict(pending_resolved_task_state.get("tool_args"))
                        if isinstance(pending_resolved_task_state.get("tool_args"), dict)
                        else {}
                    )
                    effective_text = str(pending_action.get("effective_text") or effective_text).strip() or effective_text
                    forced_first_step = {
                        "thought": "用户已补充 Adela 所需参数，参数解析完成，继续执行工具。",
                        "decision_type": agent.DECISION_TYPE_TOOL,
                        "action": pending_resolved_tool,
                        "action_input": tool_args,
                        "final_answer": "",
                    }
                    session["active_query_id"] = qid
                    qt = session.get("query_trajectories")
                    if isinstance(qt, list):
                        for tr in reversed(qt):
                            if not isinstance(tr, dict):
                                continue
                            if str(tr.get("query_id") or "") != qid:
                                continue
                            tr["query"] = effective_text
                            break
                elif pending_action_type in (clarification_state.ACTION_DIRECT_TOOL, clarification_state.ACTION_REPLAN) and str(pending_action.get("query_id") or "").strip():
                    qid = str(pending_action.get("query_id") or "").strip()
                    effective_text = str(pending_action.get("effective_text") or effective_text).strip() or effective_text
                    forced_first_step = (
                        dict(pending_action.get("forced_first_step"))
                        if isinstance(pending_action.get("forced_first_step"), dict)
                        else None
                    )
                    session["active_query_id"] = qid
                    qt = session.get("query_trajectories")
                    if isinstance(qt, list):
                        for tr in reversed(qt):
                            if not isinstance(tr, dict):
                                continue
                            if str(tr.get("query_id") or "") != qid:
                                continue
                            tr["query"] = effective_text
                            break
                elif pending_action_type == clarification_state.ACTION_RESOLVED and str(pending_action.get("query_id") or "").strip():
                    qid = str(pending_action.get("query_id") or "").strip()
                    resolved_task_state = (
                        pending_action.get("resolved_task_state")
                        if isinstance(pending_action.get("resolved_task_state"), dict)
                        else {}
                    )
                    candidate_tool = agent.normalize_agent_action(str(resolved_task_state.get("candidate_tool") or "").strip())
                    tool_args = dict(resolved_task_state.get("tool_args")) if isinstance(resolved_task_state.get("tool_args"), dict) else {}
                    effective_text = str(pending_action.get("effective_text") or effective_text).strip() or effective_text
                    forced_first_step = {
                        "thought": "用户已补充 Adela 所需参数，参数解析完成，继续执行工具。",
                        "decision_type": agent.DECISION_TYPE_TOOL,
                        "action": candidate_tool,
                        "action_input": tool_args,
                        "final_answer": "",
                    }
                    session["active_query_id"] = qid
                    qt = session.get("query_trajectories")
                    if isinstance(qt, list):
                        for tr in reversed(qt):
                            if not isinstance(tr, dict):
                                continue
                            if str(tr.get("query_id") or "") != qid:
                                continue
                            tr["query"] = effective_text
                            break
                elif pending_action_type == clarification_state.ACTION_STILL_PENDING and str(pending_action.get("query_id") or "").strip():
                    qid = str(pending_action.get("query_id") or "").strip()
                    next_pending = pending_action.get("pending_clarification") if isinstance(pending_action.get("pending_clarification"), dict) else {}
                    session["pending_clarification"] = clarification_state.normalize_pending_clarification(
                        next_pending,
                        normalize_thread_id=_normalize_thread_id,
                        normalize_action=agent.normalize_agent_action,
                    )
                    qt = session.get("query_trajectories")
                    if isinstance(qt, list):
                        for tr in reversed(qt):
                            if not isinstance(tr, dict):
                                continue
                            if str(tr.get("query_id") or "") != qid:
                                continue
                            original_user_text = str(((session.get("pending_clarification") or {}).get("task_state") or {}).get("original_user_text") or "").strip()
                            if original_user_text:
                                tr["query"] = original_user_text
                            break
                elif pending_action_type == clarification_state.ACTION_CANCEL and str(pending_action.get("query_id") or "").strip():
                    qid = str(pending_action.get("query_id") or "").strip()
                    session["active_query_id"] = qid
                    qt = session.get("query_trajectories")
                    if isinstance(qt, list):
                        for tr in reversed(qt):
                            if not isinstance(tr, dict):
                                continue
                            if str(tr.get("query_id") or "") != qid:
                                continue
                            original_user_text = str((tr.get("query") or "")).strip()
                            if original_user_text:
                                effective_text = original_user_text
                            break
                elif not pending_action_type and latest_adela_args:
                    patch = _infer_adela_patch_from_text(
                        text=effective_text,
                        known_tool_args=latest_adela_args,
                    )
                    if patch:
                        merged_args = dict(latest_adela_args)
                        merged_args.update({k: v for k, v in patch.items() if v is not None and str(v) != ""})
                        if "model_name" in patch and "rawmodel_id" not in patch:
                            merged_args["rawmodel_id"] = ""
                        if "rawmodel_id" in patch and patch.get("rawmodel_id"):
                            merged_args["model_name"] = ""
                        effective_text = str(text or "").strip()
                        qid = ms.QueryTrajectoryStore.new_query_id()
                        ms.QueryTrajectoryStore.start_query(
                            session,
                            query_id=qid,
                            thread_id=active_tid or None,
                            session_id=session_id,
                        )
                        qt = session.get("query_trajectories")
                        if isinstance(qt, list):
                            for tr in reversed(qt):
                                if not isinstance(tr, dict):
                                    continue
                                if str(tr.get("query_id") or "") != qid:
                                    continue
                                tr["query"] = effective_text
                                break
                        forced_first_step = {
                            "thought": "用户在当前线程中修改了 Adela 参数，基于最近一次任务参数进行覆盖后重新执行。",
                            "decision_type": agent.DECISION_TYPE_TOOL,
                            "action": agent.TOOL_ADELA_CLI_EVAL,
                            "action_input": merged_args,
                            "final_answer": "",
                        }
                        session["active_query_id"] = qid
                else:
                    qid = ms.QueryTrajectoryStore.new_query_id()
                    ms.QueryTrajectoryStore.start_query(
                        session,
                        query_id=qid,
                        thread_id=active_tid or None,
                        session_id=session_id,
                    )
                    qt = session.get("query_trajectories")
                    if isinstance(qt, list):
                        for tr in reversed(qt):
                            if not isinstance(tr, dict):
                                continue
                            if str(tr.get("query_id") or "") != qid:
                                continue
                            tr["query"] = effective_text
                            break
                ms.LedgerStore.append_event(
                    session,
                    event_type="USER_INPUT",
                    observation="user_text",
                    payload={
                        "text": text,
                        "query_id": qid,
                        "effective_text": effective_text,
                        "is_clarification_reply": resumed_pending,
                    },
                    external_ref=str(run_dir.resolve()),
                    thread_id=active_tid or None,
                )
                ms.LedgerStore.sync_ledger_cursor(session)

                history = session.get("summary_history", [])
                if not isinstance(history, list):
                    history = []

                self._start_ndjson()
                stream_started = True
                _terminal_log(stamp, f"request start: session_id={session_id}")
                if resumed_pending or _thread_router_enabled():
                    self._emit_stream(
                        {
                            "type": "thread_routed",
                            "thread_id": str(session.get("active_thread_id") or ""),
                            "reason": _trim_text(routed_reason, 240),
                            "router_ok": bool(rr.get("ok")),
                            "action": str(rr.get("action") or ""),
                            "target_thread_id": str(rr.get("target_thread_id") or ""),
                        }
                    )
                self._emit_stream(
                    {
                        "type": "session",
                        "session_id": session_id,
                        "thread_id": str(session.get("active_thread_id") or ""),
                    }
                )

                if is_adela_resolved_direct_tool and forced_first_step:
                    gate = self._resolve_adela_direct_tool_before_execute(
                        session=session,
                        pending_action=pending_action,
                        forced_first_step=forced_first_step,
                        qid=str(pending_action.get("query_id") or "").strip(),
                        effective_text=effective_text,
                        run_dir=run_dir,
                        run_stamp=stamp,
                    )
                    gate_action = str(gate.get("action") or "").strip()
                    if gate_action == "ready":
                        forced_first_step = (
                            dict(gate.get("forced_first_step"))
                            if isinstance(gate.get("forced_first_step"), dict)
                            else forced_first_step
                        )
                    elif gate_action == clarification_state.ACTION_STILL_PENDING:
                        session["pending_clarification"] = clarification_state.normalize_pending_clarification(
                            gate.get("pending_clarification") if isinstance(gate.get("pending_clarification"), dict) else {},
                            normalize_thread_id=_normalize_thread_id,
                            normalize_action=agent.normalize_agent_action,
                        )
                        pending_action_type = clarification_state.ACTION_STILL_PENDING
                        forced_first_step = None
                    elif gate_action == "final":
                        pending_action_type = "resolved_not_found_final"
                        forced_first_step = None
                        message = str(gate.get("message") or "").strip() or "当前 adela 平台上没有这个模型。"
                        self._emit_stream({"type": "final_answer", "text": message})
                        loop_result = {
                            "final_answer": message,
                            "assistant_text": message,
                            "assistant_event_type": "final_answer",
                            "query_completed": True,
                        }

                if pending_action_type == clarification_state.ACTION_STILL_PENDING:
                    pending_now = clarification_state.get_pending_clarification(
                        session,
                        normalize_thread_id=_normalize_thread_id,
                        normalize_action=agent.normalize_agent_action,
                    )
                    clarification_question = str(pending_now.get("clarification_question") or "").strip() or "当前还缺少继续执行所需的信息，请补充。"
                    self._emit_stream(
                        {
                            "type": "meta",
                            "flow": "clarify",
                            "decision": {
                                "action": "clarify",
                                "reason": "工具前置条件仍未满足，继续等待用户补齐缺失参数。",
                                "direct_reply": "",
                            },
                            "run_stamp": stamp,
                            "step_index": 1,
                        }
                    )
                    self._emit_stream(
                        {
                            "type": "clarification",
                            "text": clarification_question,
                            "source": str(pending_now.get("source") or ""),
                            "missing_slots": ((pending_now.get("task_state") or {}).get("missing_slots")) if isinstance((pending_now.get("task_state") or {}).get("missing_slots"), list) else [],
                            "task_state": pending_now.get("task_state") if isinstance(pending_now.get("task_state"), dict) else {},
                        }
                    )
                    loop_result = {
                        "final_answer": clarification_question,
                        "assistant_text": clarification_question,
                        "assistant_event_type": "clarification",
                        "query_completed": False,
                    }
                elif pending_action_type == clarification_state.ACTION_CANCEL:
                    self._emit_stream({"type": "final_answer", "text": "已取消当前任务"})
                    loop_result = {
                        "final_answer": "已取消当前任务",
                        "assistant_text": "已取消当前任务",
                        "assistant_event_type": "final_answer",
                        "query_completed": True,
                    }
                elif migration_choice:
                    original_user_text = str(
                        pending_migration.get("original_user_text") or effective_text
                    ).strip()
                    rag_trace = (
                        pending_migration.get("rag_round_trace")
                        if isinstance(pending_migration.get("rag_round_trace"), list)
                        else []
                    )
                    if migration_choice == "start":
                        obs = self._run_migration_advisor_streaming(
                            text=original_user_text,
                            rag_trace=rag_trace,
                            run_dir=run_dir,
                            run_stamp=stamp,
                            session_id=str(session.get("session_id") or session_id or ""),
                            image_path=effective_image_path,
                            image_paths=effective_image_paths,
                            session=session,
                            emit_done=False,
                        )
                        assistant_text = str(obs.get("summary") or "").strip()
                        loop_result = {
                            "final_answer": assistant_text,
                            "assistant_text": assistant_text,
                            "assistant_event_type": "final_answer",
                            "query_completed": True,
                        }
                    elif migration_choice == "fallback_answer":
                        all_chunks: list[dict] = []
                        for item in rag_trace:
                            if not isinstance(item, dict):
                                continue
                            chunks = item.get("retrieved_chunks")
                            if isinstance(chunks, list):
                                all_chunks.extend([x for x in chunks if isinstance(x, dict)])
                        streamed_parts: list[str] = []
                        self._emit_stream(
                            {
                                "type": "meta",
                                "flow": "direct_answer",
                                "decision": {
                                    "action": agent.TOOL_ANSWERER,
                                    "reason": "用户选择不进入迁移顾问，改为基于有限证据直接回答。",
                                    "direct_reply": "",
                                },
                                "run_stamp": stamp,
                                "step_index": 1,
                            }
                        )
                        final_text = agent.generate_final_answer_with_fallback(
                            answerer_input={
                                "user_query": original_user_text,
                                "evidence": {"retrieved_chunks": all_chunks},
                            },
                            mode="rag_evidence",
                            debug_meta={
                                "session_id": str(session.get("session_id") or ""),
                                "run_stamp": stamp,
                                "run_dir": str(run_dir),
                                "step_index": 1,
                                "stage": "answerer_migration_fallback_request",
                                "stage_response": "answerer_migration_fallback_response",
                            },
                            emit_chunk=lambda piece: (
                                streamed_parts.append(piece),
                                self._emit_stream({"type": "final_answer", "text": piece}),
                            ),
                        )
                        if not streamed_parts:
                            self._emit_stream({"type": "final_answer", "text": final_text})
                        loop_result = {
                            "final_answer": final_text,
                            "assistant_text": final_text,
                            "assistant_event_type": "final_answer",
                            "query_completed": True,
                        }
                    else:
                        message = "已取消迁移顾问分析。"
                        self._emit_stream({"type": "final_answer", "text": message})
                        loop_result = {
                            "final_answer": message,
                            "assistant_text": message,
                            "assistant_event_type": "final_answer",
                            "query_completed": True,
                        }
                elif pending_action_type != "resolved_not_found_final":
                    if forced_first_step is None:
                        hinted = _heuristic_forced_adela_cli_step(effective_text)
                        if hinted:
                            forced_first_step = hinted
                    loop_result = self._run_agent_loop(
                        text=effective_text,
                        image_path=effective_image_path,
                        api_key=api_key,
                        api_base=api_base,
                        run_dir=run_dir,
                        run_stamp=stamp,
                        session=session,
                        forced_first_step=forced_first_step,
                    )

                final_answer = ""
                assistant_text = ""
                assistant_event_type = "final_answer"
                query_completed = True
                if isinstance(locals().get("loop_result"), dict):
                    final_answer = str(loop_result.get("final_answer") or "").strip()
                    assistant_text = str(loop_result.get("assistant_text") or final_answer).strip()
                    assistant_event_type = str(loop_result.get("assistant_event_type") or "final_answer").strip() or "final_answer"
                    query_completed = bool(loop_result.get("query_completed", True))
                else:
                    assistant_text = final_answer
                if assistant_event_type == "final_answer" and final_answer:
                    captured_events.append({"type": "final_answer", "text": final_answer})

                out_ev = ms.LedgerStore.append_event(
                    session,
                    event_type="ASSISTANT_OUTPUT",
                    observation=assistant_event_type
                    if assistant_event_type in ("clarification", "migration_advisor_offer")
                    else "final_answer",
                    payload=(
                        {
                            "clarification_question": assistant_text,
                            "pending_clarification": clarification_state.get_pending_clarification(
                                session,
                                normalize_thread_id=_normalize_thread_id,
                                normalize_action=agent.normalize_agent_action,
                            ),
                        }
                        if assistant_event_type == "clarification"
                        else (
                            {
                                "migration_advisor_offer": assistant_text,
                                "pending_migration_advisor": session.get("pending_migration_advisor")
                                if isinstance(session.get("pending_migration_advisor"), dict)
                                else {},
                            }
                            if assistant_event_type == "migration_advisor_offer"
                            else {"final_answer": assistant_text}
                        )
                    ),
                    external_ref=str(run_dir.resolve()),
                    thread_id=str(session.get("active_thread_id") or "").strip() or None,
                )
                ms.QueryTrajectoryStore.append_step(
                    session,
                    action="clarify"
                    if assistant_event_type == "clarification"
                    else (
                        "migration_advisor_offer"
                        if assistant_event_type == "migration_advisor_offer"
                        else "final_answer"
                    ),
                    observation_event_id=str(out_ev.get("event_id") or ""),
                )
                ms.LedgerStore.sync_ledger_cursor(session)

                if effective_image_path:
                    session["last_image_path"] = effective_image_path

                answer_history_limit: int | None = 500
                if "### 迁移顾问报告" in assistant_text:
                    answer_history_limit = None
                history.append(
                    {
                        "run_stamp": stamp,
                        "query": _trim_text(effective_text, 500),
                        "final_answer": _trim_text(assistant_text, answer_history_limit),
                        "pending_clarification": not query_completed,
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                    }
                )
                session["summary_history"] = history
                turns = session.get("chat_turns")
                if not isinstance(turns, list):
                    turns = []
                turns.append(
                    {
                        "run_stamp": stamp,
                        "user_text": text,
                        "events": captured_events
                        + (
                            [
                                {
                                    "type": "migration_advisor_offer",
                                    "text": assistant_text,
                                    "options": [
                                        {"id": "start", "label": "生成迁移顾问报告"},
                                        {"id": "fallback_answer", "label": "直接回答"},
                                        {"id": "cancel", "label": "取消"},
                                    ],
                                }
                            ]
                            if assistant_event_type == "migration_advisor_offer"
                            else []
                        ),
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                    }
                )
                session["chat_turns"] = turns[-max(20, SESSION_SUMMARY_LIMIT):]
                summary_qid = str(qid or session.get("active_query_id") or "").strip()
                _save_session_state(session)
                if active_tid:
                    _schedule_thread_topic_refresh(session_id=session_id, thread_id=active_tid, run_dir=run_dir)
                if active_tid and summary_qid and query_completed:
                    _schedule_query_trajectory_summary(
                        session_id=session_id, thread_id=active_tid, query_id=summary_qid, run_dir=run_dir
                    )
                self._emit_stream({"type": "done", "ok": True})
                _terminal_log(stamp, "request done")
                return

        except Exception as e:
            _terminal_log("unknown", f"request exception: {e}")
            if stream_started:
                try:
                    self._emit_stream({"type": "error", "message": str(e)})
                    self._emit_stream({"type": "done", "ok": False})
                except Exception:
                    pass
            else:
                try:
                    self._start_ndjson()
                    self._emit_stream({"type": "error", "message": str(e)})
                    self._emit_stream({"type": "done", "ok": False})
                except Exception:
                    self._send_json({"ok": False, "error": str(e)}, status=500)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local visual demo for skill-agent routing")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind address: 0.0.0.0 = all interfaces (LAN); 127.0.0.1 = local only",
    )
    parser.add_argument("--port", type=int, default=18080, help="Port to bind")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"Demo listening on {args.host}:{args.port}")
    if args.host in ("0.0.0.0", "::"):
        lan = _guess_lan_ip()
        if lan:
            print(f"  From this machine: http://127.0.0.1:{args.port}")
            print(f"  From other PCs (try): http://{lan}:{args.port}")
        else:
            print(f"  From this machine: http://127.0.0.1:{args.port}")
            print("  From other PCs: use this server's LAN IP, e.g. http://<server-ip>:18080")
    else:
        print(
            f"  当前仅绑定 {args.host}：外网卡 IP（如 10.x）将无法访问，需改为 --host 0.0.0.0"
        )
        print(f"  Open: http://{args.host}:{args.port}")
    print(f"Repo root: {ROOT}")
    rag_mode = gbrain_rag.rag_api_mode()
    if rag_mode == gbrain_rag.RAG_API_MODE_UNIFIED:
        print(
            f"RAG API: unified (query={gbrain_rag.unified_query_url()}, "
            f"retrieve={gbrain_rag.unified_retrieve_url()}, "
            f"include_full_documents={gbrain_rag.include_full_documents_enabled()})"
        )
    else:
        print(
            f"RAG API: playbook (query={gbrain_rag.playbook_query_url()}, "
            f"retrieve={gbrain_rag.playbook_retrieve_url()})"
        )
    print("  Config: edit DEMO_RAG_* in demo_server.py, or export DEMO_RAG_API_MODE=unified")
    print("  Active:", json.dumps(gbrain_rag.get_rag_config(), ensure_ascii=False))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("Demo stopped.")


if __name__ == "__main__":
    main()
