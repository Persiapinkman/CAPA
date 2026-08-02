"""H20 (Hopper) inference backend built on top of vLLM.

Design goals
------------
- Serve the exact model contract used by the existing planner evaluators:
  Qwen3.5 chatml with ``enable_thinking=false``. vLLM applies the tokenizer
  chat template unchanged, so the same prompts produced by
  ``training/planner_grpo_seed_v1`` remain valid.
- Emit an OpenAI-compatible endpoint. ``run_repeated_planner_grpo_eval.py``
  already talks to arbitrary ``--api-base``.
- bf16 by default (H20 is Hopper; V100-only fp16 is no longer required).
- 4B fits on 1 GPU; the 35B-A3B MoE runs with tensor-parallel-size=4 across the
  four H20 devices.

This module only *builds* commands and *probes* the endpoint. Actually
launching vLLM is left to the shell scripts under ``scripts/reproduce/`` so
that logs, PIDs and shutdown are transparent to the operator.
"""

from __future__ import annotations

import json
import shlex
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_MODELS_ROOT = Path("/raid/zkq/models")
QWEN35_4B_ALIAS = "Qwen3.5-4B"
QWEN35_35B_A3B_ALIAS = "Qwen3.5-35B-A3B"

# Public HuggingFace repos used as defaults when the internal weights are
# unavailable. Override with ``QWEN35_4B_REPO`` / ``QWEN35_35B_REPO``.
DEFAULT_HF_REPOS: dict[str, str] = {
    QWEN35_4B_ALIAS: "Qwen/Qwen3-4B",
    QWEN35_35B_A3B_ALIAS: "Qwen/Qwen3-30B-A3B",
}


@dataclass(frozen=True)
class ServeSpec:
    """Concrete parameters for a single vLLM serve instance on H20."""

    model_alias: str
    model_path: Path
    served_model_name: str
    host: str = "127.0.0.1"
    port: int = 8000
    dtype: str = "bfloat16"
    tensor_parallel_size: int = 1
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.90
    enforce_eager: bool = False
    trust_remote_code: bool = True
    # vLLM enables MoE-specific optimizations automatically for the models we
    # target; we keep this as an opt-in override for future kernels.
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"


def resolve_model_paths(models_root: Path | None = None) -> dict[str, Path]:
    """Return the on-disk directories expected by the existing scripts.

    The training / evaluation configs hard-code
    ``/raid/zkq/models/Qwen3.5-4B``. We keep that convention so that
    ``run_qwen35_4b_planner_v6_sft.sh`` and friends do not need to change.
    """

    root = Path(models_root) if models_root else DEFAULT_MODELS_ROOT
    return {
        QWEN35_4B_ALIAS: root / QWEN35_4B_ALIAS,
        QWEN35_35B_A3B_ALIAS: root / QWEN35_35B_A3B_ALIAS,
    }


def build_vllm_command(spec: ServeSpec) -> list[str]:
    """Construct the argv for ``python -m vllm.entrypoints.openai.api_server``.

    We deliberately do *not* set ``--chat-template``: vLLM reads
    ``chat_template.jinja`` and ``tokenizer_config.json`` from the model
    directory, which is exactly the contract audited in
    ``sft_data_planner_retry_migrate_v6_qwen35_nothinking/metadata.json``.
    """

    if not spec.model_path.exists():
        raise FileNotFoundError(f"model path not found: {spec.model_path}")

    cmd: list[str] = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(spec.model_path),
        "--served-model-name",
        spec.served_model_name,
        "--host",
        spec.host,
        "--port",
        str(spec.port),
        "--dtype",
        spec.dtype,
        "--tensor-parallel-size",
        str(spec.tensor_parallel_size),
        "--max-model-len",
        str(spec.max_model_len),
        "--gpu-memory-utilization",
        f"{spec.gpu_memory_utilization:.3f}",
    ]
    if spec.trust_remote_code:
        cmd.append("--trust-remote-code")
    if spec.enforce_eager:
        cmd.append("--enforce-eager")
    cmd.extend(spec.extra_args)
    return cmd


def probe_endpoint(spec: ServeSpec, timeout_seconds: int = 900, interval: float = 5.0) -> dict:
    """Block until the vLLM server publishes ``spec.served_model_name``.

    Returns the parsed ``/v1/models`` payload. Raises ``TimeoutError`` if the
    endpoint does not become ready within ``timeout_seconds``.
    """

    deadline = time.monotonic() + timeout_seconds
    last_error: str = "no attempt"
    tcp_host, tcp_port = spec.host, spec.port

    while time.monotonic() < deadline:
        # Fast pre-check: is the port even listening?
        try:
            with socket.create_connection((tcp_host, tcp_port), timeout=2.0):
                pass
        except OSError as exc:
            last_error = f"tcp closed: {exc}"
            time.sleep(interval)
            continue

        try:
            req = urllib.request.Request(f"{spec.base_url}/models", method="GET")
            with urllib.request.urlopen(req, timeout=5.0) as resp:  # noqa: S310
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
            last_error = f"models query failed: {exc}"
            time.sleep(interval)
            continue

        served = {item.get("id") for item in payload.get("data", [])}
        if spec.served_model_name in served:
            return payload
        last_error = f"served ids={sorted(served)} does not include {spec.served_model_name}"
        time.sleep(interval)

    raise TimeoutError(
        f"vLLM endpoint {spec.base_url} did not become ready in {timeout_seconds}s: {last_error}"
    )


def render_command(cmd: list[str]) -> str:
    """Shell-safe rendering of the vLLM command (used only for logging)."""

    return " ".join(shlex.quote(part) for part in cmd)


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Print the vLLM serve command for an alias.")
    parser.add_argument("alias", choices=list(DEFAULT_HF_REPOS.keys()))
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--served-model-name", default=None)
    args = parser.parse_args()

    tp = args.tensor_parallel_size
    if tp is None:
        tp = 4 if args.alias == QWEN35_35B_A3B_ALIAS else 1

    paths = resolve_model_paths(args.models_root)
    spec = ServeSpec(
        model_alias=args.alias,
        model_path=paths[args.alias],
        served_model_name=args.served_model_name or args.alias,
        host=args.host,
        port=args.port,
        dtype=args.dtype,
        tensor_parallel_size=tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print(render_command(build_vllm_command(spec)))


if __name__ == "__main__":  # pragma: no cover - manual entrypoint
    _cli()
