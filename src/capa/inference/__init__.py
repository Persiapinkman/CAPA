"""CAPA inference backends.

The H20 backend uses vLLM in bf16 with Hopper-optimized FlashAttention 2 and
serves an OpenAI-compatible endpoint that the existing planner evaluators
(`training/planner_grpo_seed_v1/scripts/run_repeated_planner_grpo_eval.py`) can
consume unchanged.
"""

from .h20_backend import (
    ServeSpec,
    build_vllm_command,
    probe_endpoint,
    resolve_model_paths,
)

__all__ = [
    "ServeSpec",
    "build_vllm_command",
    "probe_endpoint",
    "resolve_model_paths",
]
