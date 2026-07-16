#!/usr/bin/env python3
"""Qwen3.5-4B DDP LoRA GRPO entrypoint for frozen CAPA target-step data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import accelerate
import datasets
import peft
import torch
import transformers
import trl
from accelerate.utils import GradScalerKwargs, gather_object
from datasets import Dataset
from peft import LoraConfig, PeftModel
from peft.tuners.tuners_utils import BaseTunerLayer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from trl import GRPOConfig, GRPOTrainer


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.planner_grpo_seed_v1.scripts import reward_planner_grpo as rewardlib  # noqa: E402
from training.planner_grpo_seed_v1.scripts.train_planner_grpo import (  # noqa: E402
    completion_text,
    first_json_text,
    parse_completion,
    score_step_completion,
)
from training.planner_grpo_seed_v1.scripts.train_planner_grpo_trl import (  # noqa: E402
    make_format_reward,
)
from training.wandb_observability import (  # noqa: E402
    DEFAULT_WANDB_GROUP,
    DEFAULT_WANDB_PROJECT,
    advantage_statistics,
    configure_wandb_environment,
    metric_contract,
    mirror_policy_entropy_metric,
    parse_report_to,
    weighted_reward_statistics,
)


DEFAULT_MODEL = Path("/raid/zkq/models/Qwen3.5-4B")
DEFAULT_STEP_DATA = (
    ROOT
    / "training/planner_grpo_seed_v1/step_data/"
    "planner_multistep_grpo_value_v5_train_v1_qwen35_4b_nothinking_step2.jsonl"
)
DEFAULT_OUTPUT = ROOT / "outputs/planner-grpo-qwen35-4b-v5-train-v1-seed42"
EXPECTED_DATASET_ID = "planner_multistep_grpo_value_v5_train_v1"
EXPECTED_MODEL_CLASS = "Qwen3_5ForCausalLM"
EXPECTED_EOS_ID = 248046
EXPECTED_PAD_ID = 248044
EXPECTED_LORA_MODULES = 152
EXPECTED_TRAINABLE_PARAMS = 14_376_960
NONTHINKING_SUFFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
]
GIB = 1024**3


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def parse_step_indices(value: str) -> tuple[int, ...]:
    try:
        indices = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("step indices must be comma-separated integers") from exc
    if not indices or any(index <= 0 for index in indices):
        raise argparse.ArgumentTypeError("step indices must be positive integers")
    return indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["dry-run", "g4", "train"], default="dry-run")
    parser.add_argument("--confirm-train", action="store_true")
    parser.add_argument("--model-name-or-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help="Optional SFT LoRA adapter to continue optimizing in place.",
    )
    parser.add_argument("--step-data", type=Path, default=DEFAULT_STEP_DATA)
    parser.add_argument(
        "--step-data-manifest",
        type=Path,
        default=None,
        help="Optional manifest; defaults to STEP_DATA with .manifest.json suffix.",
    )
    parser.add_argument("--expected-dataset-id", default=EXPECTED_DATASET_ID)
    parser.add_argument("--expected-rows", type=int, default=480)
    parser.add_argument(
        "--allowed-step-indices",
        type=parse_step_indices,
        default=(2,),
        help="Comma-separated frozen target steps accepted by the trainer (default: 2).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-prompt-tokens", type=int, default=4608)
    parser.add_argument("--max-completion-length", type=int, default=320)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--expected-world-size", type=int, default=8)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--task-reward-weight", type=float, default=0.95)
    parser.add_argument("--format-reward-weight", type=float, default=0.05)
    parser.add_argument("--max-allocated-gib", type=float, default=28.0)
    parser.add_argument("--minimum-free-gib", type=float, default=2.0)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument(
        "--report-to",
        default="auto",
        help="HF reporting backends. 'auto' enables W&B for train and disables it for dry-run/G4.",
    )
    parser.add_argument("--run-name", default="")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", DEFAULT_WANDB_PROJECT))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument("--wandb-group", default=os.environ.get("WANDB_RUN_GROUP", DEFAULT_WANDB_GROUP))
    parser.add_argument("--wandb-tags", default="stage2,grpo,planner-v6")
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _manifest_step_hash(manifest: dict[str, Any], path: Path) -> str:
    direct = str((manifest.get("sha256") or {}).get("step_data") or "")
    if direct:
        return direct
    files = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
    hashes = manifest.get("sha256") if isinstance(manifest.get("sha256"), dict) else {}
    try:
        relative = str(path.resolve().relative_to(ROOT))
    except ValueError:
        relative = str(path.resolve())
    matches = [
        str(hashes.get(name) or "")
        for name, value in files.items()
        if str(value) in {relative, str(path.resolve())}
    ]
    if len(matches) != 1 or not matches[0]:
        raise ValueError(f"manifest does not identify a unique hash for {path}")
    return matches[0]


def load_step_data(
    path: Path,
    tokenizer: Any,
    max_prompt_tokens: int,
    *,
    manifest_path: Path | None = None,
    expected_dataset_id: str = EXPECTED_DATASET_ID,
    expected_rows: int = 480,
    allowed_step_indices: tuple[int, ...] = (2,),
) -> tuple[Dataset, dict[str, Any]]:
    manifest_path = manifest_path or path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing frozen step-data manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = _manifest_step_hash(manifest, path)
    actual_hash = sha256_file(path)
    if expected_hash != actual_hash:
        raise ValueError(f"step-data hash mismatch: manifest={expected_hash}, actual={actual_hash}")
    rows: list[dict[str, Any]] = []
    lengths: list[int] = []
    allowed_steps = set(allowed_step_indices)
    if not allowed_steps or any(step <= 0 for step in allowed_steps):
        raise ValueError("allowed_step_indices must contain positive integers")
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: row must be an object")
        step_index = int(row.get("step_index") or 0)
        if row.get("dataset_id") != expected_dataset_id or step_index not in allowed_steps:
            raise ValueError(f"{path}:{line_number}: invalid dataset or target step")
        prompt = str(row.get("prompt") or "")
        if not prompt.endswith(NONTHINKING_SUFFIX):
            raise ValueError(f"{path}:{line_number}: prompt is not native Qwen3.5 non-thinking")
        length = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if length != int(row.get("prompt_token_count") or -1):
            raise ValueError(f"{path}:{line_number}: frozen token count changed")
        if length > max_prompt_tokens:
            raise ValueError(f"{path}:{line_number}: {length} tokens exceed hard gate {max_prompt_tokens}")
        rows.append(row)
        lengths.append(length)
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} target-step rows, got {len(rows)}")
    if len({row["prompt_sha256"] for row in rows}) != len(rows):
        raise ValueError("frozen prompts are not unique")
    lengths_sorted = sorted(lengths)
    stats = {
        "rows": len(rows),
        "min_prompt_tokens": min(lengths),
        "p50_prompt_tokens": lengths_sorted[len(lengths_sorted) // 2],
        "p95_prompt_tokens": lengths_sorted[int((len(lengths_sorted) - 1) * 0.95)],
        "p99_prompt_tokens": lengths_sorted[int((len(lengths_sorted) - 1) * 0.99)],
        "max_prompt_tokens": max(lengths),
        "step_data_sha256": actual_hash,
        "manifest_sha256": sha256_file(manifest_path),
        "manifest": str(manifest_path),
        "dataset_id": expected_dataset_id,
        "allowed_step_indices": sorted(allowed_steps),
        "step_index_counts": {
            str(step): count for step, count in sorted(Counter(int(row["step_index"]) for row in rows).items())
        },
    }
    return Dataset.from_list(rows), stats


def reward_inputs(kwargs: dict[str, Any], index: int) -> tuple[str, str, str, str, int]:
    return (
        kwargs.get("expected_step", [])[index],
        kwargs.get("forbidden_actions", [])[index],
        kwargs.get("reward_spec", [])[index],
        kwargs.get("previous_action", [])[index],
        int(kwargs.get("step_index", [])[index]),
    )


def make_task_reward() -> Callable[..., list[float]]:
    def task_reward(completions: list[Any], **kwargs: Any) -> list[float]:
        values: list[float] = []
        for index, completion in enumerate(completions):
            expected, forbidden, spec, previous, step_index = reward_inputs(kwargs, index)
            values.append(
                score_step_completion(
                    completion=first_json_text(completion),
                    expected_step=expected,
                    forbidden_actions=forbidden,
                    reward_spec=spec,
                    previous_action=previous,
                    full_expected_actions=kwargs.get("full_expected_actions", [])[index],
                    step_index=step_index,
                    first_json_only=True,
                )
            )
        return values

    return task_reward


def make_batch_format_reward(tokenizer: Any, max_completion_length: int) -> Callable[..., list[float]]:
    scalar_reward = make_format_reward(
        tokenizer=tokenizer,
        max_completion_length=max_completion_length,
        tail_penalty_tokens=64,
        prefix_penalty_tokens=16,
        penalize_truncated_completions=True,
    )

    def format_reward(completions: list[Any], **kwargs: Any) -> list[float]:
        del kwargs
        return [scalar_reward(completion) for completion in completions]

    return format_reward


def completion_diagnostics(completion: Any, expected_json: str, spec_json: str) -> dict[str, float]:
    actual = parse_completion(first_json_text(completion), first_json_only=True)
    if actual is None:
        return {"route_exact": 0.0, "argument_exact": 0.0, "stop_exact": 0.0}
    expected = json.loads(expected_json)
    spec = dict(rewardlib.DEFAULT_REWARD_SPEC)
    spec.update(json.loads(spec_json))
    _, info = rewardlib.score_expected_step(expected=expected, actual=actual, reward_spec=spec)
    detail = info.get("detail", {})
    return {
        "route_exact": float(float(detail.get("action_match") or 0.0) >= 1.0),
        "argument_exact": float(float(detail.get("argument_match") or 0.0) >= 1.0),
        "stop_exact": float(float(detail.get("finish_after_tool") or 0.0) >= 1.0),
    }


def make_diagnostic_reward(field: str) -> Callable[..., list[float]]:
    def diagnostic(completions: list[Any], **kwargs: Any) -> list[float]:
        values: list[float] = []
        for index, completion in enumerate(completions):
            values.append(
                completion_diagnostics(
                    completion,
                    kwargs.get("expected_step", [])[index],
                    kwargs.get("reward_spec", [])[index],
                )[field]
            )
        return values

    diagnostic.__name__ = field
    return diagnostic


def make_no_forbidden_action_reward() -> Callable[..., list[float]]:
    def no_forbidden_action(completions: list[Any], **kwargs: Any) -> list[float]:
        values: list[float] = []
        for index, completion in enumerate(completions):
            actual = parse_completion(first_json_text(completion), first_json_only=True)
            if actual is None:
                values.append(0.0)
                continue
            action = rewardlib.normalize_action(str(actual.get("action") or ""))
            decision_type = str(actual.get("decision_type") or "")
            if decision_type in {"clarify", "end"}:
                action = decision_type
            forbidden = {
                rewardlib.normalize_action(str(value))
                for value in json.loads(kwargs.get("forbidden_actions", [])[index])
            }
            values.append(float(action not in forbidden))
        return values

    return no_forbidden_action


def memory_snapshot(device: torch.device) -> dict[str, float]:
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    return {
        "allocated_gib": torch.cuda.memory_allocated(device) / GIB,
        "reserved_gib": torch.cuda.memory_reserved(device) / GIB,
        "max_allocated_gib": torch.cuda.max_memory_allocated(device) / GIB,
        "max_reserved_gib": torch.cuda.max_memory_reserved(device) / GIB,
        "device_free_gib": free / GIB,
        "device_total_gib": total / GIB,
    }


def physical_gpu_id() -> str:
    logical = torch.cuda.current_device()
    visible = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
    return visible[logical] if logical < len(visible) else str(logical)


def metric_tail(values: Any, offset: int) -> list[Any]:
    """Return newly appended metric values for list- and deque-backed TRL logs."""

    return list(values)[offset:]


class V100GRPOTrainer(GRPOTrainer):
    """TRL GRPO with the audited V100 scaler and rollout-distribution evidence."""

    def _build_accelerator_args(self, **kwargs: Any) -> dict[str, Any]:
        values = super()._build_accelerator_args(**kwargs)
        values.setdefault("kwargs_handlers", []).append(
            GradScalerKwargs(init_scale=1.0, growth_interval=100000)
        )
        return values

    def _event_path(self) -> Path:
        return Path(self.args.output_dir) / "telemetry" / f"rank{self.accelerator.process_index}.jsonl"

    def _record_memory(self, phase: str) -> None:
        memory = memory_snapshot(self.accelerator.device)
        append_jsonl(
            self._event_path(),
            {
                "timestamp": utc_now(),
                "event": "memory",
                "phase": phase,
                "global_step": int(self.state.global_step),
                "trainer_micro_step": int(self._step),
                "rank": self.accelerator.process_index,
                "physical_gpu_id": physical_gpu_id(),
                "memory": memory,
            },
        )
        maximum = float(getattr(self, "_capa_max_allocated_gib", math.inf))
        minimum_free = float(getattr(self, "_capa_minimum_free_gib", 0.0))
        if memory["max_allocated_gib"] > maximum:
            raise RuntimeError(
                f"{phase} memory gate failed: peak={memory['max_allocated_gib']:.3f} GiB > {maximum} GiB"
            )
        if memory["device_free_gib"] < minimum_free:
            raise RuntimeError(
                f"{phase} memory gate failed: free={memory['device_free_gib']:.3f} GiB < {minimum_free} GiB"
            )

    def _generate(self, prompts: list[Any]):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.accelerator.device)
        output = super()._generate(prompts)
        self._record_memory("generation")
        torch.cuda.reset_peak_memory_stats(self.accelerator.device)
        return output

    def _generate_and_score_completions(self, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        rank = self.accelerator.process_index
        reward_offsets = {
            name: len(self._logs["rewards"][name]) for name in self.reward_func_names
        }
        local_events = [
            {
                "rank": rank,
                "physical_gpu_id": physical_gpu_id(),
                "case_id": str(row.get("case_id") or ""),
                "prompt_sha256": str(row.get("prompt_sha256") or ""),
                "device_seed": int(torch.initial_seed()),
                "local_batch_index": index,
            }
            for index, row in enumerate(inputs)
        ]
        if len(local_events) != 1:
            raise RuntimeError(f"G3 expected local generation batch 1, got {len(local_events)} on rank {rank}")
        all_events = gather_object(local_events)
        expected = int(self.args.generation_batch_size)
        if len(all_events) != expected:
            raise RuntimeError(f"G3 gathered {len(all_events)} events, expected {expected}")
        groups: list[list[dict[str, Any]]] = []
        for offset in range(0, len(all_events), self.num_generations):
            group = all_events[offset : offset + self.num_generations]
            if len(group) != self.num_generations:
                raise RuntimeError("G3 found a partial prompt group")
            if len({event["case_id"] for event in group}) != 1:
                raise RuntimeError(f"G3 prompt group is not repeated across ranks: {group}")
            if len({event["rank"] for event in group}) != self.num_generations:
                raise RuntimeError(f"G3 completions are not on distinct ranks: {group}")
            if len({event["device_seed"] for event in group}) != self.num_generations:
                raise RuntimeError(f"G3 device-specific RNG seeds are not unique: {group}")
            for completion_index, event in enumerate(group):
                event["completion_index"] = completion_index
                event["prompt_group_id"] = (
                    f"call{getattr(self, '_capa_generation_call', 0)}-group{offset // self.num_generations}"
                )
            groups.append(group)

        result = super()._generate_and_score_completions(inputs)
        mode = "train" if self.model.training else "eval"
        # TRL 1.8 stores metric buffers as bounded deques.  Convert to a list
        # before taking the increment added by the parent call.
        new_rewards = {
            name: metric_tail(self._logs["rewards"][name], reward_offsets[name])
            for name in self.reward_func_names
        }
        weights = {
            name: float(weight)
            for name, weight in zip(
                self.reward_func_names,
                self.reward_weights.detach().cpu().tolist(),
                strict=True,
            )
        }
        for name, value in weighted_reward_statistics(new_rewards, weights).items():
            self._metrics[mode][name].append(value)

        gathered_advantages = self.accelerator.gather(
            result["advantages"].detach().float()
        ).cpu().tolist()
        for name, value in advantage_statistics(gathered_advantages).items():
            self._metrics[mode][name].append(value)
        completion_ids = result["completion_ids"]
        completion_mask = result["completion_mask"]
        local_outputs: list[dict[str, Any]] = []
        for row_index in range(completion_ids.shape[0]):
            ids = completion_ids[row_index][completion_mask[row_index].bool()].detach().cpu().tolist()
            local_outputs.append(
                {
                    "rank": rank,
                    "case_id": local_events[row_index]["case_id"],
                    "token_count": len(ids),
                    "token_sha256": hashlib.sha256(json.dumps(ids).encode("utf-8")).hexdigest(),
                    "clipped": bool(len(ids) >= int(self.max_completion_length)),
                }
            )
        all_outputs = gather_object(local_outputs)
        diverse_groups = 0
        for offset in range(0, len(all_outputs), self.num_generations):
            group_outputs = all_outputs[offset : offset + self.num_generations]
            if len({item["token_sha256"] for item in group_outputs}) > 1:
                diverse_groups += 1
        proof = {
            "timestamp": utc_now(),
            "event": "g3_distribution",
            "generation_call": int(getattr(self, "_capa_generation_call", 0)),
            "world_size": self.accelerator.num_processes,
            "local_generation_batch": len(local_events),
            "num_generations": self.num_generations,
            "generation_batch_size": self.args.generation_batch_size,
            "groups": groups,
            "outputs": all_outputs,
            "diverse_output_groups": diverse_groups,
            "all_output_groups_token_identical": diverse_groups == 0,
        }
        append_jsonl(self._event_path(), proof)
        self._capa_generation_call = int(getattr(self, "_capa_generation_call", 0)) + 1
        self._record_memory("reward_logprob")
        torch.cuda.reset_peak_memory_stats(self.accelerator.device)
        return result

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        mirror_policy_entropy_metric(self._metrics[mode])
        super().log(logs, start_time)


class FiniteGradientAndMemoryCallback(TrainerCallback):
    def __init__(self, output_dir: Path, max_allocated_gib: float, minimum_free_gib: float):
        self.output_dir = output_dir
        self.max_allocated_gib = max_allocated_gib
        self.minimum_free_gib = minimum_free_gib

    def _path(self) -> Path:
        rank = int(os.environ.get("RANK", "0"))
        return self.output_dir / "telemetry" / f"rank{rank}.jsonl"

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        local_bad = 0
        missing = 0
        checked = 0
        if model is None:
            raise RuntimeError("finite-gradient callback did not receive the model")
        for parameter in model.parameters():
            if not parameter.requires_grad:
                continue
            if parameter.grad is None:
                missing += 1
                continue
            checked += 1
            if not bool(torch.isfinite(parameter.grad).all().item()):
                local_bad += 1
        flag = torch.tensor(
            [int(local_bad > 0 or missing > 0)],
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
        append_jsonl(
            self._path(),
            {
                "timestamp": utc_now(),
                "event": "pre_optimizer_finite_gradient",
                "global_step": int(state.global_step),
                "rank": int(os.environ.get("RANK", "0")),
                "checked_gradient_tensors": checked,
                "missing_gradient_tensors": missing,
                "nonfinite_gradient_tensors": local_bad,
                "global_failure": bool(flag.item()),
            },
        )
        if bool(flag.item()):
            raise FloatingPointError(
                f"finite-gradient gate failed: local_nonfinite={local_bad}, local_missing={missing}"
            )
        return control

    def on_step_end(self, args, state, control, **kwargs):
        device = torch.device("cuda", torch.cuda.current_device())
        memory = memory_snapshot(device)
        append_jsonl(
            self._path(),
            {
                "timestamp": utc_now(),
                "event": "optimizer_step_end",
                "global_step": int(state.global_step),
                "rank": int(os.environ.get("RANK", "0")),
                "memory": memory,
            },
        )
        if memory["max_allocated_gib"] > self.max_allocated_gib:
            raise RuntimeError(
                f"memory gate failed: peak={memory['max_allocated_gib']:.3f} GiB > {self.max_allocated_gib} GiB"
            )
        if memory["device_free_gib"] < self.minimum_free_gib:
            raise RuntimeError(
                f"memory gate failed: free={memory['device_free_gib']:.3f} GiB < {self.minimum_free_gib} GiB"
            )
        torch.cuda.reset_peak_memory_stats(device)
        return control


def model_and_lora_audit(model: Any) -> dict[str, Any]:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if base.__class__.__name__ != EXPECTED_MODEL_CLASS:
        raise TypeError(f"expected {EXPECTED_MODEL_CLASS}, got {base.__class__.__name__}")
    visual = [name for name, _ in base.named_modules() if name == "visual" or name.startswith("visual.")]
    if visual:
        raise RuntimeError(f"text policy instantiated visual modules: {visual[:5]}")
    modules = [(name, module) for name, module in model.named_modules() if isinstance(module, BaseTunerLayer)]
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    target_counts: dict[str, int] = {}
    for name, _ in modules:
        leaf = name.rsplit(".", 1)[-1]
        target_counts[leaf] = target_counts.get(leaf, 0) + 1
    if len(modules) != EXPECTED_LORA_MODULES or trainable != EXPECTED_TRAINABLE_PARAMS:
        raise RuntimeError(
            f"LoRA mismatch: modules={len(modules)}, trainable={trainable}, targets={target_counts}"
        )
    return {
        "base_model_class": base.__class__.__name__,
        "visual_module_count": len(visual),
        "lora_module_count": len(modules),
        "trainable_parameter_count": trainable,
        "target_counts": dict(sorted(target_counts.items())),
        "module_names": [name for name, _ in modules],
    }


def trainable_fingerprint(model: Any) -> dict[str, float | int]:
    parameters = [parameter.detach() for parameter in model.parameters() if parameter.requires_grad]
    return {
        "tensors": len(parameters),
        "elements": sum(parameter.numel() for parameter in parameters),
        "sum": sum(float(parameter.float().sum().item()) for parameter in parameters),
        "square_sum": sum(float(parameter.float().square().sum().item()) for parameter in parameters),
    }


def package_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "trl": trl.__version__,
        "peft": peft.__version__,
        "accelerate": accelerate.__version__,
        "datasets": datasets.__version__,
    }


def summarize_log_history(log_history: list[dict[str, Any]]) -> dict[str, Any]:
    steps = [entry for entry in log_history if entry.get("step") is not None and entry.get("reward") is not None]
    keys = [
        "reward",
        "reward_std",
        "frac_reward_zero_std",
        "rewards/task_reward/mean",
        "rewards/format_reward/mean",
        "rewards/route_exact/mean",
        "rewards/argument_exact/mean",
        "rewards/stop_exact/mean",
        "rewards/no_forbidden_action/mean",
        "completions/clipped_ratio",
        "grad_norm",
    ]
    means: dict[str, float] = {}
    for key in keys:
        values = [float(entry[key]) for entry in steps if entry.get(key) is not None]
        if values:
            means[key] = sum(values) / len(values)
    return {
        "logged_steps": len(steps),
        "means": means,
        "nonfinite_gradient_logs": sum(
            not math.isfinite(float(entry["grad_norm"]))
            for entry in steps
            if entry.get("grad_norm") is not None
        ),
        "final": steps[-1] if steps else {},
    }


def main() -> None:
    args = parse_args()
    started_at = utc_now()
    started_perf = time.perf_counter()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_main = rank == 0
    model_path = args.model_name_or_path.resolve()
    adapter_path = args.adapter_path.resolve() if args.adapter_path is not None else None
    step_data = args.step_data if args.step_data.is_absolute() else ROOT / args.step_data
    step_data_manifest = None
    if args.step_data_manifest is not None:
        step_data_manifest = (
            args.step_data_manifest
            if args.step_data_manifest.is_absolute()
            else ROOT / args.step_data_manifest
        )
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir

    effective_report_to = (
        "wandb" if args.report_to == "auto" and args.mode == "train" else
        "none" if args.report_to == "auto" else args.report_to
    )
    report_to = parse_report_to(effective_report_to)
    run_name = args.run_name or output_dir.name
    wandb_settings = configure_wandb_environment(
        report_to=report_to,
        output_dir=output_dir,
        stage="grpo",
        run_name=run_name,
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        tags=args.wandb_tags,
        mode=args.wandb_mode,
    )

    if args.mode == "train" and not args.confirm_train:
        raise PermissionError("optimizer steps require explicit --confirm-train")
    if args.expected_world_size not in {4, 8}:
        raise ValueError("expected_world_size must be 4 or 8")
    if args.mode in {"g4", "train"} and world_size != args.expected_world_size:
        raise RuntimeError(
            f"Qwen3.5 GRPO expected world_size={args.expected_world_size}, got {world_size}"
        )
    if args.num_generations != 4 or args.generation_batch_size != args.expected_world_size:
        raise ValueError(
            "the distributed rollout requires num_generations=4 and "
            "generation_batch_size=expected_world_size"
        )
    if args.per_device_train_batch_size != 1:
        raise ValueError("the audited first run requires per_device_train_batch_size=1")
    if args.max_completion_length != 320:
        raise ValueError("the measured first run requires max_completion_length=320")
    if args.mode == "g4":
        args.max_steps = 1
        args.gradient_accumulation_steps = 1
    elif args.mode == "train" and world_size * args.gradient_accumulation_steps != 32:
        raise ValueError(
            "training must preserve 32 completions per optimizer step; expected "
            f"world_size * gradient_accumulation_steps = 32, got "
            f"{world_size} * {args.gradient_accumulation_steps}"
        )
    if adapter_path is not None and not (adapter_path / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"missing SFT adapter weights: {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=False,
        use_fast=True,
        padding_side="left",
    )
    if tokenizer.eos_token_id != EXPECTED_EOS_ID or tokenizer.pad_token_id != EXPECTED_PAD_ID:
        raise RuntimeError(
            f"tokenizer stop contract mismatch: eos={tokenizer.eos_token_id}, pad={tokenizer.pad_token_id}"
        )
    dataset, data_stats = load_step_data(
        step_data,
        tokenizer,
        args.max_prompt_tokens,
        manifest_path=step_data_manifest,
        expected_dataset_id=args.expected_dataset_id,
        expected_rows=args.expected_rows,
        allowed_step_indices=args.allowed_step_indices,
    )

    run_config: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "prepared",
        "started_at": started_at,
        "mode": args.mode,
        "optimizer_steps_authorized": bool(args.mode == "train" and args.confirm_train),
        "model_name_or_path": str(model_path),
        "adapter_path": str(adapter_path) if adapter_path is not None else "",
        "adapter_sha256": (
            sha256_file(adapter_path / "adapter_model.safetensors")
            if adapter_path is not None
            else ""
        ),
        "step_data": str(step_data),
        "output_dir": str(output_dir),
        "dataset": data_stats,
        "packages": package_versions(),
        "distributed": {
            "world_size": world_size,
            "expected_world_size": args.expected_world_size,
            "rank": rank,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "num_generations": args.num_generations,
            "generation_batch_size": args.generation_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
        },
        "generation": {
            "max_completion_length": args.max_completion_length,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "use_cache": True,
            "remove_invalid_values": True,
            "renormalize_logits": True,
        },
        "optimization": {
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": 0.0,
            "lr_scheduler_type": "constant_with_warmup",
            "warmup_steps": args.warmup_steps,
            "max_grad_norm": 1.0,
            "loss_type": "dr_grpo",
            "scale_rewards": "group",
            "beta": 0.0,
            "epsilon": 0.2,
            "fp16": True,
            "bf16": False,
            "grad_scaler_init_scale": 1.0,
            "grad_scaler_growth_interval": 100000,
        },
        "reward_weights": {
            "task_reward": args.task_reward_weight,
            "format_reward": args.format_reward_weight,
            "route_exact": 0.0,
            "argument_exact": 0.0,
            "stop_exact": 0.0,
            "no_forbidden_action": 0.0,
        },
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.0,
            "target_modules": TARGET_MODULES,
        },
        "hardware_guards": {
            "cudnn_enabled": False,
            "max_allocated_gib": args.max_allocated_gib,
            "minimum_free_gib": args.minimum_free_gib,
        },
        "observability": {
            "wandb": wandb_settings,
            "metric_contract": metric_contract("grpo"),
            "dashboard_config": "configs/wandb/post_training_v1.json",
        },
    }
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "capa_qwen35_grpo_config.json", run_config)
        print(json.dumps(run_config, ensure_ascii=False, indent=2), flush=True)
    if args.mode == "dry-run":
        return

    torch.backends.cudnn.enabled = False
    set_seed(args.seed)
    reward_funcs = [
        make_task_reward(),
        make_batch_format_reward(tokenizer, args.max_completion_length),
        make_diagnostic_reward("route_exact"),
        make_diagnostic_reward("argument_exact"),
        make_diagnostic_reward("stop_exact"),
        make_no_forbidden_action_reward(),
    ]
    reward_weights = [
        args.task_reward_weight,
        args.format_reward_weight,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    training_args = GRPOConfig(
        output_dir=str(output_dir),
        do_train=True,
        save_strategy="no" if args.mode == "g4" else "steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        logging_nan_inf_filter=False,
        report_to=report_to,
        run_name=run_name,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.0,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        max_grad_norm=1.0,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_cache=False,
        remove_unused_columns=False,
        optim="adamw_torch",
        seed=args.seed,
        data_seed=args.seed,
        num_generations=args.num_generations,
        generation_batch_size=args.generation_batch_size,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=0,
        repetition_penalty=1.0,
        generation_kwargs={
            "remove_invalid_values": True,
            "renormalize_logits": True,
            "use_cache": True,
            "eos_token_id": EXPECTED_EOS_ID,
            "pad_token_id": EXPECTED_PAD_ID,
        },
        use_vllm=False,
        beta=0.0,
        num_iterations=1,
        epsilon=0.2,
        loss_type="dr_grpo",
        scale_rewards="group",
        reward_weights=reward_weights,
        mask_truncated_completions=False,
        ddp_find_unused_parameters=False,
        average_tokens_across_devices=True,
        torch_empty_cache_steps=1,
        torch_compile=False,
        trust_remote_code=False,
        model_init_kwargs=(
            None
            if adapter_path is not None
            else {
                "dtype": "float16",
                "attn_implementation": "sdpa",
                "low_cpu_mem_usage": True,
                "trust_remote_code": False,
            }
        ),
        disable_dropout=True,
        shuffle_dataset=True,
    )
    if training_args.steps_per_generation != 1:
        raise RuntimeError(f"expected steps_per_generation=1, got {training_args.steps_per_generation}")
    model_input: str | Any = str(model_path)
    peft_config: LoraConfig | None = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    if adapter_path is not None:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        model_input = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            is_trainable=True,
        )
        peft_config = None
    callback = FiniteGradientAndMemoryCallback(
        output_dir=output_dir,
        max_allocated_gib=args.max_allocated_gib,
        minimum_free_gib=args.minimum_free_gib,
    )
    trainer: V100GRPOTrainer | None = None
    failure_path = output_dir / "telemetry" / f"rank{rank}_failure.json"
    try:
        trainer = V100GRPOTrainer(
            model=model_input,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
            callbacks=[callback],
        )
        trainer._capa_max_allocated_gib = args.max_allocated_gib
        trainer._capa_minimum_free_gib = args.minimum_free_gib
        audit = model_and_lora_audit(trainer.model)
        scaler = trainer.accelerator.scaler
        if scaler is None or float(scaler.get_scale()) != 1.0:
            raise RuntimeError(f"unexpected GradScaler: {scaler}")
        scaler_growth_interval = int(getattr(scaler, "_growth_interval", -1))
        if scaler_growth_interval != 100000:
            raise RuntimeError(f"unexpected GradScaler growth_interval={scaler_growth_interval}")
        if is_main:
            run_config["model_audit"] = audit
            run_config["grad_scaler"] = {
                "scale": float(scaler.get_scale()),
                "growth_interval": scaler_growth_interval,
            }
            write_json(output_dir / "capa_qwen35_grpo_config.json", run_config)

        before_fingerprint: dict[str, float | int] | None = None
        if args.mode == "g4":
            trainer.create_optimizer_and_scheduler(num_training_steps=1)
            if trainer.optimizer is None:
                raise RuntimeError("G4 could not construct an optimizer for the no-update gate")
            before_fingerprint = trainable_fingerprint(trainer.model)

            def no_update_step(closure=None):
                if closure is not None:
                    closure()
                return None

            trainer.optimizer.step = no_update_step  # type: ignore[method-assign]

        train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        if args.mode == "g4":
            after_fingerprint = trainable_fingerprint(trainer.model)
            if before_fingerprint != after_fingerprint:
                raise RuntimeError(
                    f"G4 no-update fingerprint changed: before={before_fingerprint}, after={after_fingerprint}"
                )
            optimizer_steps = 0
        else:
            trainer.save_model(str(output_dir))
            if is_main:
                tokenizer.save_pretrained(str(output_dir))
            optimizer_steps = int(trainer.state.global_step)

        result = {
            "schema_version": "1.0",
            "status": "pass" if args.mode == "g4" else "completed",
            "mode": args.mode,
            "started_at": started_at,
            "finished_at": utc_now(),
            "runtime_seconds": time.perf_counter() - started_perf,
            "optimizer_steps": optimizer_steps,
            "train_metrics": dict(train_result.metrics),
            "log_summary": summarize_log_history(list(trainer.state.log_history)),
            "model_audit": audit,
            "g4_no_update_fingerprint": before_fingerprint if args.mode == "g4" else None,
            "output_dir": str(output_dir),
        }
        if is_main:
            write_json(output_dir / "capa_qwen35_grpo_result.json", result)
    except Exception as exc:
        failure = {
            "schema_version": "1.0",
            "status": "fail",
            "mode": args.mode,
            "rank": rank,
            "started_at": started_at,
            "finished_at": utc_now(),
            "optimizer_steps": int(trainer.state.global_step) if trainer is not None else 0,
            "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
        }
        write_json(failure_path, failure)
        raise


if __name__ == "__main__":
    main()
