#!/usr/bin/env python3
"""Train the CAPA Planner with a lightweight clipped PPO objective.

This intentionally reuses the GRPO step dataset and reward function so PPO and
GRPO differ mainly in the optimizer objective, not in data construction.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


ROOT = Path(__file__).resolve().parents[3]
DEMO_DIR = ROOT / "demo"
for path in (ROOT, DEMO_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from training.planner_grpo_seed_v1.scripts.train_planner_grpo import (  # noqa: E402
    DEFAULT_CASES,
    build_step_dataset,
    expected_decision_to_planner_step,
    load_jsonl,
    score_step_completion,
)
from util.path_resolver import resolve_model_name_or_path  # noqa: E402

torch.backends.cudnn.enabled = False


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clipped PPO for the CAPA Planner.")
    parser.add_argument("--model-name-or-path", default="/raid/zkq/models/Qwen3.5-4B")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "planner-ppo-qwen35-4b-focused-full"))
    parser.add_argument("--max-prompt-length", type=int, default=3072)
    parser.add_argument("--max-completion-length", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--kl-coef", type=float, default=0.02)
    parser.add_argument("--rollout-source", choices=["model", "expected"], default="model")
    parser.add_argument(
        "--advantage-baseline",
        type=float,
        default=None,
        help="Fallback scalar baseline when reward std is zero, useful for fixed expected rollouts.",
    )
    parser.add_argument("--max-updates", type=int, default=-1)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", type=parse_bool, default=True)
    parser.add_argument("--bf16", type=parse_bool, default=False)
    parser.add_argument("--gradient-checkpointing", type=parse_bool, default=True)
    parser.add_argument("--trust-remote-code", type=parse_bool, default=True)
    parser.add_argument("--resume-from-checkpoint", default="")
    return parser.parse_args()


def collate_rows(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    keys = rows[0].keys()
    return {key: [row[key] for row in rows] for key in keys}


def sequence_logprobs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_starts: int | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits[:, :-1, :].float()
    labels = input_ids[:, 1:]
    shifted_mask = attention_mask[:, 1:].bool()
    positions = torch.arange(labels.shape[1], device=labels.device).unsqueeze(0) + 1
    if isinstance(completion_starts, torch.Tensor):
        starts = completion_starts.to(labels.device).long().unsqueeze(1)
    else:
        starts = torch.full((labels.shape[0], 1), int(completion_starts), device=labels.device, dtype=torch.long)
    completion_mask = shifted_mask & (positions >= starts)
    token_logprobs = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    lengths = completion_mask.sum(dim=1).clamp(min=1)
    seq_logprobs = (token_logprobs * completion_mask).sum(dim=1) / lengths
    return seq_logprobs, lengths


def completion_rewards(completions: list[str], batch: dict[str, list[Any]]) -> list[float]:
    scores: list[float] = []
    for idx, completion in enumerate(completions):
        scores.append(
            score_step_completion(
                completion=completion,
                expected_step=batch["expected_step"][idx],
                forbidden_actions=batch["forbidden_actions"][idx],
                reward_spec=batch["reward_spec"][idx],
                previous_action=batch["previous_action"][idx],
                full_expected_actions=batch["full_expected_actions"][idx],
                step_index=int(batch["step_index"][idx]),
            )
        )
    return scores


def expected_completions(batch: dict[str, list[Any]]) -> list[str]:
    completions: list[str] = []
    for raw in batch["expected_step"]:
        expected = json.loads(raw)
        decision = expected_decision_to_planner_step(expected)
        completions.append(json.dumps(decision, ensure_ascii=False))
    return completions


def tokenize_prompt_completions(
    tokenizer: Any,
    prompts: list[str],
    completions: list[str],
    max_prompt_length: int,
    max_completion_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded_rows: list[list[int]] = []
    starts: list[int] = []
    for prompt, completion in zip(prompts, completions):
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_length,
        )["input_ids"]
        completion_ids = tokenizer(
            completion,
            add_special_tokens=False,
            truncation=True,
            max_length=max_completion_length,
        )["input_ids"]
        starts.append(len(prompt_ids))
        encoded_rows.append(prompt_ids + completion_ids)

    max_len = max(len(row) for row in encoded_rows)
    pad_id = tokenizer.pad_token_id
    input_ids = torch.full((len(encoded_rows), max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(encoded_rows), max_len), dtype=torch.long, device=device)
    padded_starts = []
    for idx, row in enumerate(encoded_rows):
        offset = max_len - len(row)
        input_ids[idx, offset:] = torch.tensor(row, dtype=torch.long, device=device)
        attention_mask[idx, offset:] = 1
        padded_starts.append(offset + starts[idx])
    return input_ids, attention_mask, torch.tensor(padded_starts, dtype=torch.long, device=device)


def save_model(accelerator: Accelerator, model: torch.nn.Module, tokenizer: Any, output_dir: Path) -> None:
    accelerator.wait_for_everyone()


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            values[key] = str(value)
        else:
            values[key] = value
    return values
    unwrapped = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        unwrapped.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
    accelerator.wait_for_everyone()


def main() -> None:
    args = parse_args()
    args.model_name_or_path = resolve_model_name_or_path(args.model_name_or_path, ROOT)
    set_seed(args.seed)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)

    cases_path = args.cases if args.cases.is_absolute() else ROOT / args.cases
    dataset = build_step_dataset(load_jsonl(cases_path))
    rows = [dataset[idx] for idx in range(len(dataset))]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    if args.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    dataloader = DataLoader(
        rows,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_rows,
        drop_last=False,
    )
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "training_args.json").write_text(json.dumps(jsonable_args(args), indent=2), encoding="utf-8")
    accelerator.wait_for_everyone()

    global_update = 0
    running = []
    model.train()
    for batch in dataloader:
        if args.max_updates > 0 and global_update >= args.max_updates:
            break

        prompts = [str(item) for item in batch["prompt"]]
        tokenized = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_prompt_length,
        )
        tokenized = {key: value.to(accelerator.device) for key, value in tokenized.items()}
        prompt_width = int(tokenized["input_ids"].shape[1])

        if args.rollout_source == "expected":
            completions = expected_completions(batch)
            generated, full_attention_mask, completion_starts = tokenize_prompt_completions(
                tokenizer,
                prompts,
                completions,
                args.max_prompt_length,
                args.max_completion_length,
                accelerator.device,
            )
            with torch.no_grad():
                old_logprobs, lengths = sequence_logprobs(model, generated, full_attention_mask, completion_starts)
        else:
            with torch.no_grad():
                generated = model.generate(
                    **tokenized,
                    max_new_tokens=args.max_completion_length,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    synced_gpus=accelerator.num_processes > 1,
                )
                gen_width = int(generated.shape[1] - prompt_width)
                full_attention_mask = torch.cat(
                    [
                        tokenized["attention_mask"],
                        torch.ones((generated.shape[0], gen_width), dtype=tokenized["attention_mask"].dtype, device=accelerator.device),
                    ],
                    dim=1,
                )
                completion_starts = prompt_width
                old_logprobs, lengths = sequence_logprobs(model, generated, full_attention_mask, completion_starts)

            completions = [
                tokenizer.decode(seq[prompt_width:], skip_special_tokens=True).strip()
                for seq in generated.detach().cpu()
            ]
        rewards = torch.tensor(completion_rewards(completions, batch), dtype=torch.float32, device=accelerator.device)
        gathered_rewards = accelerator.gather(rewards)
        reward_mean = gathered_rewards.mean()
        reward_std_raw = gathered_rewards.std(unbiased=False)
        if float(reward_std_raw.detach().cpu()) < 1e-6 and args.advantage_baseline is not None:
            advantages = rewards - float(args.advantage_baseline)
            reward_std = torch.tensor(0.0, device=accelerator.device)
        else:
            reward_std = reward_std_raw.clamp(min=1e-6)
            advantages = (rewards - reward_mean) / reward_std
        if not torch.isfinite(advantages).all():
            advantages = torch.zeros_like(rewards)

        last_loss = torch.tensor(0.0, device=accelerator.device)
        last_kl = torch.tensor(0.0, device=accelerator.device)
        last_clipfrac = torch.tensor(0.0, device=accelerator.device)
        for _ in range(max(1, args.ppo_epochs)):
            with accelerator.accumulate(model):
                new_logprobs, _ = sequence_logprobs(model, generated, full_attention_mask, completion_starts)
                logratio = (new_logprobs - old_logprobs.detach()).clamp(min=-20.0, max=20.0)
                ratio = torch.exp(logratio)
                unclipped = ratio * advantages
                clipped = ratio.clamp(1.0 - args.clip_range, 1.0 + args.clip_range) * advantages
                policy_loss = -torch.min(unclipped, clipped).mean()
                approx_kl = ((ratio - 1.0) - logratio).mean()
                loss = policy_loss + args.kl_coef * approx_kl
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                last_loss = loss.detach()
                last_kl = approx_kl.detach()
                last_clipfrac = ((ratio - 1.0).abs() > args.clip_range).float().mean().detach()

        global_update += 1
        metric = {
            "update": global_update,
            "reward_mean": float(reward_mean.detach().cpu()),
            "reward_local": float(rewards.mean().detach().cpu()),
            "reward_min_local": float(rewards.min().detach().cpu()),
            "reward_max_local": float(rewards.max().detach().cpu()),
            "advantage_mean_local": float(advantages.mean().detach().cpu()),
            "loss": float(last_loss.detach().cpu()),
            "approx_kl": float(last_kl.detach().cpu()),
            "clipfrac": float(last_clipfrac.detach().cpu()),
            "completion_tokens_mean": float(lengths.float().mean().detach().cpu()),
            "cuda_max_memory_mb": (
                round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)
                if torch.cuda.is_available()
                else None
            ),
        }
        running.append(metric)
        if accelerator.is_main_process and (global_update == 1 or global_update % args.logging_steps == 0):
            with (output_dir / "train_metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metric, ensure_ascii=False) + "\n")
            accelerator.print(json.dumps(metric, ensure_ascii=False))
        if args.save_steps > 0 and global_update % args.save_steps == 0:
            save_model(accelerator, model, tokenizer, output_dir / f"checkpoint-{global_update}")

    if not running and accelerator.is_main_process:
        raise RuntimeError("no PPO updates were run")
    save_model(accelerator, model, tokenizer, output_dir)
    if accelerator.is_main_process:
        summary = {
            "updates": global_update,
            "last": running[-1] if running else None,
            "mean_reward_observed": sum(item["reward_local"] for item in running) / max(1, len(running)),
            "objective": "clipped PPO policy surrogate without critic; reward baseline normalized across visible processes",
        }
        (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
