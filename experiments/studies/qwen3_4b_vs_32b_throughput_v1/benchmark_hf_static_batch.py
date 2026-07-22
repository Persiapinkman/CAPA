#!/usr/bin/env python3
"""Benchmark the verified Qwen3.5 causal LM with synchronized static batches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shlex
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import benchmark_openai_throughput as shared


torch.backends.cudnn.enabled = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def query_gpu_rows(indices: list[int], relative_s: float) -> list[dict[str, Any]]:
    query = "index,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm"
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
            "-i",
            ",".join(str(index) for index in indices),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    fields = ["gpu_index", "utilization_pct", "memory_used_mib", "power_w", "temperature_c", "sm_clock_mhz"]
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        values = [part.strip() for part in line.split(",")]
        if len(values) != len(fields):
            continue
        row: dict[str, Any] = {"relative_s": relative_s}
        for field, value in zip(fields, values):
            try:
                row[field] = int(float(value)) if field in {"gpu_index", "temperature_c", "sm_clock_mhz"} else float(value)
            except ValueError:
                row[field] = None
        rows.append(row)
    return rows


def telemetry_worker(
    gpu_indices: list[int],
    started: float,
    stop: threading.Event,
    rows: list[dict[str, Any]],
    errors: list[str],
) -> None:
    while not stop.is_set():
        try:
            rows.extend(query_gpu_rows(gpu_indices, time.perf_counter() - started))
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        stop.wait(0.5)


def generation_config(model: Any, tokenizer: Any, output_length: int) -> GenerationConfig:
    config = GenerationConfig.from_model_config(model.config)
    config.max_new_tokens = output_length
    config.min_new_tokens = output_length
    config.do_sample = False
    config.eos_token_id = None
    config.pad_token_id = tokenizer.pad_token_id
    config.use_cache = True
    config.return_dict_in_generate = False
    return config


def run_wave(
    model: Any,
    tokenizer: Any,
    prompts: list[list[int]],
    input_length: int,
    output_length: int,
    device: torch.device,
    request_prefix: str,
) -> tuple[list[dict[str, Any]], float, str]:
    input_ids = torch.tensor(prompts, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    config = generation_config(model, tokenizer, output_length)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        sequences = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=config,
        )
    torch.cuda.synchronize(device)
    duration_s = time.perf_counter() - started
    generated = sequences[:, input_length:]
    generated_cpu = generated.detach().cpu()
    preview = tokenizer.decode(generated_cpu[0].tolist(), skip_special_tokens=True)
    requests: list[dict[str, Any]] = []
    for index, token_ids in enumerate(generated_cpu):
        output_ids = token_ids.tolist()
        output_hash = hashlib.sha256(json.dumps(output_ids, separators=(",", ":")).encode()).hexdigest()
        requests.append({
            "request_id": f"{request_prefix}-r{index:03d}",
            "success": len(output_ids) == output_length,
            "http_status": None,
            "prompt_tokens": input_length,
            "completion_tokens": len(output_ids),
            "total_tokens": input_length + len(output_ids),
            "latency_s": duration_s,
            "ttfb_s": None,
            "ttft_s": None,
            "tpot_s": None,
            "stream_content_chunks": None,
            "finish_reason": "length" if len(output_ids) == output_length else "invalid_length",
            "output_chars": len(tokenizer.decode(output_ids, skip_special_tokens=True)),
            "output_sha256": output_hash,
            "error": None if len(output_ids) == output_length else f"generated {len(output_ids)} tokens",
        })
    del input_ids, attention_mask, sequences, generated, generated_cpu
    return requests, duration_s, preview


def summarize_requests(requests: list[dict[str, Any]], duration_s: float) -> dict[str, Any]:
    successful = [request for request in requests if request["success"]]
    prompt_tokens = sum(request["prompt_tokens"] for request in successful)
    completion_tokens = sum(request["completion_tokens"] for request in successful)
    latencies = [request["latency_s"] for request in successful]
    return {
        "request_count": len(requests),
        "success_count": len(successful),
        "error_count": len(requests) - len(successful),
        "duration_s": duration_s,
        "request_per_s": len(successful) / duration_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prompt_tok_per_s": prompt_tokens / duration_s,
        "output_tok_per_s": completion_tokens / duration_s,
        "total_tok_per_s": (prompt_tokens + completion_tokens) / duration_s,
        "latency_mean_s": statistics.fmean(latencies),
        "latency_p50_s": shared.percentile(latencies, 0.50),
        "latency_p95_s": shared.percentile(latencies, 0.95),
        "ttft_p50_s": None,
        "ttft_p95_s": None,
        "tpot_p50_s": None,
        "tpot_p95_s": None,
    }


def run_trial(
    model: Any,
    tokenizer: Any,
    prompt_rows: list[list[int]],
    input_length: int,
    output_length: int,
    batch_size: int,
    waves: int,
    phase: str,
    block: int,
    order_index: int,
    gpu_indices: list[int],
    device: torch.device,
) -> dict[str, Any]:
    expected = batch_size * waves
    if len(prompt_rows) != expected:
        raise ValueError(f"received {len(prompt_rows)} prompts, expected {expected}")

    trial_started_at = utc_now()
    trial_started = time.perf_counter()
    telemetry_rows: list[dict[str, Any]] = []
    telemetry_errors: list[str] = []
    telemetry_stop = threading.Event()
    telemetry_thread = threading.Thread(
        target=telemetry_worker,
        args=(gpu_indices, trial_started, telemetry_stop, telemetry_rows, telemetry_errors),
        daemon=True,
    )
    telemetry_thread.start()

    requests: list[dict[str, Any]] = []
    wave_durations: list[float] = []
    previews: list[str] = []
    for wave in range(waves):
        start = wave * batch_size
        selected = prompt_rows[start : start + batch_size]
        wave_requests, wave_duration, preview = run_wave(
            model=model,
            tokenizer=tokenizer,
            prompts=selected,
            input_length=input_length,
            output_length=output_length,
            device=device,
            request_prefix=(
                f"{phase}-b{block:02d}-o{order_index:02d}-i{input_length}-c{batch_size}-w{wave:02d}"
            ),
        )
        requests.extend(wave_requests)
        wave_durations.append(wave_duration)
        previews.append(preview[:500])

    duration_s = time.perf_counter() - trial_started
    telemetry_stop.set()
    telemetry_thread.join(timeout=5)
    telemetry = shared.summarize_telemetry(telemetry_rows, gpu_indices, duration_s)
    summary = summarize_requests(requests, duration_s)
    invalid_reasons: list[str] = []
    if any(not request["success"] for request in requests):
        invalid_reasons.extend(request["error"] for request in requests if not request["success"])
    if phase != "calibration" and telemetry["sample_rows"] < 2 * len(gpu_indices):
        invalid_reasons.append(f"only {telemetry['sample_rows']} telemetry rows")
    return {
        "phase": phase,
        "block": block,
        "condition_order_index": order_index,
        "input_length": input_length,
        "output_length": output_length,
        "concurrency": batch_size,
        "execution_semantics": "synchronized_static_batch",
        "waves": waves,
        "request_count_planned": expected,
        "started_at": trial_started_at,
        "ended_at": utc_now(),
        "valid": not invalid_reasons,
        "invalid_reasons": invalid_reasons,
        "summary": summary,
        "wave_duration_s": wave_durations,
        "output_previews": previews,
        "telemetry_summary": telemetry,
        "telemetry_errors": telemetry_errors,
        "requests": requests,
        "telemetry": telemetry_rows,
    }


def chat_sanity(model: Any, tokenizer: Any, device: torch.device) -> dict[str, Any]:
    messages = [{
        "role": "user",
        "content": "Write one concise sentence explaining why reproducible benchmarks need fixed workloads.",
    }]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(device)
    config = GenerationConfig.from_model_config(model.config)
    config.max_new_tokens = 64
    config.do_sample = False
    config.pad_token_id = tokenizer.pad_token_id
    config.use_cache = True
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        sequences = model.generate(**inputs, generation_config=config)
    torch.cuda.synchronize(device)
    duration_s = time.perf_counter() - started
    output_ids = sequences[0, inputs.input_ids.shape[1] :].detach().cpu().tolist()
    text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    unique_ratio = len(set(output_ids)) / max(len(output_ids), 1)
    valid = len(text) >= 20 and unique_ratio >= 0.20
    return {
        "valid": valid,
        "model_class": model.__class__.__name__,
        "input_tokens": int(inputs.input_ids.shape[1]),
        "completion_tokens": len(output_ids),
        "duration_s": duration_s,
        "unique_token_ratio": unique_ratio,
        "text": text,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--prompt-pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--phase", default="formal")
    parser.add_argument("--gpu-indices", type=int, nargs="+", default=[0])
    parser.add_argument("--conditions", default="256:1,256:4,256:16,2048:1,2048:4,2048:16")
    parser.add_argument("--output-length", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--waves-per-trial", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if len(args.gpu_indices) != 1:
        raise ValueError("the verified Qwen3.5 static-batch path is single-GPU")

    conditions = shared.parse_conditions(args.conditions)
    prompt_pool = json.loads(args.prompt_pool.read_text())
    prompts = prompt_pool["prompts"]
    device = torch.device("cuda:0")
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False, use_fast=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    loaded = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        output_loading_info=True,
    )
    model, loading_info = loaded
    model.to(device).eval()
    load_duration_s = time.perf_counter() - load_started
    if model.__class__.__name__ != "Qwen3_5ForCausalLM":
        raise TypeError(f"expected Qwen3_5ForCausalLM, got {model.__class__.__name__}")
    loading_contract = {
        key: list(loading_info.get(key) or [])
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    }
    if any(loading_contract.values()):
        raise RuntimeError(f"weight loading contract failed: {loading_contract}")

    sanity = chat_sanity(model, tokenizer, device)
    if not sanity["valid"]:
        raise RuntimeError(f"chat sanity failed: {sanity}")

    state: dict[str, Any] = {
        "schema_version": 1,
        "study_id": "qwen3_4b_vs_32b_throughput_v1",
        "run_id": args.run_id,
        "phase": args.phase,
        "created_at": utc_now(),
        "completed_at": None,
        "updated_at": utc_now(),
        "status": "running",
        "model_label": args.model_label,
        "served_model_name": str(args.model.resolve()),
        "backend": "transformers_static_batch",
        "execution_semantics": "two_synchronized_static_batch_waves_per_trial",
        "gpu_indices": args.gpu_indices,
        "gpu_count": 1,
        "conditions": [{"input_length": item[0], "concurrency": item[1]} for item in conditions],
        "output_length": args.output_length,
        "blocks": args.blocks,
        "waves_per_trial": args.waves_per_trial,
        "seed": args.seed,
        "prompt_pool_path": str(args.prompt_pool.resolve()),
        "prompt_pool_sha256": sha256_file(args.prompt_pool),
        "prompt_ids_sha256": prompt_pool.get("prompt_ids_sha256"),
        "client_script_sha256": sha256_file(Path(__file__)),
        "client_python": sys.version,
        "client_platform": platform.platform(),
        "argv": shlex.join(sys.argv),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "model_class": model.__class__.__name__,
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "model_load_duration_s": load_duration_s,
        "loading_contract": loading_contract,
        "sanity": sanity,
        "calibration": None,
        "warmups": [],
        "trials": [],
    }
    shared.write_state(args.output, state)

    cursors = {input_length: 0 for input_length, _ in conditions}
    calibration_length = conditions[0][0]
    calibration_prompt = prompts[str(calibration_length)][0]
    cursors[calibration_length] += 1
    calibration = run_trial(
        model, tokenizer, [calibration_prompt], calibration_length, 8, 1, 1,
        "calibration", 0, 0, args.gpu_indices, device,
    )
    state["calibration"] = calibration
    shared.write_state(args.output, state)
    if not calibration["valid"]:
        state["status"] = "failed_calibration"
        shared.write_state(args.output, state)
        raise RuntimeError(calibration["invalid_reasons"])

    for warmup_index, (input_length, batch_size) in enumerate(conditions):
        start = cursors[input_length]
        stop = start + batch_size
        selected = prompts[str(input_length)][start:stop]
        cursors[input_length] = stop
        warmup = run_trial(
            model, tokenizer, selected, input_length, 32, batch_size, 1,
            "warmup", 0, warmup_index, args.gpu_indices, device,
        )
        state["warmups"].append(warmup)
        shared.write_state(args.output, state)
        if not warmup["valid"]:
            state["status"] = "failed_warmup"
            shared.write_state(args.output, state)
            raise RuntimeError(warmup["invalid_reasons"])

    rng = random.Random(args.seed)
    for block in range(1, args.blocks + 1):
        order = list(conditions)
        rng.shuffle(order)
        for order_index, (input_length, batch_size) in enumerate(order, start=1):
            request_count = args.waves_per_trial * batch_size
            start = cursors[input_length]
            stop = start + request_count
            selected = prompts[str(input_length)][start:stop]
            if len(selected) != request_count:
                raise RuntimeError(f"prompt pool exhausted for {input_length}")
            cursors[input_length] = stop
            print(
                f"[{utc_now()}] {args.model_label} block={block} input={input_length} "
                f"batch={batch_size} requests={request_count}",
                flush=True,
            )
            trial = run_trial(
                model, tokenizer, selected, input_length, args.output_length, batch_size,
                args.waves_per_trial, args.phase, block, order_index, args.gpu_indices, device,
            )
            state["trials"].append(trial)
            shared.write_state(args.output, state)
            print(
                f"  valid={trial['valid']} output_tok_s={trial['summary']['output_tok_per_s']:.3f} "
                f"p95={trial['summary']['latency_p95_s']:.3f}s errors={trial['summary']['error_count']}",
                flush=True,
            )
            if not trial["valid"]:
                state["status"] = "failed_formal_trial"
                shared.write_state(args.output, state)
                raise RuntimeError(trial["invalid_reasons"])

    state["status"] = "complete"
    state["completed_at"] = utc_now()
    state["prompt_cursors_final"] = cursors
    shared.write_state(args.output, state)


if __name__ == "__main__":
    main()
