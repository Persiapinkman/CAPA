#!/usr/bin/env python3
"""Run a fixed-length OpenAI-completions throughput benchmark with GPU telemetry."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import random
import shlex
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_conditions(raw: str) -> list[tuple[int, int]]:
    conditions: list[tuple[int, int]] = []
    for item in raw.split(","):
        input_length, concurrency = item.split(":", maxsplit=1)
        condition = (int(input_length), int(concurrency))
        if condition[0] <= 0 or condition[1] <= 0:
            raise ValueError(f"invalid condition: {item}")
        conditions.append(condition)
    if len(conditions) != len(set(conditions)):
        raise ValueError("conditions must be unique")
    return conditions


async def get_json(session: aiohttp.ClientSession, url: str) -> dict[str, Any]:
    async with session.get(url) as response:
        body = await response.text()
        if response.status != 200:
            raise RuntimeError(f"GET {url} returned {response.status}: {body[:500]}")
        return json.loads(body)


async def one_completion(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    prompt_token_ids: list[int],
    output_length: int,
    request_id: str,
    seed: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt_token_ids,
        "temperature": 0.0,
        "max_tokens": output_length,
        "min_tokens": output_length,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "seed": seed,
    }
    started = time.perf_counter()
    first_content_at: float | None = None
    first_byte_at: float | None = None
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    text_parts: list[str] = []
    chunks = 0
    status: int | None = None

    try:
        async with session.post(
            f"{endpoint.rstrip('/')}/v1/completions",
            json=payload,
            headers={"X-Request-ID": request_id},
        ) as response:
            status = response.status
            if response.status != 200:
                body = await response.text()
                raise RuntimeError(f"HTTP {response.status}: {body[:1000]}")
            async for raw_line in response.content:
                now = time.perf_counter()
                if first_byte_at is None and raw_line:
                    first_byte_at = now
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                event = json.loads(data)
                if event.get("usage") is not None:
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                content = choice.get("text") or ""
                if content:
                    chunks += 1
                    text_parts.append(content)
                    if first_content_at is None:
                        first_content_at = now
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
        ended = time.perf_counter()
        if first_content_at is None:
            raise RuntimeError("stream completed without a non-empty content chunk")
        if usage is None:
            raise RuntimeError("stream completed without server usage statistics")
        completion_tokens = int(usage.get("completion_tokens", -1))
        prompt_tokens = int(usage.get("prompt_tokens", -1))
        latency = ended - started
        ttft = first_content_at - started
        tpot = (latency - ttft) / max(completion_tokens - 1, 1)
        return {
            "request_id": request_id,
            "success": True,
            "http_status": status,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": int(usage.get("total_tokens", prompt_tokens + completion_tokens)),
            "latency_s": latency,
            "ttfb_s": None if first_byte_at is None else first_byte_at - started,
            "ttft_s": ttft,
            "tpot_s": tpot,
            "stream_content_chunks": chunks,
            "finish_reason": finish_reason,
            "output_chars": sum(len(part) for part in text_parts),
            "output_sha256": hashlib.sha256("".join(text_parts).encode("utf-8")).hexdigest(),
            "error": None,
        }
    except Exception as exc:  # Preserve every failed request in the raw record.
        ended = time.perf_counter()
        return {
            "request_id": request_id,
            "success": False,
            "http_status": status,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "latency_s": ended - started,
            "ttfb_s": None if first_byte_at is None else first_byte_at - started,
            "ttft_s": None if first_content_at is None else first_content_at - started,
            "tpot_s": None,
            "stream_content_chunks": chunks,
            "finish_reason": finish_reason,
            "output_chars": sum(len(part) for part in text_parts),
            "output_sha256": hashlib.sha256("".join(text_parts).encode("utf-8")).hexdigest(),
            "error": f"{type(exc).__name__}: {exc}",
        }


async def query_gpu(indices: list[int], relative_s: float) -> list[dict[str, Any]]:
    query = "index,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm"
    process = await asyncio.create_subprocess_exec(
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
        "-i",
        ",".join(str(index) for index in indices),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(stderr.decode("utf-8", errors="replace").strip())
    rows: list[dict[str, Any]] = []
    fields = ["gpu_index", "utilization_pct", "memory_used_mib", "power_w", "temperature_c", "sm_clock_mhz"]
    for line in stdout.decode().splitlines():
        values = [part.strip() for part in line.split(",")]
        if len(values) != len(fields):
            continue
        parsed: dict[str, Any] = {"relative_s": relative_s}
        for field, value in zip(fields, values):
            try:
                parsed[field] = int(float(value)) if field in {"gpu_index", "temperature_c", "sm_clock_mhz"} else float(value)
            except ValueError:
                parsed[field] = None
        rows.append(parsed)
    return rows


async def collect_telemetry(
    gpu_indices: list[int],
    trial_started: float,
    stop: asyncio.Event,
    destination: list[dict[str, Any]],
    errors: list[str],
    interval_s: float = 0.5,
) -> None:
    while not stop.is_set():
        try:
            destination.extend(await query_gpu(gpu_indices, time.perf_counter() - trial_started))
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass


def summarize_telemetry(rows: list[dict[str, Any]], gpu_indices: list[int], duration_s: float) -> dict[str, Any]:
    per_gpu: dict[str, Any] = {}
    total_average_power = 0.0
    usable_power = True
    for index in gpu_indices:
        gpu_rows = [row for row in rows if row.get("gpu_index") == index]
        util = [row["utilization_pct"] for row in gpu_rows if row.get("utilization_pct") is not None]
        memory = [row["memory_used_mib"] for row in gpu_rows if row.get("memory_used_mib") is not None]
        power = [row["power_w"] for row in gpu_rows if row.get("power_w") is not None]
        temperature = [row["temperature_c"] for row in gpu_rows if row.get("temperature_c") is not None]
        average_power = statistics.fmean(power) if power else None
        if average_power is None:
            usable_power = False
        else:
            total_average_power += average_power
        per_gpu[str(index)] = {
            "samples": len(gpu_rows),
            "utilization_mean_pct": statistics.fmean(util) if util else None,
            "utilization_peak_pct": max(util) if util else None,
            "memory_mean_mib": statistics.fmean(memory) if memory else None,
            "memory_peak_mib": max(memory) if memory else None,
            "power_mean_w": average_power,
            "power_peak_w": max(power) if power else None,
            "temperature_peak_c": max(temperature) if temperature else None,
        }
    return {
        "sample_rows": len(rows),
        "per_gpu": per_gpu,
        "service_power_mean_w": total_average_power if usable_power else None,
        "estimated_energy_wh": total_average_power * duration_s / 3600 if usable_power else None,
    }


def summarize_requests(requests: list[dict[str, Any]], duration_s: float) -> dict[str, Any]:
    successes = [row for row in requests if row["success"]]
    prompt_tokens = sum(int(row["prompt_tokens"]) for row in successes)
    completion_tokens = sum(int(row["completion_tokens"]) for row in successes)
    latencies = [float(row["latency_s"]) for row in successes]
    ttfts = [float(row["ttft_s"]) for row in successes]
    tpots = [float(row["tpot_s"]) for row in successes]
    return {
        "request_count": len(requests),
        "success_count": len(successes),
        "error_count": len(requests) - len(successes),
        "duration_s": duration_s,
        "request_per_s": len(successes) / duration_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prompt_tok_per_s": prompt_tokens / duration_s,
        "output_tok_per_s": completion_tokens / duration_s,
        "total_tok_per_s": (prompt_tokens + completion_tokens) / duration_s,
        "latency_mean_s": statistics.fmean(latencies) if latencies else None,
        "latency_p50_s": percentile(latencies, 0.50),
        "latency_p95_s": percentile(latencies, 0.95),
        "ttft_p50_s": percentile(ttfts, 0.50),
        "ttft_p95_s": percentile(ttfts, 0.95),
        "tpot_p50_s": percentile(tpots, 0.50),
        "tpot_p95_s": percentile(tpots, 0.95),
    }


def validate_trial(
    requests: list[dict[str, Any]],
    expected_requests: int,
    input_length: int,
    output_length: int,
    telemetry: dict[str, Any],
    gpu_count: int,
) -> list[str]:
    reasons: list[str] = []
    if len(requests) != expected_requests:
        reasons.append(f"request count {len(requests)} != {expected_requests}")
    for request in requests:
        request_id = request["request_id"]
        if not request["success"]:
            reasons.append(f"{request_id}: {request['error']}")
            continue
        if request["prompt_tokens"] != input_length:
            reasons.append(f"{request_id}: prompt_tokens {request['prompt_tokens']} != {input_length}")
        if request["completion_tokens"] != output_length:
            reasons.append(f"{request_id}: completion_tokens {request['completion_tokens']} != {output_length}")
        if request["finish_reason"] != "length":
            reasons.append(f"{request_id}: finish_reason {request['finish_reason']!r} != 'length'")
        for metric in ("latency_s", "ttft_s", "tpot_s"):
            value = request.get(metric)
            if value is None or not math.isfinite(float(value)):
                reasons.append(f"{request_id}: non-finite {metric}")
    if telemetry["sample_rows"] < 2 * gpu_count:
        reasons.append(f"only {telemetry['sample_rows']} telemetry rows for {gpu_count} GPUs")
    return reasons


async def run_trial(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    prompt_rows: list[list[int]],
    input_length: int,
    output_length: int,
    concurrency: int,
    phase: str,
    block: int,
    order_index: int,
    gpu_indices: list[int],
    seed: int,
) -> dict[str, Any]:
    semaphore = asyncio.Semaphore(concurrency)
    trial_started_at = utc_now()
    trial_started = time.perf_counter()
    telemetry_rows: list[dict[str, Any]] = []
    telemetry_errors: list[str] = []
    telemetry_stop = asyncio.Event()
    telemetry_task = asyncio.create_task(
        collect_telemetry(gpu_indices, trial_started, telemetry_stop, telemetry_rows, telemetry_errors)
    )

    async def limited_request(index: int, prompt_ids: list[int]) -> dict[str, Any]:
        async with semaphore:
            request_id = f"{phase}-b{block:02d}-o{order_index:02d}-i{input_length}-c{concurrency}-r{index:03d}"
            return await one_completion(
                session=session,
                endpoint=endpoint,
                model=model,
                prompt_token_ids=prompt_ids,
                output_length=output_length,
                request_id=request_id,
                seed=seed + block * 10_000 + order_index * 1_000 + index,
            )

    requests = await asyncio.gather(
        *(limited_request(index, prompt_ids) for index, prompt_ids in enumerate(prompt_rows))
    )
    duration_s = time.perf_counter() - trial_started
    telemetry_stop.set()
    await telemetry_task
    telemetry = summarize_telemetry(telemetry_rows, gpu_indices, duration_s)
    summary = summarize_requests(requests, duration_s)
    invalid_reasons = validate_trial(
        requests=requests,
        expected_requests=len(prompt_rows),
        input_length=input_length,
        output_length=output_length,
        telemetry=telemetry,
        gpu_count=len(gpu_indices),
    )
    return {
        "phase": phase,
        "block": block,
        "condition_order_index": order_index,
        "input_length": input_length,
        "output_length": output_length,
        "concurrency": concurrency,
        "request_count_planned": len(prompt_rows),
        "started_at": trial_started_at,
        "ended_at": utc_now(),
        "valid": not invalid_reasons,
        "invalid_reasons": invalid_reasons,
        "summary": summary,
        "telemetry_summary": telemetry,
        "telemetry_errors": telemetry_errors,
        "requests": requests,
        "telemetry": telemetry_rows,
    }


def write_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


async def async_main(args: argparse.Namespace) -> None:
    conditions = parse_conditions(args.conditions)
    prompt_pool = json.loads(args.prompt_pool.read_text())
    prompts: dict[str, list[list[int]]] = prompt_pool["prompts"]
    for input_length, _ in conditions:
        if str(input_length) not in prompts:
            raise ValueError(f"prompt pool does not contain input length {input_length}")

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}; pass --overwrite explicitly")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        models = await get_json(session, f"{args.endpoint.rstrip('/')}/v1/models")
        try:
            version = await get_json(session, f"{args.endpoint.rstrip('/')}/version")
        except Exception as exc:
            version = {"unavailable": f"{type(exc).__name__}: {exc}"}

        state: dict[str, Any] = {
            "schema_version": 1,
            "study_id": "qwen3_4b_vs_32b_throughput_v1",
            "run_id": args.run_id,
            "phase": args.phase,
            "created_at": utc_now(),
            "completed_at": None,
            "status": "running",
            "model_label": args.model_label,
            "served_model_name": args.model,
            "endpoint": args.endpoint,
            "gpu_indices": args.gpu_indices,
            "gpu_count": len(args.gpu_indices),
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
            "server_models_response": models,
            "server_version_response": version,
            "calibration": None,
            "warmups": [],
            "trials": [],
        }
        write_state(args.output, state)

        cursors = {input_length: 0 for input_length, _ in conditions}

        calibration_length = conditions[0][0]
        calibration_prompt = prompts[str(calibration_length)][cursors[calibration_length]]
        cursors[calibration_length] += 1
        calibration = await one_completion(
            session=session,
            endpoint=args.endpoint,
            model=args.model,
            prompt_token_ids=calibration_prompt,
            output_length=8,
            request_id=f"{args.phase}-calibration",
            seed=args.seed,
        )
        state["calibration"] = calibration
        if not calibration["success"]:
            state["status"] = "failed_calibration"
            write_state(args.output, state)
            raise RuntimeError(f"calibration request failed: {calibration['error']}")
        if calibration["prompt_tokens"] != calibration_length or calibration["completion_tokens"] != 8:
            state["status"] = "failed_calibration"
            write_state(args.output, state)
            raise RuntimeError(f"calibration token mismatch: {calibration}")

        if not args.skip_warmup:
            for warmup_index, (input_length, concurrency) in enumerate(conditions):
                request_count = concurrency
                start = cursors[input_length]
                stop = start + request_count
                selected = prompts[str(input_length)][start:stop]
                if len(selected) != request_count:
                    raise RuntimeError(f"prompt pool exhausted during warmup for length {input_length}")
                cursors[input_length] = stop
                warmup = await run_trial(
                    session=session,
                    endpoint=args.endpoint,
                    model=args.model,
                    prompt_rows=selected,
                    input_length=input_length,
                    output_length=32,
                    concurrency=concurrency,
                    phase="warmup",
                    block=0,
                    order_index=warmup_index,
                    gpu_indices=args.gpu_indices,
                    seed=args.seed,
                )
                state["warmups"].append(warmup)
                write_state(args.output, state)
                if not warmup["valid"]:
                    state["status"] = "failed_warmup"
                    write_state(args.output, state)
                    raise RuntimeError(f"warmup invalid: {warmup['invalid_reasons']}")

        rng = random.Random(args.seed)
        for block in range(1, args.blocks + 1):
            order = list(conditions)
            rng.shuffle(order)
            for order_index, (input_length, concurrency) in enumerate(order, start=1):
                request_count = args.waves_per_trial * concurrency
                start = cursors[input_length]
                stop = start + request_count
                selected = prompts[str(input_length)][start:stop]
                if len(selected) != request_count:
                    raise RuntimeError(
                        f"prompt pool exhausted for length {input_length}: need through {stop}, have {len(prompts[str(input_length)])}"
                    )
                cursors[input_length] = stop
                print(
                    f"[{utc_now()}] {args.model_label} block={block} input={input_length} "
                    f"concurrency={concurrency} requests={request_count}",
                    flush=True,
                )
                trial = await run_trial(
                    session=session,
                    endpoint=args.endpoint,
                    model=args.model,
                    prompt_rows=selected,
                    input_length=input_length,
                    output_length=args.output_length,
                    concurrency=concurrency,
                    phase=args.phase,
                    block=block,
                    order_index=order_index,
                    gpu_indices=args.gpu_indices,
                    seed=args.seed,
                )
                state["trials"].append(trial)
                write_state(args.output, state)
                summary = trial["summary"]
                print(
                    f"  valid={trial['valid']} output_tok_s={summary['output_tok_per_s']:.3f} "
                    f"p95={summary['latency_p95_s']:.3f}s errors={summary['error_count']}",
                    flush=True,
                )
                if not trial["valid"]:
                    state["status"] = "failed_formal_trial"
                    write_state(args.output, state)
                    raise RuntimeError(f"formal trial invalid: {trial['invalid_reasons']}")

        state["status"] = "complete"
        state["completed_at"] = utc_now()
        state["prompt_cursors_final"] = cursors
        write_state(args.output, state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:18080")
    parser.add_argument("--model", required=True, help="Served model name")
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--prompt-pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--phase", default="formal")
    parser.add_argument("--gpu-indices", type=int, nargs="+", required=True)
    parser.add_argument("--conditions", default="256:1,256:4,256:16,2048:1,2048:4,2048:16")
    parser.add_argument("--output-length", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--waves-per-trial", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
