# Qwen3.5-4B Full-Parameter GRPO FSDP Recovery v4

## Why v3 OOMed

v3 used physical GPUs `4,5,6,7`, which were empty at launch. The traceback reports `GPU 0` and `GPU 2` because CUDA remaps visible devices to local ranks. With `CUDA_VISIBLE_DEVICES=4,5,6,7`, local `GPU 0` is physical GPU 4 and local `GPU 2` is physical GPU 6.

The OOM was still caused by this training job's own online generation memory peak, not by a foreign process on physical GPU 0/1/2:

- failure stage: TRL GRPO `generate()` prefill
- model path: Qwen3.5 linear attention `torch_chunk_gated_delta_rule`
- requested allocation: 554 MiB
- free memory on the local failing rank: 316-420 MiB

## Recovery Strategy

- Use only currently empty physical GPUs: `3,4,5,6`.
- Resume from v3 `checkpoint-2`, preserving model, optimizer, scheduler, scaler, RNG, and trainer state.
- Reduce `max_completion_length` from 192 to 160 to give generation and linear-attention kernels more headroom.
- Keep `num_generations=2` and `gradient_accumulation_steps=16`.
- Keep `max_steps=30`, so resume should continue from global step 2.
- Keep `save_steps=2` for progress protection.
