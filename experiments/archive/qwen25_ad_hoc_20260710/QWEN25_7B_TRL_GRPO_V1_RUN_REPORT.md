# Qwen2.5-7B TRL GRPO LoRA v1 Run Report

Date: 2026-07-10

## Result

已按 TRL 路径跑通一版 Qwen2.5-7B-Instruct fp16 LoRA GRPO 训练。

- Output: `outputs/planner-grpo-qwen25-7b-trl-lora-v1`
- Final adapter: `outputs/planner-grpo-qwen25-7b-trl-lora-v1/adapter_model.safetensors`
- Final checkpoint: `outputs/planner-grpo-qwen25-7b-trl-lora-v1/checkpoint-20`
- Training steps: 20
- GPUs: 4 x Tesla V100-SXM2-32GB
- Runtime: 618.4s
- Train samples/sec: 0.259
- Train steps/sec: 0.032
- Adapter smoke report: `outputs/planner-grpo-qwen25-7b-trl-lora-v1/adapter_smoke_predictions.jsonl`

Adapter smoke used 6 representative step prompts. Mean verifier score was `0.5476`.

## Environment

Clean environment was created instead of reusing the broken existing envs:

```bash
uv venv .venv-trl-grpo-cu124 --python 3.10
uv pip install --python .venv-trl-grpo-cu124/bin/python \
  --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0
uv pip install --python .venv-trl-grpo-cu124/bin/python \
  'trl==1.8.0' 'transformers>=4.57,<5' 'accelerate>=1.10' \
  'peft>=0.17' 'datasets>=2.21' 'tensorboard' openai
```

Resolved core versions:

- `torch==2.6.0+cu124`
- `trl==1.8.0`
- `transformers==4.57.6`
- `accelerate==1.14.0`
- `peft==0.19.1`
- `datasets==5.0.0`

## Reusable Command

Default rerun:

```bash
bash scripts/run_qwen25_7b_trl_grpo_lora.sh
```

The script defaults to:

- model: `/raid/zkq/models/Qwen2.5-7B-Instruct`
- cases: `training/planner_grpo_seed_v1/cases/planner_grpo_focused_4b_cases.jsonl`
- output: `outputs/planner-grpo-qwen25-7b-trl-lora-v1`
- 4 GPUs, `num_generations=2`, `generation_batch_size=8`
- LoRA `r=16`, `alpha=32`, dropout `0.05`
- `attn_implementation=sdpa`
- `remove_invalid_values=true`, `renormalize_logits=true`
- `mask_truncated_completions=false`

## What Was Learned

1. `attn_implementation=eager` is not acceptable for Qwen2.5-7B on this host.
   It produced corrupted text even for a short JSON prompt. Default attention and
   explicit `sdpa` generated normal text. The TRL training script now defaults
   to `sdpa`.

2. V100 fp16 sampling can hit invalid probability tensors. Adding
   `remove_invalid_values=true` and `renormalize_logits=true` stabilized
   generation. This is a generation configuration, not a local TRL/vLLM patch.

3. `mask_truncated_completions=true` made the first smoke run produce no useful
   loss signal because all completions reached `max_completion_length=128`.
   For v1 it is set to `false` so high-reward truncated completions still
   contribute gradients.

4. All training prompts are long but manageable: focused data has 245 step
   samples, prompt token lengths are roughly 3993-4261 tokens, so
   `max_prompt_tokens=6144` keeps all samples.

5. This TRL route does not require vLLM. Avoiding vLLM is the right first path
   for V100 stability.

## Caveats

- This is a runnable v1 adapter, not a quality claim. The focused train set
  overlaps earlier eval assets, so do not report generalization from this run.
- Completion termination is poor: every logged completion hit 128 tokens.
  The model emits a valid first JSON in many cases, then continues. The reward
  parser scores the first JSON, but training still sees continuation tokens.
- Several GRPO groups had zero reward variance, which leads to zero-gradient
  steps. The dataset/sampling mix needs improvement before scaling.
- `loss` logs rounded to `0.0`, while `grad_norm` was often non-zero and final
  `train_loss` was `2.4959e-08`. Treat gradient/reward diagnostics as the
  useful signal for this smoke-scale run.
- Adapter smoke still failed some representative categories:
  `single_image_probe` and `clarify_intent_ambiguity` scored `0.0`.

## Next Fixes Before a Larger Run

1. Add SFT warmup on exact planner JSON format, especially "emit one JSON then
   stop", before longer GRPO.
2. Add stop handling or post-JSON EOS shaping so completions terminate instead
   of always clipping at 128 tokens.
3. Increase per-prompt diversity only where reward variance is useful. Current
   `num_generations=2` is cheap but often collapses to zero variance.
4. Build a non-overlapping held-out eval split before claiming improvement.
5. Consider category-balanced sampling so `clarify_intent_ambiguity` and
   `single_image_probe` do not get drowned out by probe-to-migration examples.
