# Qwen2.5-7B TRL SFT LoRA Warmup v2 Run Report

Date: 2026-07-10

## Result

SFT v2 completed. It improves verifier score over both baseline and SFT v1 on
the same SFT validation split, but it still does not solve generation stopping.

- Output: `outputs/planner-sft-qwen25-7b-trl-lora-warmup-v2`
- Final adapter: `outputs/planner-sft-qwen25-7b-trl-lora-warmup-v2/adapter_model.safetensors`
- Data: `training/planner_grpo_seed_v1/sft_data_v2`
- Train setup: Qwen2.5-7B-Instruct, fp16, LoRA r16, TRL SFT, SDPA
- Steps: 26
- Epochs: 2
- Runtime: 830.9s
- Train loss: `1.3509`
- Final eval loss: `1.2488`
- Final eval token accuracy: `0.8252`

## Change vs v1

v1 used pretty-printed JSON completion targets.

v2 uses compact single-line JSON targets from:

```bash
.venv-trl-grpo-cu124/bin/python \
  training/planner_grpo_seed_v1/scripts/build_planner_sft_data.py \
  --output-dir training/planner_grpo_seed_v1/sft_data_v2 \
  --indent -1
```

An attempted continuation from the SFT v1 adapter was blocked by a PEFT /
Transformers compatibility issue:

```text
ImportError: cannot import name 'EmbeddingParallel' from transformers.integrations.tensor_parallel
```

To avoid an environment patch, v2 was trained from the base Qwen2.5-7B-Instruct
model using the compact targets.

## Baseline Comparison

Same validation split: `training/planner_grpo_seed_v1/sft_data/val.jsonl`.

| Run | Adapter | Mean Score | JSON Valid | Extra Text Rate | Mean Extra Chars |
|---|---|---:|---:|---:|---:|
| Baseline | none | 0.6407 | 1.000 | 1.000 | 160.96 |
| SFT v1 | `outputs/planner-sft-qwen25-7b-trl-lora-warmup-v1` | 0.6173 | 1.000 | 1.000 | 157.14 |
| SFT v2 | `outputs/planner-sft-qwen25-7b-trl-lora-warmup-v2` | 0.6931 | 1.000 | 1.000 | 131.47 |

Reports:

- Baseline: `training/planner_grpo_seed_v1/reports/qwen25_baseline_sft_val_eval.json`
- SFT v1: `training/planner_grpo_seed_v1/reports/qwen25_sft_v1_sft_val_eval.json`
- SFT v2: `training/planner_grpo_seed_v1/reports/qwen25_sft_v2_sft_val_eval.json`
- Three-way comparison:
  `training/planner_grpo_seed_v1/reports/qwen25_baseline_vs_sft_v1_vs_sft_v2_sft_val_compare.md`

## Decision

SFT v2 is a better initializer than SFT v1 by score:

- v2 vs baseline: `+0.0524`
- v2 vs v1: `+0.0758`

However, it still emits extra text after the first JSON on every validation
sample. For GRPO formal-v2, either:

1. proceed with first-JSON reward parsing and accept extra continuation noise,
   or
2. implement a generation-side stop after the first complete JSON before longer
   GRPO.

The cleaner next engineering step is a stop-aware rollout wrapper/evaluator, not
another SFT-only run.
