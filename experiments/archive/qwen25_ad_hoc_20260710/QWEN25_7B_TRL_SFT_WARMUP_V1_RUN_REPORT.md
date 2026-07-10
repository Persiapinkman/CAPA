# Qwen2.5-7B TRL SFT LoRA Warmup v1 Run Report

Date: 2026-07-10

## Result

已完成一版 SFT warmup adapter，用于后续 GRPO formal-v2 的初始化候选。

- Output: `outputs/planner-sft-qwen25-7b-trl-lora-warmup-v1`
- Final adapter: `outputs/planner-sft-qwen25-7b-trl-lora-warmup-v1/adapter_model.safetensors`
- Final checkpoint: `outputs/planner-sft-qwen25-7b-trl-lora-warmup-v1/checkpoint-13`
- Training steps: 13
- GPUs: 4 x Tesla V100-SXM2-32GB
- Runtime: 414.8s
- Train loss: `1.4474`
- Step-10 eval loss: `1.4077`
- Step-10 eval token accuracy: `0.8163`

## Data

Data was built by:

```bash
.venv-trl-grpo-cu124/bin/python \
  training/planner_grpo_seed_v1/scripts/build_planner_sft_data.py
```

Generated files:

- `training/planner_grpo_seed_v1/sft_data/train.jsonl`
- `training/planner_grpo_seed_v1/sft_data/val.jsonl`
- `training/planner_grpo_seed_v1/sft_data/metadata.json`

Split:

- Train: 123 cases / 196 step samples
- Val: 31 cases / 49 step samples
- Split is grouped by `case_id`, so no case leaks across train and val.

Each row contains:

- `prompt`: current planner prompt ending at the assistant turn
- `completion`: one canonical planner JSON decision
- reward/eval metadata: `expected_step`, `forbidden_actions`, `reward_spec`,
  `previous_action`, `full_expected_actions`

TRL SFT uses `completion_only_loss=True`, so prompt tokens do not contribute to
loss.

## Command

Reusable script:

```bash
bash scripts/run_qwen25_7b_trl_sft_lora.sh
```

Actual warmup v1 command used the same defaults with:

- `NUM_PROCESSES=4`
- `NUM_TRAIN_EPOCHS=1`
- `GRAD_ACCUM_STEPS=4`
- `MAX_LENGTH=5120`
- `LR=1e-5`
- LoRA `r=16`, `alpha=32`, dropout `0.05`
- `attn_implementation=sdpa`

## Generation Smoke

Adapter generation smoke on the first 8 val samples:

- Report: `training/planner_grpo_seed_v1/reports/sft_warmup_v1_val8_eval.json`
- Predictions: `training/planner_grpo_seed_v1/reports/sft_warmup_v1_val8_predictions.jsonl`
- Mean verifier score: `0.3155`
- JSON valid rate: `1.0`
- Extra text after first JSON rate: `1.0`
- Mean extra text after first JSON: `188.75` chars

Interpretation:

- SFT warmup v1 improved first-JSON validity, but did not solve stopping.
- The model often emits a correct first JSON and then continues with prompt-like
  `<|system|>` / `<|user|>` text.
- This adapter is useful as a format warmup baseline, but should not yet be
  treated as the final GRPO initializer until stopping is improved.

## Next Fix

Before GRPO formal-v2, improve SFT data/prompting for termination:

1. Try target completion with explicit Qwen chat EOS behavior and verify whether
   generation stops at `<|im_end|>`.
2. Add a stop-aware generation wrapper for evaluation and GRPO rollout if TRL
   regular generation cannot stop on first complete JSON.
3. Consider a stronger SFT warmup with duplicated stop-critical examples and
   shorter canonical completions.
4. Keep `sdpa` attention and fp16 invalid-logit safeguards.
