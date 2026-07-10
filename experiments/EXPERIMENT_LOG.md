# CAPA Experiment Log

This log is the human-readable record for the current Qwen2.5-7B TRL
post-training line. Older Qwen3.5/vLLM/FSDP records were archived on
2026-07-10:

- `experiments/archive/qwen35_legacy_EXPERIMENT_LOG_20260710.md`
- `experiments/archive/qwen35_legacy_manifest_20260710.jsonl`
- `experiments/archive/qwen25_ad_hoc_20260710/`

The ad-hoc archive also contains the earlier verl-route `JISHU.md` notes and
resource snapshots captured while converging on the current TRL path.

Machine-readable progress is tracked in `experiments/manifest.jsonl`.

## Active Direction

Current path:

```text
Qwen2.5-7B-Instruct + fp16 + LoRA + TRL
SFT warmup -> first-complete-JSON rollout postprocess / stopping objective
-> GRPO formal-v2 -> held-out eval -> PPO candidate
```

Current V100-safe defaults:

- no full-parameter training
- no vLLM in the training loop
- `attn_implementation=sdpa`
- `remove_invalid_values=true`
- `renormalize_logits=true`

## Current Artifacts

| Date | Run ID | Type | Output | Status | Notes |
|---|---|---|---|---|---|
| 2026-07-10 | `qwen25_trl_grpo_lora_v1` | GRPO LoRA smoke-scale train | `outputs/planner-grpo-qwen25-7b-trl-lora-v1` | completed | 20 GRPO steps; adapter saved; runnable baseline for TRL GRPO path. |
| 2026-07-10 | `qwen25_trl_sft_lora_warmup_v1` | SFT LoRA warmup | `outputs/planner-sft-qwen25-7b-trl-lora-warmup-v1` | completed | 13 SFT steps / 1 epoch; improves first JSON validity but does not solve stopping. |
| 2026-07-10 | `qwen25_trl_sft_lora_warmup_v2` | SFT LoRA warmup | `outputs/planner-sft-qwen25-7b-trl-lora-warmup-v2` | completed | Compact JSON targets, 26 steps / 2 epochs; better score than baseline and SFT v1, still no hard stop. |
| 2026-07-10 | `qwen25_trl_grpo_lora_v2_firstjson_attempt` | GRPO LoRA diagnostic | `outputs/planner-grpo-qwen25-7b-trl-lora-v2-firstjson` | stopped | Manually stopped at 4/60 steps because all completions were clipped at 128 tokens; first-JSON reward alone does not teach stopping. |

## Evaluation Status

| Date | Run ID | Model / Adapter | Eval Set | Metric | Result | Notes |
|---|---|---|---|---|---:|---|
| 2026-07-10 | `qwen25_sft_v1_val8_smoke` | SFT v1 | `sft_data/val.jsonl` first 8 | mean verifier score | 0.3155 | JSON valid 1.0, but extra text after first JSON 1.0. |
| 2026-07-10 | `qwen25_baseline_sft_val_eval` | Baseline | `sft_data/val.jsonl` 49 | mean verifier score | 0.6407 | JSON valid 1.0; extra text after JSON 1.0; mean extra chars 160.96. |
| 2026-07-10 | `qwen25_sft_v1_sft_val_eval` | SFT v1 | `sft_data/val.jsonl` 49 | mean verifier score | 0.6173 | Worse than baseline by -0.0234; extra text still 1.0. |
| 2026-07-10 | `qwen25_sft_v2_sft_val_eval` | SFT v2 | `sft_data/val.jsonl` 49 | mean verifier score | 0.6931 | Better than baseline by +0.0524 and SFT v1 by +0.0758; extra text still 1.0, but mean extra chars reduced to 131.47. |
| 2026-07-10 | `qwen25_baseline_sft_val_firstjson_eval` | Baseline | `sft_data/val.jsonl` 49 | first-JSON mean verifier score | 0.8454 | Raw extra text rate 1.0; effective extra text rate 0.0 after first-complete-JSON postprocess. |
| 2026-07-10 | `qwen25_sft_v1_sft_val_firstjson_eval` | SFT v1 | `sft_data/val.jsonl` 49 | first-JSON mean verifier score | 0.8580 | Best first-JSON score, but raw extra text rate remains 1.0. |
| 2026-07-10 | `qwen25_sft_v2_sft_val_firstjson_eval` | SFT v2 | `sft_data/val.jsonl` 49 | first-JSON mean verifier score | 0.8495 | Slightly below SFT v1 by first-JSON score; shortest raw tail at 131.47 chars. |

Three-way comparison:

- `training/planner_grpo_seed_v1/reports/qwen25_baseline_vs_sft_v1_vs_sft_v2_sft_val_compare.md`
- `training/planner_grpo_seed_v1/reports/qwen25_baseline_vs_sft_v1_vs_sft_v2_firstjson_compare.md`

## Current Caveats

- `training/planner_grpo_seed_v1/cases/planner_grpo_compound245_eval_cases.jsonl`
  remains useful as regression eval, not as a clean held-out generalization set.
- First-complete-JSON postprocessing fixes execution/scoring tail pollution, but
  does not make the model stop. Under this effective scoring, SFT v1 is slightly
  best, baseline and SFT v2 are close.
- Multi-GPU GRPO continuation from an SFT adapter still fails in PEFT with
  `EmbeddingParallel` import mismatch. Single-GPU adapter loading works, so this
  is a distributed PEFT/Transformers compatibility issue.
- Long GRPO with first-JSON reward alone is not a formal training path: the
  diagnostic run reached clipped_ratio 1.0 for every observed step. Add an
  explicit stopping objective or rollout postprocessor before GRPO formal-v2.
