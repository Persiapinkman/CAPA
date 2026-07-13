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
ChatML SFT warmup -> merge SFT adapter -> GRPO LoRA with ChatML prompts
and combined task/format reward -> held-out eval -> PPO candidate
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
| 2026-07-10 | `qwen25_trl_sft_lora_warmup_v3_chatml` | SFT LoRA warmup | `outputs/planner-sft-qwen25-7b-trl-lora-warmup-v3-chatml` | completed | ChatML prompt, compact JSON, explicit `<|im_end|>` completion target; fixes raw extra text. |
| 2026-07-10 | `qwen25_trl_grpo_lora_v4_chatml_format_smoke` | GRPO LoRA smoke | `outputs/planner-grpo-qwen25-7b-trl-lora-v4-chatml-format-smoke` | completed | Merged SFTv3 init + ChatML rollout + task/format reward; clipped_ratio 0.0 for most observed steps. |
| 2026-07-11 | `qwen25_trl_grpo_lora_v4_chatml_format_formal_v1` | GRPO LoRA train | `outputs/planner-grpo-qwen25-7b-trl-lora-v4-chatml-format-formal-v1` | completed | 60 GRPO steps on train-case split only; clipped_ratio stayed 0.0 through the final step. |
| 2026-07-12 | `qwen25_trl_sft_lora_v4_hard_chatml` | SFT LoRA diagnostic | `outputs/planner-sft-qwen25-7b-trl-lora-v4-hard-chatml` | rejected | Hard-only refresh from merged SFTv3; did not fix clarify and regressed held-out score/stopping. |
| 2026-07-12 | `qwen25_trl_sft_lora_v5_mixed_hard_chatml` | SFT LoRA diagnostic | `outputs/planner-sft-qwen25-7b-trl-lora-v5-mixed-hard-chatml` | rejected | Mixed SFTv3 train + train-only hard augmentation; still did not fix clarify, so not used as GRPO initializer. |

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
| 2026-07-10 | `qwen25_sft_v3_chatml_sft_val_firstjson_eval` | SFT v3 ChatML | `sft_data_v3_chatml/val.jsonl` 49 | first-JSON mean verifier score | 0.8564 | JSON valid 1.0; raw extra text rate 0.0; stopping fixed at SFT stage. |
| 2026-07-10 | `qwen25_grpo_v4_chatml_format_smoke_sft_val_firstjson_eval` | GRPO v4 smoke | `sft_data_v3_chatml/val.jsonl` 49 | first-JSON mean verifier score | 0.8578 | JSON valid 1.0; raw extra text rate 0.0; no stopping regression after GRPO smoke. |
| 2026-07-11 | `qwen25_grpo_v4_chatml_format_formal_v1_sft_val_firstjson_eval` | GRPO v4 formal-v1 | `sft_data_v3_chatml/val.jsonl` 49 | first-JSON mean verifier score | 0.8597 | JSON valid 1.0; raw extra text rate 0.0; +0.0033 over SFTv3 on held-out val split. |
| 2026-07-12 | `qwen25_sft_v4_hard_chatml_sft_val_firstjson_eval` | SFT v4 hard diagnostic | `sft_data_v3_chatml/val.jsonl` 49 | first-JSON mean verifier score | 0.8407 | Rejected: clarify stayed 0.1, full_detection_eval 0.75, raw extra text rate regressed to 0.1224. |
| 2026-07-12 | `qwen25_sft_v5_mixed_hard_chatml_sft_val_firstjson_eval` | SFT v5 mixed hard diagnostic | `sft_data_v3_chatml/val.jsonl` 49 | first-JSON mean verifier score | 0.8457 | Rejected: clarify stayed 0.1, full_detection_eval 0.75, raw extra text rate regressed to 0.1020. |

Three-way comparison:

- `training/planner_grpo_seed_v1/reports/qwen25_baseline_vs_sft_v1_vs_sft_v2_sft_val_compare.md`
- `training/planner_grpo_seed_v1/reports/qwen25_baseline_vs_sft_v1_vs_sft_v2_firstjson_compare.md`
- `training/planner_grpo_seed_v1/reports/qwen25_sft_v3_vs_grpo_v4_chatml_compare.md`
- `training/planner_grpo_seed_v1/reports/qwen25_sft_v3_vs_grpo_v4_formal_v1_chatml_compare.md`

## Current Caveats

- `training/planner_grpo_seed_v1/cases/planner_grpo_compound245_eval_cases.jsonl`
  remains useful as regression eval, not as a clean held-out generalization set.
- First-complete-JSON postprocessing fixes execution/scoring tail pollution, but
  does not make the model stop. Under this effective scoring, SFT v1 is slightly
  best, baseline and SFT v2 are close.
- Root cause of the stopping failure was prompt/template mismatch plus missing
  explicit ChatML EOS: old data used pseudo `<|system|>/<|user|>/<|assistant|>`
  tags, while Qwen2.5-Instruct expects `<|im_start|>...<|im_end|>`.
- Multi-GPU GRPO continuation from an SFT adapter still fails in PEFT with
  `EmbeddingParallel` import mismatch. Single-GPU adapter loading works, so this
  is a distributed PEFT/Transformers compatibility issue. Current workaround:
  merge the SFT adapter once, then use the merged model as GRPO base with a new
  LoRA adapter.
- Long GRPO with first-JSON reward alone is not a formal training path. Correct
  GRPO uses ChatML prompts and a combined task/format reward. The format reward
  penalizes prefix text, tail text, and max-length clipping while task reward
  scores the first complete JSON decision.
- GRPOv4 formal-v1 remains the current best held-out checkpoint. Error audit
  shows the unresolved slice is not a simple "more steps" problem:
  `clarify_intent_ambiguity` remains at 0.1 because generated samples collapse
  to `flux-image-generation`, leaving GRPO with little/no positive within-group
  advantage signal. Hard-only and mixed hard SFT attempts on 2026-07-12 did not
  repair this and also reintroduced raw tail text.
- Do not start GRPO from SFTv4-hard or SFTv5-mixed-hard. The next reasonable
  GRPO iteration should either change the ambiguity task definition/prompt
  boundary or use a preference dataset that guarantees contrastive positive and
  negative completions for clarify, rather than hoping online sampling discovers
  rare `decision_type="clarify"` outputs.
