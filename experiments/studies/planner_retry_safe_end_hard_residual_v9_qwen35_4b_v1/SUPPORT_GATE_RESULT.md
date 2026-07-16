# V9 hard-residual support gate result

Status: **PASS — optimizer authorized**

The Qwen3.5-4B V6 SFT checkpoint-100 was sampled on 144 entity-disjoint V9 support prompts with four completions per prompt at the preregistered `temperature=0.9`, `top_p=0.9`, and `max_new_tokens=320`.

| Gate | Observed | Threshold | Result |
| --- | ---: | ---: | :---: |
| Complete prompt groups | 144/144 | 144 | Pass |
| Complete samples | 576/576 | 576 | Pass |
| JSON valid | 100% | ≥99% | Pass |
| Clipped | 0% | ≤1% | Pass |
| Primary gold-action support | 80.56% | ≥70% | Pass |
| Primary nonzero reward variance | 36.11% (52/144) | ≥20% | Pass |
| Support block A gold / variance | 84.72% / 31.94% | ≥60% / ≥15% | Pass |
| Support block B gold / variance | 76.39% / 40.28% | ≥60% / ≥15% | Pass |

Every one of the six scenario × detector strata also passed the preregistered `gold ≥50%` and `nonzero-variance groups ≥3` gates. The weakest observed gold stratum was `post_retry_success_step3 × Rex` at 75%; the weakest variance strata were `fresh_retry_step2 × Qwen` at 5/24 and `current_success_step2 × Rex` plus `post_retry_success_step3 × Rex` at 6/24.

The all-or-none optimizer scope is therefore frozen to:

- `current_success_step2`
- `fresh_retry_step2`
- `post_retry_success_step3`

The optimizer artifact contains exactly 144 rows over 24 train entities: 48 rows per scenario, 72 rows per detector, 96 step-2 rows, and 48 step-3 rows. Its SHA-256 is `a3f5e5f0cadab6ac3ceba9cba42b1570c18bb45a53702c45f1b12744ebcfe47b`.

Evidence:

- Support decision: `support_decision.json` (`13ff5023eb97c8e35020d5e37c704aab80d4a07e4ddb5513d60e4e50455eee81`)
- Combined samples: `experiments/runs/20260716_qwen35_4b_v9_hard_residual_support4x_sft100/samples.jsonl` (`44fbc5879d0a435283ef52ed97e83624bb4b7773ba63c2a4815167ffcf1b7101`)
- Sealed-test commitment remains unopened: `7119b1a06ef528b93c8d400f4e17c272eb1379969517b76dc6c155782a338113`

The next allowed action is the five-step GRPO canary. Checkpoint promotion and the larger-model comparison remain unauthorized until the preregistered screen and selection-dev evaluation finish.
