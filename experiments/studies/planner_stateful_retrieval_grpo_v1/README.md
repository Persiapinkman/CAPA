# Planner Stateful Retrieval GRPO Study

## Objective

Find a Planner task family where GRPO has measurable room to improve beyond the fixed SFTv3 ChatML initializer. The target is state-conditioned retrieval planning rather than JSON formatting.

## Why This Scenario

The previous focused dataset saturated sampled rewards: mean `frac_reward_zero_std` was 0.658 and 27 of 60 optimizer steps had zero gradient. Clarify cases had the opposite problem because the desired action was not sampled. This study uses one- to three-step trajectories where the model already knows all actions but must condition on history and observations.

The five balanced scenario families are:

- Resolve a pronoun with `re_question`, then call `rag_answer`.
- On a mocked RAG miss, rewrite once and retry RAG.
- End with `memory_hit` when the prior result is already sufficient.
- Call RAG directly when no rewrite is needed.
- Use `answerer` for general questions to prevent unconditional retrieval loops.

## Design

Entities and wording templates are disjoint across train, dev, and test. The test split is not used for support auditing or hyperparameter selection. Confidence intervals must cluster by `entity_id`, because each entity contributes all five scenario families.

Training starts only if the development sampling audit passes all support thresholds in `study.json`. The support gate applies to the two learnable state-transition families; saturation in the three guardrail families is expected and is monitored for regression after training. A single seed is screened on dev at step 40. If the development gate passes, step 40 is locked; otherwise the same run continues to step 80 for one final screen. Hyperparameters are then locked and repeated with seeds 42, 43, and 44 before the test split is opened.

## Decision Rule

Development promotion requires at least +0.02 case-macro score and no category regression larger than 0.02. Final support requires a positive mean test delta whose entity-clustered 95% interval excludes zero, plus guardrail preservation.

This is a scoped causal comparison against the fixed SFTv3 initializer. It does not establish that GRPO is generally superior to SFT or DPO.

## Execution Amendments

- The first dynamic-prompt screen was stopped after five steps because independently generated ledger event IDs changed prompt tokenization across DDP ranks. It produced no checkpoint and is excluded from all comparisons.
- All valid training runs read the registered `train.jsonl` step artifact directly, so prompts are byte-identical across ranks and training seeds.
- The support saturation gate is computed on the two target transition families. Saturation in direct RAG and memory reuse is intentional; those categories remain regression guardrails.
- The first attempt to resume checkpoint 40 hit a PEFT 0.19.1 / Transformers 4.57.6 incompatibility: PEFT imported a tensor-parallel-only class during ordinary DDP adapter loading. A local compatibility guard now skips that import only when no tensor-parallel layer exists. Exact resume was verified by starting at step 41 with the preserved optimizer schedule (`1e-6`).

## Development Screens

- Step 40 failed: case-macro delta `-0.002536`, with no guardrail regression. The dense coreference category increased `+0.016304`, but strict coreference step-1 action match remained `0/8`; this was parameter-level partial credit, not a routing improvement.
- Step 80 failed the preregistered `+0.02` case-macro gate: observed delta `+0.005797`, entity-clustered 95% interval `[0.000000, 0.013406]`. Strict overall action match increased by one of 64 steps, while coreference step-1 remained `0/8`. No evaluated step regressed and all three guardrail categories were unchanged.
- Because step 80 failed, the preregistered `coref_contrast_v1` conditional arm was activated. It uses fourfold exposure for the supported coreference first step and onefold replay of four anti-shortcut/guardrail decisions. Its gate requires both a `+0.05` coreference verifier delta and a `+0.25` strict step-1 action-match delta.
- The coreference-focused arm also failed: strict target action remained `0/8`, dense coreference delta was only `+0.016304`, and RAG-miss regressed `-0.041667`. It was stopped after the locked 40 steps; no confirmation seeds were launched.
- A second exploratory arm, `rag_miss_state_machine_v1`, was registered before training. This choice follows the only observed strict gain in the mixed arm (RAG-miss step 1, `2/8` to `3/8`) and reinforces all three retrieve-rewrite-retrieve states while replaying four contrast categories.
- The dense RAG-miss arm produced no greedy change. Its paired temperature-0.7 audit increased dense verifier mean from `0.8325` to `0.8400`, but strict sampled actions decreased from `147/192` to `146/192`; this established reward misalignment rather than task learning.
- The final v1 arm capped wrong-action task reward at `0.20` and raised action-match weight to `0.75`. It had 32 informative reward steps out of 40, yet greedy RAG-miss score/action deltas were both zero and sampled actions returned exactly to the SFT baseline (`147/192`). The arm failed and seeds 43/44 were not run.
- Both focused 40-step adapters had effective LoRA `BA` norms near `0.012`. The next registered study is `planner_complex_retrieval_grpo_v2`, which replaces duplicated weighting with 1080 unique training states and preregisters a larger 80-step, `1e-5` update dose on a five-step recovery task.
