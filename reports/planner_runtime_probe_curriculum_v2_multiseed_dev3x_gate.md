# Runtime Probe Curriculum Multi-Seed Development Gate

## Decision

The preregistered development replication gate failed. The frozen test remains sealed and was not evaluated.

All four models produced identical outputs across three deterministic repeats (`agreement_rate=1.0`). The primary runtime-owned scenario replicated, but the no-increase side-effect guardrail failed.

## Primary Evidence

| Metric | Baseline | Three-seed result | Delta / interval | Gate |
|---|---:|---:|---:|---|
| `qwen_probe_then_migration` strict complete-case action | 0.6250 | seed42 1.0000; seed43 1.0000; seed44 0.8750 | mean delta +0.3333 | pass, minimum +0.1250 |
| Primary verifier | 0.6466 | mean 0.8614 | +0.2147 | pass, minimum +0.0500 |
| Overall step action | 0.6833 | mean 0.8167 | +0.1333, entity CI [+0.0917, +0.1778] | pass, minimum +0.1000 |
| Overall case-macro verifier | 0.6889 | mean 0.8044 | +0.1155, entity CI [+0.0938, +0.1376] | diagnostic support |
| Primary step 2 action | 1.0000 | mean 1.0000 | +0.0000 | pass |
| Probe-only contrast action | 1.0000 | mean 1.0000 | +0.0000 | pass |

## Failed Guardrail

| Arm | Wrong side-effecting actions | Pipeline action match |
|---|---:|---:|
| SFTv3 | 11 | 0.625 |
| seed42 | 11 | 0.625 |
| seed43 | 11 | 0.625 |
| seed44 | 13 | 0.375 |
| Three-seed mean delta | +0.6667 | -0.0833 |

The baseline and every seed route all eight underspecified `clarify_incomplete` cases to side-effecting `adela_cli_eval`; this pre-existing safety defect did not change. The gate failure comes from seed44 adding two pipeline errors: five of eight complete-evaluation requests route to `flux-image-generation`, versus three of eight for the baseline, seed42, and seed43.

The mean pipeline action regression (`-0.0833`) stays within the separately preregistered per-category tolerance (`0.125`), but the stricter rule requiring no mean increase in any wrong side-effecting action fails. Thresholds were not changed and no extra seed was added.

## Interpretation

This study found a reproducible narrow GRPO growth scene: explicit Qwen probe followed by migration advice. It did not produce a safely promotable whole Planner because side-effect control failed in one independent seed. The appropriate next study is a newly versioned, side-effect-constrained routing dataset with independent entities and a new sealed test, not reuse of this development or test split.

Machine-readable gate: `experiments/studies/planner_runtime_routing_grpo_v1/development_gate_runtime_probe_v2_multiseed_dev3x.json`.
