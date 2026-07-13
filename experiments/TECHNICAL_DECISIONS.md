# CAPA Planner Technical Decisions

This document records current decisions and their rationale. Run-level facts and metrics belong in `experiments/registry.jsonl`.

## Research Objective

Determine whether a Qwen2.5-7B Planner can approach the routing quality of the larger demo model under V100 constraints, while preserving valid JSON, correct stopping, parameter accuracy, and multi-step state transitions.

## Active Choices

| Decision | Choice | Rationale | Status |
|---|---|---|---|
| Research model | Qwen2.5-7B-Instruct | Fits V100 fp16 LoRA training and parallel evaluation | active |
| Precision | fp16 | V100 has no BF16 support | active |
| Attention | SDPA | Eager attention produced corrupted output on this host | active |
| Parameter update | LoRA on q/k/v/o projections | Stable memory footprint; avoids full-parameter FSDP generation peaks | active |
| Prompt format | Native Qwen ChatML | Base model tail-text failures changed from 49/49 to 0/49 versus pseudo tags | validated on dev |
| SFT warmup | SFTv3 ChatML adapter | Case-macro delta vs Base ChatML `+0.001503`, CI crosses zero | inconclusive |
| GRPO reward | Task verifier plus prefix/tail/truncation format reward | Training path is stable, but case-macro delta vs SFTv3 is `-0.001390`, CI crosses zero | no quality promotion |
| Evaluation unit | `case_id` | Prevents two-step workflows from receiving double weight | active |
| Artifact storage | `/raid/zkq/artifacts/CAPA` | Keeps checkpoints, environments, caches, and traces outside Git | active |

## Evidence Boundaries

- Native ChatML alone produces clean stopping on the reused development split; SFT is not required to explain this behavior.
- SFTv3 and GRPOv4 have stable technical paths, but neither has a supported case-macro improvement over its matched control.
- `planner_grpo_compound245_eval_cases.jsonl` is a regression suite, not a held-out test set.
- PPO is gated until a clean test set and a supported SFT/DPO/GRPO comparison exist.

The Qwen3.5-era SFT/DPO/GRPO notes are archived at `experiments/archive/2026-07-12_legacy_tracking/JISHU.md`.
