# CAPA Planner Technical Decisions

This document records current decisions and their rationale. Run-level facts and metrics belong in `experiments/registry.jsonl`.

## Research Objective

Determine whether a Qwen3.5-4B Planner can approach the routing quality of the larger demo model under V100 constraints, while preserving valid JSON, correct stopping, parameter accuracy, and multi-step state transitions. Qwen2.5-7B remains the proven-stack engineering fallback and historical control.

## Active Choices

| Decision | Choice | Rationale | Status |
|---|---|---|---|
| Research target | Qwen3.5-4B | V5 has 73.33% route versus 100% for 35B, giving supported non-saturated GRPO headroom | selected; G0-G2 passed; optimizer steps 0 |
| Engineering fallback | Qwen2.5-7B-Instruct | Existing V100 fp16/SDPA/TRL LoRA path has completed distributed runs | retained |
| Precision | fp16 | V100 has no BF16 support | active |
| Attention | SDPA for full-attention layers; native PyTorch Gated DeltaNet fallback | V100 Qwen3.5 path has no validated fast hybrid kernel | G1 passed through 4096 tokens |
| Parameter update | Hybrid LoRA on q/k/v/o plus in_proj_qkv/z/a/b/out | Covers all 8 full-attention and 24 linear-attention layers while avoiding full-parameter generation peaks | G2 passed: 152 modules, 14,376,960 params |
| Prompt format | Native Qwen3.5 non-thinking ChatML | Training and inference must include the tokenizer-rendered empty think block | 480 target-step prompts frozen; max 4319 tokens |
| Completion budget | 320-token candidate rejected by full support audit | The one-case probe ended by 269, but 25/640 full-pool samples clipped at 320 (3.91% versus the 1% gate) | 128 rejected; 320 rejected; new full-pool probe required |
| Distributed topology | 8-rank primary; 4-rank shared-host fallback with generation batch 4 and grad accumulation 8 | Both keep G=4 across distinct ranks, local generation batch 1, and 32 completions/8 groups per optimizer step | fallback dry-run passed; G3/G4 not started because data gate failed |
| Initializer | Disjoint minimal SFT initializer or rebuilt retry-boundary dataset; raw-base GRPO is blocked for train-v1 | Frozen 80×8 audit found migrate saturated (68.75%), retry exact-action support only 56.25%, and completion clipping 3.91% | support gate failed; optimizer steps 0 |
| Policy target | Planner-owned routes, arguments, stopping, and post-probe continuation | Hard-coded RAG recovery labels cannot change deployed behavior | active |
| GRPO reward | Strict named-action reward with wrong-action cap plus argument, stopping, process, and format terms | Produces replicated growth on probe-to-migration while distinguishing Qwen from Rex | validated narrowly |
| Safety gate | No mean increase in wrong side-effecting actions | Seed44 added two wrong Flux routes, so positive primary metrics cannot promote the policy | failed, test sealed |
| Training replication | Three independent seeds; 3x deterministic inference per model | Primary strict-case mean delta `+0.3333`, but whole-policy gate failed | development only |
| Evaluation unit | `case_id` | Prevents two-step workflows from receiving double weight | active |
| Artifact storage | `/raid/zkq/artifacts/CAPA` | Keeps checkpoints, environments, caches, and traces outside Git | active |

## Evidence Boundaries

- Native ChatML alone produces clean stopping on the reused development split; SFT is not required to explain this behavior.
- The historical GRPOv4 route adapter failed the fixed-candidate runtime gate. The new strict curriculum supports a narrow runtime-owned effect, not whole-policy promotion.
- The runtime-probe three-seed mean improves overall action by `+0.1333` with entity-clustered 95% CI `[+0.0917,+0.1778]`, but mean wrong side-effecting actions increase by `+0.6667`.
- The runtime-probe test remains sealed under the preregistered stop rule. No threshold was relaxed and no extra seed was added after the guardrail failure.
- All models still fail the underspecified clarify category by routing to Adela; the next dataset must treat side-effect abstention as a primary task, not a secondary score term.
- `planner_grpo_compound245_eval_cases.jsonl` is a regression suite, not a held-out test set.
- PPO is gated until a clean test set and a supported SFT/DPO/GRPO comparison exist.

The Qwen3.5-era SFT/DPO/GRPO notes are archived at `experiments/archive/2026-07-12_legacy_tracking/JISHU.md`.
