# Dataset Card: planner_coref_contrast_v1

## Purpose

This is a fixed GRPO training view for testing whether the Planner can learn a real state-conditioned coreference action. It is activated only if the mixed `planner_stateful_retrieval_v1` step-80 development gate fails.

## Composition

- 192 weighted training rows from 120 unique source step rows and 24 training entities.
- 96 rows: `coref_rewrite_then_rag#step1` (`re_question` target).
- 24 rows: `rag_miss_rewrite_then_rag#step1` (`rag_answer` target).
- 24 rows each: direct RAG, memory end, and general-answer guardrails.
- Development and sealed test evaluation reuse the entity- and template-disjoint splits of `planner_stateful_retrieval_v1`.

The 96 coreference rows are four deterministic sampling replicas of 24 independent source prompts. They increase optimization exposure; they do not increase the number of independent entities or examples for statistical inference.

## Why These Contrasts

The target rows require `re_question` when a pronoun can be resolved from history. The RAG-miss and direct-RAG rows require `rag_answer` on the first step, while memory and general rows require non-retrieval decisions. This prevents a trivial policy that emits `re_question` for every retrieval-related prompt or whenever history is present.

## Concrete Training Pair

Target example (`SRV1-TRAIN-COREF-001#step1`):

```text
History: 安全绳佩戴检测是否有历史能力？
Current: 这个项目的模型版本再从内部资料查一下。
Expected: re_question(rewrite_reason="coref_resolve", context_hint="安全绳佩戴检测")
```

Contrast example (`SRV1-TRAIN-DIRECT-001#step1`):

```text
Current: 请从公司内部资料查询安全绳佩戴检测的模型版本。
Expected: rag_answer(finish_after_tool=true)
```

The first query needs its entity copied from history before retrieval. The second already names the entity, so rewriting it would be an unnecessary state-transition error.

## Evaluation

The primary development endpoint is strict action match on step 1 of `coref_rewrite_then_rag`, not only the dense verifier score. Promotion requires both an action-match increase and a verifier increase, with guardrail and anti-shortcut limits defined in `conditional_coref_contrast_arm.json`.

## Human Review

Read `HUMAN_REVIEW.md`, then review the independent source rows in `planner_stateful_retrieval_v1`. Do not review sampling replicas as if they were independent examples. Test predictions remain sealed until the arm and training seeds are locked.

## Limitations

- This view changes sampling weights but does not add independent entities.
- Prompts are synthetic Planner states with mocked observations.
- A positive result supports only resolvable coreference routing until replicated on production-like traffic.

Machine-readable counts, paths, and hashes are in `manifest.json` and the generated training `metadata.json`.
