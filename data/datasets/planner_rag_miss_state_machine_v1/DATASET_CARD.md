# Dataset Card: planner_rag_miss_state_machine_v1

## Purpose

This fixed GRPO training view tests a three-state retrieval policy: retrieve once, rewrite only after an observed miss, then retrieve once more and stop. It follows the failed coreference-focused screen and targets the only strict action improvement observed in the mixed step-80 arm.

## Composition

- 240 weighted rows from 168 unique source steps and 24 training entities.
- 48 rows for each RAG-miss step: initial `rag_answer`, miss-conditioned `re_question`, and final `rag_answer`.
- 24 rows each for coreference rewrite, direct RAG, memory end, and general-answer contrasts.
- Development and sealed test evaluation remain the original unweighted, entity-disjoint `planner_stateful_retrieval_v1` splits.

The two replicas of each RAG-miss state are optimization weights, not independent observations.

## Concrete Trajectory

```text
User: 请查安全绳佩戴检测的模型版本；如果第一轮没有命中，做一次小步改写后再查，最多两轮。
Step 1 expected: rag_answer(finish_after_tool=false)
Observation: 首次检索未命中安全绳佩戴检测的模型版本，需要小步改写。
Step 2 expected: re_question(rewrite_reason="rag_miss", retrieval_round=2)
Observation: 改写完成，查询范围已收窄。
Step 3 expected: rag_answer(finish_after_tool=true)
```

The policy must condition on the observation: rewriting before the miss is premature, while repeating RAG immediately after the miss ignores the state transition.

## Evaluation

Promotion requires at least `+0.10` strict action-match improvement across all three RAG-miss steps and at least `+0.05` dense category improvement. Direct RAG, general answer, and memory end are guardrails; coreference is an anti-shortcut category.

## Human Review

Read `HUMAN_REVIEW.md`, then inspect independent source cases in `planner_stateful_retrieval_v1`. Test predictions remain sealed until the screen passes and confirmation seeds are trained.

## Limitations

- Observations are deterministic mocks.
- Weighting increases exposure but not entity diversity.
- A positive result supports this state machine, not end-to-end RAG quality.

Counts and hashes are in `manifest.json` and the generated `metadata.json`.
