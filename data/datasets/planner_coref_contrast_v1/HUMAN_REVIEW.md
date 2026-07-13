# Human Review: planner_coref_contrast_v1

## Review Order

1. Read `DATASET_CARD.md` and `manifest.json`.
2. Read the source guide at `data/datasets/planner_stateful_retrieval_v1/HUMAN_REVIEW.md`.
3. Review the 24 unique `coref_rewrite_then_rag#step1` source prompts.
4. Review the four 24-row contrast slices by expected action.
5. Review development predictions only after training; keep test predictions sealed.

## Acceptance Checks

- A coreference prompt contains a pronoun and a unique resolvable entity in prior history.
- The expected first action is `re_question` with `rewrite_reason=coref_resolve` and `finish_after_tool=false`.
- A RAG-miss first step has no miss observation yet, so its expected action remains `rag_answer`.
- A direct-RAG prompt names the entity explicitly and does not need rewriting.
- Memory and general prompts remain valid non-rewrite guardrails.
- Sampling replicas differ only by `sampling_replica` and `training_row_id`.

## Statistical Warning

Sampling replicas are optimization weights, not independent observations. Development confidence intervals continue to cluster by `entity_id` using the original unweighted evaluation split.
