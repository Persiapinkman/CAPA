# Human Review Guide

## Review Order

1. Read `DATASET_CARD.md` and `manifest.json` for scope and split integrity.
2. Review all 40 development cases in `planner_stateful_retrieval_v1_dev_cases.jsonl`.
3. Review training cases stratified by `category`, `template_id`, `fact_id`, and `entity_id`.
4. Spot-check derived ChatML rows only after the source labels are approved.
5. Do not inspect test predictions until the development checkpoint and training seeds are locked.

## Label Checklist

- `coref_rewrite_then_rag`: the first action must preserve the historical entity in `re_question`; the second action must retrieve the rewritten query.
- `rag_miss_rewrite_then_rag`: the sequence must be RAG with `finish_after_tool=false`, one rewrite, then final RAG with `finish_after_tool=true`.
- `memory_hit_end`: prior evidence must contain the requested fact and justify `end_reason=memory_hit`.
- `direct_rag_guardrail`: no pronoun or miss observation should require rewriting.
- `general_answer_guardrail`: the request must be answerable without private company facts.

## Rejection Conditions

- The expected action depends on information absent from the prompt.
- An entity, wording template, or exact query crosses split boundaries.
- A mocked observation contradicts the next expected action.
- The query admits another equally valid tool path under the current system prompt.
- The test split is used to select checkpoint, reward weights, temperature, or training duration.

Record human decisions outside the generated JSONL and rebuild a new dataset version after corrections; do not silently edit registered rows in place.
