# Human Review: planner_rag_miss_state_machine_v1

## Review Order

1. Read `DATASET_CARD.md` and `manifest.json`.
2. Review the 24 independent RAG-miss source cases and all three expected steps.
3. Verify the four 24-row contrast slices.
4. Treat `sampling_replica` as a weight, not a new example.
5. Keep test predictions sealed until the confirmation rule allows them.

## Acceptance Checks

- Step 1 retrieves the explicit entity and sets `finish_after_tool=false`.
- The miss observation is present before step 2 requests a rewrite.
- Step 2 uses `rewrite_reason=rag_miss`, `retrieval_round=2`, and does not finish.
- Step 3 retrieves the rewritten entity/fact and sets `finish_after_tool=true`.
- No fourth retrieval or repeated rewrite is labeled valid.
- Coreference contrast still requires `re_question` before any RAG call.

## Statistical Warning

Development and test inference uses unweighted rows and clusters by `entity_id`; weighted training replicas must never be counted as independent units.
