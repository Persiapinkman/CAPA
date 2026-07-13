# Dataset Card: planner_stateful_retrieval_v1

State-conditioned Planner routing cases for retrieval, rewrite, retry, memory reuse, and guardrails.

## Research Question

Can GRPO improve multi-step state transitions when the policy has non-saturated sampled rewards?

## Composition

- Train: 120 cases, 192 step rows, and 24 entity groups.
- Dev: 40 cases, 64 step rows, and 8 entity groups.
- Test: 120 cases, 192 step rows, and 24 entity groups.

## Isolation

- Train/dev entity overlap: 0
- Train/test entity overlap: 0
- Dev/test entity overlap: 0
- Train/test template overlap: 0
- Test status: `sealed_until_hyperparameters_are_locked`

## Scenario Families

- `coref_rewrite_then_rag`: resolve an entity from history before retrieval.
- `rag_miss_rewrite_then_rag`: retrieve, rewrite after a miss, then retrieve once more.
- `memory_hit_end`: reuse sufficient prior evidence without another tool call.
- `direct_rag_guardrail`: retrieve directly when no rewrite is needed.
- `general_answer_guardrail`: answer general knowledge without private retrieval.

## Human Review

Review the source case JSONL before derived ChatML rows. Audit all development cases and stratify training samples by category, step index, entity, and template. Do not inspect model predictions on the test split until hyperparameters and training seeds are locked.
Use `HUMAN_REVIEW.md` for the label checklist and rejection conditions.

## Known Limitations

- Cases are curated templates rather than production traffic.
- Tool observations are deterministic mocks; this isolates Planner policy learning, not tool reliability.
- Derived prompts contain fixed opaque ledger identifiers. Training reads the registered step file directly so these nuisance tokens remain byte-identical across ranks and seeds.
- Entity grouping must be respected in confidence intervals.
- A positive result applies to this stateful retrieval task family until replicated on real traffic.

Hashes, category distributions, and similarity diagnostics are in `manifest.json`.
