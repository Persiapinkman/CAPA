# Planner retry/migrate V6 human review protocol

_Independent semantic-review worksheet for the frozen 45-case stratified sample_

---

## 📋 Status

Machine validation is complete, but **independent human review is still pending**. The review sample in [human_review_sample.jsonl](human_review_sample.jsonl) contains one case from every scenario in every split: 5 splits × 9 scenarios = 45 cases.

Do not change gold labels in place. Record reviewer decisions separately, adjudicate disagreements, then rebuild a new dataset version if any label or wording must change. This preserves the current manifest hashes.

## 🔍 Review questions

For each sampled case, answer all four fields already present under `review_checks`:

1. `policy_label_correct`: Does the expected action follow only from the latest structured state and the stated policy?
2. `observation_has_no_action_hint`: Does the observation avoid telling the model to retry, migrate, or end?
3. `counterfactual_isolation_valid`: For core cases, are query, alias, badge, detector, entity, and fixture held fixed across the three states?
4. `language_natural_enough`: Is the query understandable and plausible enough for training or evaluation?

Use `true`, `false`, or `needs_adjudication`; add a concise note for every value other than `true`.

## 🎯 Scenario checklist

| Scenario | Expected target transition | Critical check |
| --- | --- | --- |
| `core_retryable_fresh` | Same detector | `retryable=true`, `retry_count=0`; second observation exists |
| `core_nonretryable` | `migration_advisor` | `retryable=false`, `retry_count=0` |
| `core_budget_exhausted` | `migration_advisor` | `retryable=true`, `retry_count>=1` |
| `guard_initial_success_end` | `end` | All four metric gates pass |
| `guard_initial_metric_veto_migrate` | `migration_advisor` | At least one metric gate fails |
| `guard_missing_required_state_migrate` | `migration_advisor` | A required routing field is absent |
| `guard_conflicting_state_migrate` | `migration_advisor` | Fields are internally inconsistent |
| `guard_stale_history_current_success_end` | `end` | Current success overrides archived error |
| `guard_stale_history_current_error_migrate` | `migration_advisor` | Current error overrides archived success |

## ⚙️ Acceptance rule

Accept V6 for training only when:

- Two independent reviewers complete all 45 rows
- No `policy_label_correct=false` remains
- No action hint remains in an observation
- Every disagreement is adjudicated and documented
- Any text correction produces a new manifest and reruns all machine audits

The review is a release gate for data use; it is not permission to start SFT or GRPO.
