# V10 support-block execution correction

Status: correction frozen before inspecting any per-block support metrics and
before running any optimizer step.

The committed V10 preregistration already contained:

- separate `support_dev_a` and `support_dev_b` splits;
- the rule `all_scenarios_and_both_blocks_must_pass`;
- fixed per-block gold-support and reward-variance thresholds.

The executable binding list `support_blocks: [A, B]` was accidentally omitted.
Consequently the first gate payload did not calculate `by_support_block` and is
retained as `support_decision.invalid_missing_block_enforcement.json` rather
than accepted as optimizer authorization.

This correction adds only the missing block names. It does not change any
sample, seed, scenario, model, threshold, or pass/fail rule. The already frozen
1728-sample file is reused without resampling; its SHA-256 is
`3a0198e6c9f7d059651dcb267a8d37155345fe67b9e7c3810d95861d983912e7`.
The original invalid gate payload SHA-256 is
`0ce8828f8f9ce7e8e0e06110d78904a095811c90a7e99158b82d9033d22dce2e`.

Before applying the correction, only combined and scenario/detector results
had been inspected; no A/B metrics had been produced. If either corrected
block check fails, V10 runs zero optimizer steps.
