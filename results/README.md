## Results Layout

`results/` stores reproducible evaluation outputs that are useful for review.

Tracked artifacts:

- `results/planner_routing_eval/`: single-step Planner routing eval outputs, including per-run JSON, 3x aggregate summaries, case audit CSVs, and failed-case CSVs.
- `results/compound_planner_eval/`: formal multi-step compound Planner routing eval outputs.

Every formal eval must write all of these review artifacts:

- `<REPORT_PREFIX>_aggregate.json`
- `<REPORT_PREFIX>_case_audit.csv`
- `<REPORT_PREFIX>_failed_cases.csv`

The audit CSV is the human review surface. The failed-cases CSV is required even
when it is empty, so regressions and residual bad cases are easy to inspect.

Ignored artifacts:

- Any other ad-hoc or temporary files under `results/` remain ignored by default unless explicitly unignored in `.gitignore`.

If a new results subdirectory should be preserved in git, add a focused allowlist entry instead of unignoring the whole `results/` tree.
