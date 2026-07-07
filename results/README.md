## Results Layout

`results/` stores reproducible evaluation outputs that are useful for review.

Tracked artifacts:

- `results/planner_routing_eval/`: single-step Planner routing eval outputs, including per-run JSON, 3x aggregate summaries, case audit CSVs, and failed-case CSVs.

Ignored artifacts:

- Any other ad-hoc or temporary files under `results/` remain ignored by default unless explicitly unignored in `.gitignore`.

If a new results subdirectory should be preserved in git, add a focused allowlist entry instead of unignoring the whole `results/` tree.
