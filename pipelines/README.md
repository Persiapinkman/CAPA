# Reproducible Pipelines

- `data/register_planner_dataset.py`: rebuild dataset statistics, leakage checks, duplicate rates, similarity diagnostics, and hashes.
- `eval/run_generation_eval.py`: run deterministic repeated generation while recording full provenance and external predictions.
- `eval/compare_generation_runs.py`: compute step, case-macro, category-macro, and paired case-clustered intervals.
- `experiments/registry_cli.py`: migrate, validate, append, and render the authoritative experiment registry.

Package-specific historical trainers remain under `training/` during the compatibility migration. New study orchestration should call these normalized pipeline surfaces rather than writing ad hoc report files.
