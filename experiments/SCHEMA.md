# Experiment Record Schema

`experiments/registry.jsonl` is append-only and contains one JSON object per immutable run. Schema version `2.0` requires the following blocks.

| Field | Meaning |
|---|---|
| `run_id` | Globally unique stable identifier |
| `study_id` | Research question and experiment matrix owning the run |
| `purpose` | Why the run exists |
| `hypothesis` | Falsifiable expectation before observing results |
| `parent_run_id` | Training run, initializer, or direct predecessor |
| `provenance` | Git commit/dirty state, exact command, seed, timestamps, and environment |
| `data` | Dataset ID, split, paths, counts, and SHA256 |
| `method` | Model, adapter, prompt, generation, and training configuration |
| `metrics` | Primary metric plus complete aggregate and comparison statistics |
| `artifacts` | Small run files and external raw artifact paths |
| `decision` | `baseline`, `promote_development`, `inconclusive`, `reject`, or historical outcome with rationale |

## Lifecycle

1. Define the research question and arms under `experiments/studies/<study_id>/`.
2. Register/version the dataset before running a model.
3. Write each run to `experiments/runs/<run_id>/` and raw outputs to the external artifact root.
4. Complete paired or multi-seed comparison before assigning a promotion decision.
5. Append the finalized `run_record.json` to the registry.
6. Validate and regenerate `reports/CURRENT.md` and `reports/leaderboard.csv`.

Registry rows are never edited to reuse a `run_id`. Corrections are new runs linked through `parent_run_id` or an explicit supersession field.
