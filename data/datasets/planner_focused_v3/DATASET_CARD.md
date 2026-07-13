# Dataset Card: planner_focused_v3

Focused CAPA Planner routing cases and derived ChatML step data.

## Intended Use

Training and development diagnostics for CAPA Planner routing. This dataset must not support final generalization claims.

## Composition

- Source: 154 cases and 245 expected decisions.
- Train: 123 cases and 196 step rows.
- Dev: 31 cases and 49 step rows.
- Regression: 245 cases.

## Integrity

- Status: `development_only_not_sealed`
- Train/dev case ID overlap: 0
- Train/regression case ID overlap: 69
- Dev nearest-template median similarity: 0.7941

## Known Limitations

- The dev split has been reused for model selection and is not a sealed test set.
- Case-level grouping prevents exact ID leakage, but template families span train and dev.
- The regression set overlaps train by 69 case IDs.

Hashes and complete distributions are in `manifest.json`.
