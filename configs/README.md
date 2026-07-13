# Configuration Registry

- `models/`: immutable model identity and hardware-selection rationale.
- `train/`: normalized training configurations separate from launcher defaults.
- `eval/`: study arms and evaluation protocols.
- `environments/`: exact snapshots of environments used by recorded runs.

Run records must copy resolved values rather than only referring to mutable shell defaults.
