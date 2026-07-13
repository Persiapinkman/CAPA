# Versioned Datasets

Each dataset directory contains a human-readable `DATASET_CARD.md` and machine-readable `manifest.json` with source paths, split counts, category/action distributions, integrity findings, and SHA256 hashes.

The role field distinguishes training, development, regression, and sealed-test assets. A development or regression dataset must never be relabeled as a sealed test without creating a new dataset ID and untouched cases.

## Active Planner Datasets

- `planner_stateful_retrieval_v1`: independent train/dev/sealed-test cases for state-conditioned retrieval trajectories.
- `planner_coref_contrast_v1`: conditional weighted training view for strict coreference-action learning; evaluation remains on the unweighted stateful-retrieval splits.
- `planner_rag_miss_state_machine_v1`: exploratory weighted view for the full retrieve-rewrite-retrieve miss-recovery sequence.
- `planner_rag_miss_action_reward_v1`: the same state-machine view with an action-dominant, wrong-action-capped GRPO reward.
- `planner_focused_v3`: historical focused Planner dataset with a reused development split, not an untouched test set.
