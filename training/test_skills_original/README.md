# test-skills Original Data

This directory vendors the original case and trace data from the predecessor
repository:

`/media/nvme1n1p1/zengkeqin/projects/test-skills`

The copy is intended to make CAPA self-contained for later all-in-one planner
routing, DPO/SFT/GRPO data construction, and multi-step agent evaluation.

## Contents

- `dataset/`: original dataset CSV/JSON files, including planner routing and
  migration advisor evaluation data.
- `demo/eval/`: original evaluation case files, reports, and DPO extraction
  artifacts.
- `demo/llm_debug/`: original planner/tool request-response traces used as raw
  material for multi-step query and trajectory mining.
- `examples/images/`: image fixtures referenced by the original cases.
- `training/planner_dpo_train_seed_v1/`: original DPO seed data, reviews, and
  routing eval summaries.
- `skills/*/references/` plus skill README/SKILL metadata: tool reference data
  and examples needed to understand historical case/tool boundaries.
- `MANIFEST.txt`: sorted file list for this vendored snapshot.

## Exclusions

Environment/runtime and sensitive files were intentionally not copied:

- `.git/`
- `.venv/`
- `.codegraph/`
- `__pycache__/`
- `api_key.txt`

