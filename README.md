# CAPA

CAPA is a Planner-centric agent and evaluation workspace for visual AI capability routing. Given a user query, optional image, conversation trajectory, and tool state, the Planner emits one structured JSON decision for the next action.

## Current Research Question

The demo defaults to Qwen3.5-35B-A3B, while the confirmed research recipe uses a Qwen2.5-7B SFTv3 initializer. The current study establishes that action-dominant GRPO produces reproducible strict-action growth on an entity-isolated five-step retrieval-recovery state machine; its sealed-test three-seed mean action gain is `+0.1292` with entity-clustered 95% CI `[+0.0750, +0.1896]`.

Current status is generated at:

```text
reports/CURRENT.md
```

The authoritative machine-readable run history is:

```text
experiments/registry.jsonl
```

## Repository Layout

| Path | Responsibility |
|---|---|
| `src/capa/` | Planner, prompts, clarification, memory, tool contracts, registry, and evaluation helpers |
| `demo/` | HTTP/UI application and compatibility imports for legacy entrypoints |
| `pipelines/` | Reproducible dataset registration, evaluation, comparison, and registry commands |
| `configs/` | Versioned model, training, evaluation, and environment configurations |
| `data/datasets/` | Dataset cards, split statistics, integrity audits, and SHA256 manifests |
| `training/` | Historical/package-specific data builders and training implementations |
| `experiments/studies/` | Research questions, hypotheses, arms, metrics, and decision rules |
| `experiments/runs/` | Immutable per-run config, metrics, notes, and provenance |
| `reports/` | Generated current status, leaderboard, and compact study/error reports |
| `artifacts/` | Metadata for the external artifact store |

Large checkpoints, local environments, caches, logs, and traces live under `/raid/zkq/artifacts/CAPA`. Legacy paths such as `outputs/` remain compatibility symlinks.

## Runtime Architecture

The request path is:

```text
demo/demo_server.py
  -> src/capa/agent.py
  -> src/capa/memory.py
  -> src/capa/tools/executor.py
  -> skills/*
```

The Planner currently routes among RAG, query rewriting, direct answering, image generation, Qwen/Rex-Omni detection, full pipeline evaluation, migration advice, and Adela evaluation.

## Reproducible Commands

Validate data and experiment records:

```bash
python pipelines/data/register_planner_dataset.py
python pipelines/experiments/registry_cli.py validate
python pipelines/experiments/registry_cli.py render
```

Run unit tests:

```bash
PYTHONPATH=src:. .venv-trl-grpo-cu124/bin/python -m unittest discover -s tests -v
```

Start the demo:

```bash
python demo/demo_server.py --port 18080
```

## Evidence Rules

- `planner_focused_v3` is a reused development split, not a sealed test set.
- The compound245 set is a regression suite with historical train overlap.
- `planner_stateful_retrieval_v2` test was opened once after its preregistered development replication gate passed and is now frozen against further model selection.
- Offline multi-step evaluation uses mock observations and does not establish end-to-end tool success.
- A training-method promotion requires case-level confidence intervals and independent training seeds.

See `experiments/EVALUATION_POLICY.md` and `experiments/TECHNICAL_DECISIONS.md` for the full protocol and current rationale.
