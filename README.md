# CAPA

CAPA is a Planner-centric agent demo and evaluation workspace for visual AI capability routing.

The current focus is the Planner: given a user query, optional image, and conversation/tool state, it chooses the next high-level action as JSON. The repo contains the demo agent, tool registry, routing evaluation scripts, and seed datasets for SFT/DPO/GRPO-style Planner work.

## Repository Layout

- `demo/`: agent runtime, prompts, memory system, tool schemas/registry/executors, web demo, and routing eval code.
- `skills/`: executable skills/tools used by the demo, including open-set detection, Flux image generation, RAG answer, target detection evaluation, and report generation.
- `training/planner_dpo_train_seed_v1/`: single-step Planner routing data and historical SFT/DPO seed assets.
- `training/planner_grpo_seed_v1/`: compound/multi-step Planner cases, reward verifier, offline rollout scripts, repeated eval scripts, and reports.
- `scripts/`: reproducible serving/eval/train entrypoints.
- `experiments/`: active experiment ledger plus archived historical results.
- `examples/images/`: local image fixtures used by eval cases.

## Current Evaluation Tracking

Human-readable active results:

```text
experiments/EXPERIMENT_LOG.md
```

Machine-readable active manifest:

```text
experiments/manifest.jsonl
```

Older or superseded results are archived under:

```text
experiments/archive/
```

The current active tracking has two lanes:

- single-step routing: `training/planner_dpo_train_seed_v1/eval/planner_routing_eval_90cases.json`
- multi-step routing: `training/planner_grpo_seed_v1/cases/planner_grpo_train_cases.jsonl`

## Formal Eval Protocol

Formal evals use vLLM/OpenAI-compatible serving with deterministic generation:

- `temperature=0`
- `top_p=1`
- `do_sample=false`
- `seed=42`
- 3 repeats, then aggregate
- per-case CSV audit artifacts for review

See:

```text
experiments/EVALUATION_POLICY.md
```

## Single-Step Routing Eval

Run the 90-case single-step Planner routing eval:

```bash
MODEL='Qwen3.5-35B-A3B' \
API_BASE='http://10.111.32.253:8000/v1' \
REPORT_PREFIX='qwen35_35b_a3b_timing_v2_zip90' \
bash scripts/run_vllm_repro_eval_3x.sh
```

Default cases:

```text
training/planner_dpo_train_seed_v1/eval/planner_routing_eval_90cases.json
```

Default output:

```text
results/planner_routing_eval/
```

Each repeated single-step eval writes:

```text
results/planner_routing_eval/<REPORT_PREFIX>_aggregate.json
results/planner_routing_eval/<REPORT_PREFIX>_case_audit.csv
results/planner_routing_eval/<REPORT_PREFIX>_failed_cases.csv
```

The audit CSV is the required review surface: one row per case with query, expected action/slots, each repeat's actual action/input, failure reason, timing, and token usage. The failed-cases CSV is the same schema filtered to rows where any repeat failed.

## Multi-Step Compound Routing Eval

Run the compound Planner eval used for GRPO diagnostics:

```bash
MODEL='Qwen3.5-35B-A3B' \
API_BASE='http://10.111.32.253:8000/v1' \
REPORT_PREFIX='qwen35_35b_a3b_grpo_compound245_stateprompt_t60_3x' \
RUNS=3 \
TIMEOUT_SECONDS=60 \
OPENAI_TIMEOUT_SECONDS=60 \
bash scripts/run_grpo_repro_eval_3x.sh
```

Default cases:

```text
training/planner_grpo_seed_v1/cases/planner_grpo_train_cases.jsonl
```

Default output:

```text
training/planner_grpo_seed_v1/reports/repro_eval/
```

The multi-step eval is offline: it rolls out Planner decisions and injects mock tool observations instead of executing real tools.

## Current GRPO Direction

The current GRPO work is treated as a diagnostic/regression suite first, training set second.

After cleaning reward noise and prompt/tool-description ambiguity, 35B passes the compound eval overall. Residual failures concentrate in compound state transitions, especially:

```text
single-image detection probe -> migration_advisor
```

This is the main candidate direction for GRPO data construction. Generic single-step routing is not the priority unless a new eval shows fresh hard cases.

## Demo

Start the local demo server:

```bash
python3 demo/demo_server.py --port 18080
```

Then open:

```text
http://127.0.0.1:18080
```

More demo details are in:

```text
demo/README.md
```

## Notes

- The Planner output is a structured JSON decision, not the final user-facing answer.
- Some transitions are handled by engineering in `demo/agent.py`; do not train the model to duplicate those if the runtime already owns them.
- Generated eval reports and rollout predictions are intentionally kept under `results/` or `training/*/reports/` so they can be audited and re-scored.
