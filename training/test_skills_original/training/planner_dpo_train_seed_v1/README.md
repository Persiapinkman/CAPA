# Planner DPO Train Seed v1

This directory packages the reviewed DPO data and validation assets needed to
train and evaluate a small Planner model.

## Repository

- Repo: `git@gitlab.sz.sensetime.com:xiaokun1/test-skills.git`
- Branch used when packaging: `eval/planner-routing-dpo`

## What This Trains

This dataset trains single-step Planner routing boundaries only:

- `answerer > rag_answer`: generic CV concepts and methods should not overuse RAG.
- `qwen_detection > pipeline_eval`: single-image detection should not be upgraded to a full probe pipeline.
- `rag_answer > migration_advisor`: historical asset inventory should not be upgraded to migration reporting.

Excluded from DPO:

- Adela model-name/rawmodel_id resolution and clarification state machine.
- Multi-step or composite workflows.
- Questions likely governed by company policy, online acceptance, monitoring, or annotation QA standards.

## Files

Training data:

- `training_data/planner_dpo_text_train.jsonl`
- `training_data/planner_dpo_text_val.jsonl`
- `training_data/planner_dpo_chat_train.jsonl`
- `training_data/planner_dpo_chat_val.jsonl`
- `training_data/planner_dpo_training_data_report.json`

Review/audit:

- `review/planner_dpo_train_seed_pairs.approved.jsonl`
- `review/planner_dpo_train_seed_review.csv`
- `review/planner_dpo_train_seed_review_report.json`

Validation/eval:

- `eval/planner_routing_eval_90cases.json`
- `eval/planner_routing_report_Qwen3.5-4B_90cases_baseline_summary.json`
- `eval/planner_routing_report_Qwen3.5-9B_90cases_arch_rescored.json`
- `eval/planner_routing_report_Qwen3.5-35B-A3B_90cases_baseline_summary.json`

Image fixtures are tracked in the repo and reused by the eval set:

- `examples/images/banner.jpg`
- `examples/images/fisherman.jpg`
- `examples/images/person_with_bag.png`
- `examples/images/smoke.jpg`
- `examples/images/trash_truck.jpg`

## Data Counts

Approved DPO rows: 113

Train/val split:

- Train: 102
- Val: 11

Approved distribution:

- `answerer > rag_answer`: 33
- `qwen_detection > pipeline_eval`: 40
- `rag_answer > migration_advisor`: 40

## Baselines On 90-Case Eval

| Model | Accuracy |
|---|---:|
| Qwen3.5-4B | 68/90 = 75.56% |
| Qwen3.5-9B | 76/90 = 84.44% |
| Qwen3.5-35B-A3B | 82/90 = 91.11% |

## Suggested Training Input

Use text format first unless your trainer has native chat-template support:

```text
training_data/planner_dpo_text_train.jsonl
training_data/planner_dpo_text_val.jsonl
```

Each row has:

```json
{
  "prompt": "...",
  "chosen": "{...planner JSON...}",
  "rejected": "{...planner JSON...}",
  "meta": {}
}
```

## Evaluation Command

From repo root:

```bash
python3 demo/eval/run_planner_routing_eval.py \
  --cases training/planner_dpo_train_seed_v1/eval/planner_routing_eval_90cases.json \
  --model <DPO_MODEL_ENDPOINT_OR_NAME> \
  --timeout-seconds 90 \
  --resume \
  --out results/planner_routing_eval/planner_routing_report_<MODEL>_90cases_after_dpo.json
```

Compare against the packaged baseline summaries under `eval/`.
