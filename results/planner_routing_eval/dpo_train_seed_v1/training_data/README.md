# Planner DPO Training Data

This directory contains reviewed Planner DPO data for single-step routing policy
experiments. It is intended for a small planner model, such as Qwen3.5-4B, not
for tool-internal state machines or multi-step workflow RL.

## Scope

Approved preference types:

- `answerer > rag_answer`: generic CV concepts and methods should not overuse RAG.
- `qwen_detection > pipeline_eval`: single-image detection should not be upgraded to a full probe pipeline.
- `rag_answer > migration_advisor`: historical asset inventory should not be upgraded to migration reporting.

Excluded from this dataset:

- Adela model-id resolution and clarification state machine.
- Multi-step composite tasks such as feasibility plus executable probe.
- Company-policy-like questions where RAG may be appropriate.

## Files

- `planner_dpo_chat.jsonl`: all approved rows in chat-message format.
- `planner_dpo_chat_train.jsonl`: train split, chat-message format.
- `planner_dpo_chat_val.jsonl`: validation split, chat-message format.
- `planner_dpo_text.jsonl`: all approved rows in plain prompt format.
- `planner_dpo_text_train.jsonl`: train split, plain prompt format.
- `planner_dpo_text_val.jsonl`: validation split, plain prompt format.
- `planner_dpo_training_data_report.json`: split and distribution report.

Current split:

- Total: 113
- Train: 102
- Val: 11

## Suggested TRL Shape

For TRL-style DPO training, the text files already expose:

```json
{"prompt": "...", "chosen": "{...planner json...}", "rejected": "{...planner json...}", "meta": {...}}
```

If using a chat-template aware trainer, use the chat files:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}], "chosen": "{...}", "rejected": "{...}", "meta": {...}}
```

## Training Command

A project-local training environment can be prepared with `uv`:

```bash
uv venv --python /usr/bin/python3.10 .venv-train
uv pip install --python .venv-train/bin/python --link-mode=copy \
  "torch==2.8.0" \
  "transformers>=4.44,<5" \
  "datasets>=2.20,<4" \
  "accelerate>=0.33,<2" \
  "peft>=0.12,<1" \
  "trl>=0.12,<0.22" \
  sentencepiece protobuf scipy scikit-learn tensorboard
```

Default smoke/bring-up command, using a small public Qwen instruct model:

```bash
.venv-train/bin/python demo/eval/train_planner_dpo.py
```

For the intended planner base model, override `--model-name-or-path`:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-train/bin/python demo/eval/train_planner_dpo.py \
  --model-name-or-path /path/to/Qwen3.5-4B \
  --train-file results/planner_routing_eval/dpo_train_seed_v1/training_data/planner_dpo_text_train.jsonl \
  --validation-file results/planner_routing_eval/dpo_train_seed_v1/training_data/planner_dpo_text_val.jsonl \
  --output-dir outputs/planner-qwen35-4b-dpo \
  --learning-rate 5e-6 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --beta 0.1 \
  --use-lora true \
  --fp16 true
```

## Evaluation After Training

Use the frozen 90-case routing eval as dev/eval:

```bash
python3 demo/eval/run_planner_routing_eval.py \
  --model <DPO_MODEL_ENDPOINT_OR_NAME> \
  --timeout-seconds 90 \
  --out results/planner_routing_eval/planner_routing_report_<MODEL>_90cases_after_dpo.json
```

Compare against:

- original `Qwen3.5-4B`
- original `Qwen3.5-9B`
- default / 35B prompt-only

Primary metrics:

- overall routing accuracy
- `answerer` vs `rag_answer` error rate
- single-image detection vs `pipeline_eval` error rate
- historical asset QA vs `migration_advisor` error rate
- planner latency and cost
