# Qwen3.5-4B State-Prompt Zip90 Repro Eval

- Date: 2026-07-07
- Runner: `scripts/run_vllm_repro_eval_3x.sh`
- Model: `Qwen3.5-4B`
- API base: `http://10.111.32.253:8000/v1`
- Cases: `training/planner_dpo_train_seed_v1/eval/planner_routing_eval_90cases.json`
- Prompt/tool descriptions: current state-transition cleanup version
- Generation: `temperature=0`, `top_p=1`, `seed=42`, `do_sample=false`
- Timeout: 180 seconds per case

Aggregate result:

- Accuracy mean: 0.9074
- Accuracy stdev: 0.0064
- Passed mean: 81.6667/90
- Mean case elapsed: 4397.8243 ms
- Mean API elapsed: 2395.8863 ms
- Mean slow cases per run: 1.3333
- Timeout count mean: 0.0
- Error types across all runs: {"api_error": 3}

Artifacts:

- Aggregate: `results/planner_routing_eval/qwen35_4b_stateprompt_zip90_3x_aggregate.json`
- Run reports: `results/planner_routing_eval/qwen35_4b_stateprompt_zip90_3x_run1.json`, `run2.json`, `run3.json`
