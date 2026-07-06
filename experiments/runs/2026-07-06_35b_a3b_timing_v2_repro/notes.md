# Qwen3.5-35B-A3B Timing V2 Repro Eval

- Date: 2026-07-06
- Runner: `scripts/run_vllm_repro_eval_3x.sh`
- Model: `Qwen3.5-35B-A3B`
- API base: `http://10.111.32.253:8000/v1`
- Cases: `training/planner_dpo_train_seed_v1/eval/planner_routing_eval_90cases.json`
- Generation: `temperature=0`, `top_p=1`, `seed=42`, `do_sample=false`
- Timeout: 180 seconds per case

Aggregate result:

- Accuracy mean: 80/90 = 0.8889
- Accuracy stdev across 3 runs: 0.0
- Mean case end-to-end elapsed: 1096.9063 ms
- Mean API call elapsed: 1068.4413 ms
- Mean slow cases per run: 0.0
- Error types across all runs: none
- Retry count: 0

This is the cleanest timing run among the three models: no slow cases, no retry,
and no API errors.
