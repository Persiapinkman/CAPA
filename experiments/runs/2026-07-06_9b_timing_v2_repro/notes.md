# Qwen3.5-9B Timing V2 Repro Eval

- Date: 2026-07-06
- Runner: `scripts/run_vllm_repro_eval_3x.sh`
- Model: `Qwen3.5-9B`
- API base: `http://10.111.32.253:8000/v1`
- Cases: `training/planner_dpo_train_seed_v1/eval/planner_routing_eval_90cases.json`
- Generation: `temperature=0`, `top_p=1`, `seed=42`, `do_sample=false`
- Timeout: 180 seconds per case

Aggregate result:

- Accuracy mean: 76/90 = 0.8445
- Accuracy stdev across 3 runs: 0.0193
- Mean case end-to-end elapsed: 8460.3903 ms
- Mean API call elapsed: 2512.3907 ms
- Mean slow cases per run: 3.3333
- Error types across all runs: `api_error=9`
- Retry count: 0

Accuracy varies because several slow API errors trigger fallback decisions and are
counted as failed routing cases.
