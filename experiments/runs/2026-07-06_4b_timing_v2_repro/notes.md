# Qwen3.5-4B Timing V2 Repro Eval

- Date: 2026-07-06
- Runner: `scripts/run_vllm_repro_eval_3x.sh`
- Model: `Qwen3.5-4B`
- API base: `http://10.111.32.253:8000/v1`
- Cases: `training/planner_dpo_train_seed_v1/eval/planner_routing_eval_90cases.json`
- Generation: `temperature=0`, `top_p=1`, `seed=42`, `do_sample=false`
- Timeout: 180 seconds per case

Aggregate result:

- Accuracy mean: 75/90 = 0.8333
- Accuracy stdev across 3 runs: 0.0
- Mean case end-to-end elapsed: 5458.97 ms
- Mean API call elapsed: 1463.7013 ms
- Mean slow cases per run: 2.0
- Error types across all runs: `api_error=6`
- Retry count: 0

Timing V2 excludes report writing from per-case timing and records API/tokens/retry/error
metadata per case.
