# Qwen3.5-35B-A3B vLLM Repro Eval

- Date: 2026-07-06
- Runner: `scripts/run_vllm_repro_eval_3x.sh`
- Model: `Qwen3.5-35B-A3B`
- API base: `http://10.111.32.253:8000/v1`
- Cases: `training/planner_dpo_train_seed_v1/eval/planner_routing_eval_90cases.json`
- Generation: `temperature=0`, `top_p=1`, `seed=42`, `do_sample=false`
- Timeout: 180 seconds per case

Aggregate result:

- Accuracy: 80/90 = 0.8889
- Accuracy stdev across 3 runs: 0.0
- Mean run elapsed: 99011.615 ms
- Mean case elapsed: 1097.5127 ms

This deterministic re-eval is lower than the old packaged summary result
(82/90), but unlike the packaged summary it includes measured per-case timing.
