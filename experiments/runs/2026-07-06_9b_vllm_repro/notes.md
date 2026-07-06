# Qwen3.5-9B vLLM Repro Eval

- Date: 2026-07-06
- Runner: `scripts/run_vllm_repro_eval_3x.sh`
- Model: `Qwen3.5-9B`
- API base: `http://10.111.32.253:8000/v1`
- Cases: `training/planner_dpo_train_seed_v1/eval/planner_routing_eval_90cases.json`
- Generation: `temperature=0`, `top_p=1`, `seed=42`, `do_sample=false`
- Timeout: 180 seconds per case

Aggregate result:

- Accuracy: 76/90 = 0.8444
- Accuracy stdev across 3 runs: 0.0
- Mean run elapsed: 719195.8883 ms
- Mean case elapsed: 7988.4377 ms

The run is deterministic across all 3 repeats. Timing is dominated by repeated
180 second timeouts on several cases; the timeout cost is intentionally retained
in the efficiency metrics.
