# Qwen3.5 Planner GRPO Stateprompt Eval Compare

## Setup

- Cases: `training/planner_grpo_seed_v1/cases/planner_grpo_train_cases.jsonl`
- API base: `http://10.111.32.253:8000/v1`
- Decoding: `temperature=0`, `top_p=1`, `seed=42`, `do_sample=false`
- Timeout: `60s`
- Reward: detection tool equivalence enabled (`qwen_detection` ~= `rexomni_detection`)
- Agent: stateprompt/tool-description cleanup enabled

## Overall

| Model | Runs | Mean Score | Pass Rate Mean | Pass All Runs | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen3.5-4B | 3 | 0.978686 | 0.851701 | 0.804082 | Full 3x completed |
| Qwen3.5-9B | 1 | 0.966591 | 0.897959 | N/A | Single completed run; run2 interrupted because server throughput was very slow |
| Qwen3.5-35B-A3B | 3 | 0.991211 | 0.964626 | 0.959184 | Full 3x completed |

## Key Categories

| Model | Single Image Probe | Probe -> Migration | Migration Text | Migration + Image | Full Eval | General Answer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B | 0.995495 | 0.868852 | 0.491228 | 0.869281 | 0.600000 | 1.000000 |
| Qwen3.5-9B | 1.000000 | 0.672131 | 0.947368 | 0.960784 | 0.800000 | 1.000000 |
| Qwen3.5-35B-A3B | 1.000000 | 0.874317 | 1.000000 | 1.000000 | 1.000000 | 0.833333 |

For 4B and 35B, category numbers are 3-run pass-rate means. For 9B, category numbers are from the single completed run.

## Interpretation

- The cleaned agent/prompt setup is enough for 35B to exceed the reproducibility target: pass-all-runs is `95.9%`.
- 4B still misses the strict pass target. Its mean score is high, but strict full-pass is pulled down by migration advisor argument exactness and some full-eval boundary cases.
- 9B single-run pass rate is close to `90%`, but `probe_then_migration` remains weak (`67.2%`). This is the most relevant remaining GRPO target.
- The main GRPO direction should remain compound state transition, especially detection probe -> migration advisor. Single-step detection routing is now largely solved by prompt/schema cleanup.
