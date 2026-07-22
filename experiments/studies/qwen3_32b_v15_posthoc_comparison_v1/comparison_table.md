# Qwen3-32B on V15 comparison

Status: **valid**. Qwen3-32B is a post-hoc read-only extension and does not alter the original V15 confirmation.

| Model | Run 1 (%) | Run 2 (%) | Run 3 (%) | Mean (%) | Range (pp) |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-4B Base | 14.8000 | 14.8000 | 14.8000 | 14.8000 | 0.0000 |
| Qwen3.5-4B original SFT | 75.0000 | 67.6000 | 81.4667 | 74.6889 | 13.8667 |
| Qwen3-32B local FP16 TP4 (post-hoc) | 74.0000 | 81.4000 | 88.8000 | 81.4000 | 14.8000 |
| Qwen3.5-35B-A3B | 87.9333 | 95.3333 | 93.4667 | 92.2444 | 7.4000 |
| Qwen3.5-4B targeted-SFT + one-step GRPO | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 0.0000 |

## Qwen3-32B scenario counts

| Run | Current-success passed | Metric-veto passed | Strict passed | Weighted (%) |
|---:|---:|---:|---:|---:|
| 1 | 0/12 | 10/12 | 10/24 | 74.0000 |
| 2 | 0/12 | 11/12 | 11/24 | 81.4000 |
| 3 | 0/12 | 12/12 | 12/24 | 88.8000 |

Runtime errors: 0; length finishes: 0; parse retries: 0.

Weighted formula: `(111 * metric_veto_pass_rate + 14 * current_success_pass_rate) / 125`.
