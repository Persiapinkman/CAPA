# V14 confirmation result and post-open audit remediation

V14 was opened once at `2026-07-20T06:34:45.946695+00:00`. All four frozen
arms completed three 24-case runs, for 288 independent top-level prediction
rows. No prediction file was resumed, selectively rerun, or modified.

The original aggregate command then failed before producing a result because
the frozen auditor compared four means against a three-element shifted list
with `zip(..., strict=True)`. Before remediation, all 12 prediction files were
hashed and confirmed to contain exactly 24 rows. The remediation only replaced
that invalid adjacent-list construction with the equivalent equal-length
comparison `zip(values[:-1], values[1:], strict=True)` and added a regression
test. It did not alter cases, predictions, scoring, weights, thresholds, or
model artifacts.

The opening receipt records the original auditor SHA-256
`0a9394ce66bf46c1ba33853a11e8411b430971c65cf75a87752c577ffcfd1dd4`.
The remediated auditor SHA-256 is
`80f0842d9b6579ccce95c3173b0968a13299dcf289c96736cefb8930c5959de5`.
The targeted regression suite passed (`4 passed`).

## Frozen result

| Model | Run 1 (%) | Run 2 (%) | Run 3 (%) | Mean (%) | Range (pp) |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-4B Base | 29.6000 | 29.6000 | 29.6000 | 29.6000 | 0.0000 |
| Qwen3.5-4B original SFT | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 0.0000 |
| Qwen3.5-35B-A3B | 97.2000 | 87.9333 | 97.2000 | 94.1111 | 9.2667 |
| Qwen3.5-4B targeted-SFT + one-step GRPO (n60) | 77.8000 | 92.6000 | 85.2000 | 85.2000 | 14.8000 |

Status: **failed confirmation**. Base was below 65%, and every 35B run was
above 85%, but the required strict order failed, GRPO trailed 35B by 8.9111
percentage points, and the 35B range exceeded the preregistered 5-point bound.
V14 may be used only as development evidence for subsequent work; any new
confirmation must use a newly sealed, disjoint version.

## Evidence

- Opening receipt SHA-256: `f1dd199d89f6e6e871a4042a463204de07a57b0b6a3e250cc91117e3074dad1f`
- Final report SHA-256: `d7094090055f7729bf20030f59d695e3af05cbc8aa9b703a21fd0bcfd9943600`
- Final table SHA-256: `30042005a10a05a3a72cf4ff287a87356c8c4a48b42224ad91ae1edef7daee44`
- Final report: `/raid/zkq/artifacts/CAPA/final/planner_retry_ladder_v14_n62/final_open_once/final_report.json`
- Prediction hashes are embedded under `artifacts` in the final report.
