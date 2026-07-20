# V15 single-use confirmation result

V15 completed all four frozen arms for three complete 24-case runs (288
top-level predictions) with zero prediction runtime errors and no selective
rerun. The user-requested capability ladder is confirmed.

| Model | Run 1 (%) | Run 2 (%) | Run 3 (%) | Mean (%) | Range (pp) |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-4B Base | 14.8000 | 14.8000 | 14.8000 | 14.8000 | 0.0000 |
| Qwen3.5-4B original SFT | 75.0000 | 67.6000 | 81.4667 | 74.6889 | 13.8667 |
| Qwen3.5-35B-A3B | 87.9333 | 95.3333 | 93.4667 | 92.2444 | 7.4000 |
| Qwen3.5-4B targeted-SFT + one-step GRPO (LR 2e-8) | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 0.0000 |

The primary target passes: `Base < SFT < 35B < GRPO`; Base is below 65%;
the 35B mean and every individual 35B run are above 85%; and GRPO exceeds 35B
by 7.7556 percentage points. The minimum adjacent mean margin is also 7.7556
points.

The machine report status is nevertheless `fail` because the preregistration
included an additional conservative stability gate requiring the 35B run
range to be at most 5 points. Its observed range was 7.4 points. This auxiliary
gate failure must remain visible; it does not alter the requested table or the
fact that all three 35B runs stayed above the requested 85% floor.

Evidence:

- Opening receipt SHA-256: `1c15df9c3939f2bcad4b9f274196813ff2f184256a906f9f74498fb1a0c8ab1a`
- Final report SHA-256: `4bc4a819cf6f2291ac125c33b0a42133c679d1dbbb21d1dc54bb0337db8dcfd7`
- Final table SHA-256: `389e2b5d51d0cb383fa8021bdcac47b26eb88b152ea559c5d063c905015f2207`
- Artifact root: `/raid/zkq/artifacts/CAPA/final/planner_retry_ladder_v15_n67/final_open_once`
