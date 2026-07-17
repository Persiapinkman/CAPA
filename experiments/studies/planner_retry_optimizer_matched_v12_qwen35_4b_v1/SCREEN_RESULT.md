# V12 safety-weighted GRPO screen result

Status: **PASS — selection-dev evaluation authorized**

The preregistered eight-step screen restarted from the unchanged V10 checkpoint-10 continuation initializer. It did not continue from the canary. The first two steps and checkpoint-2 adapter reproduced the canary byte-for-byte.

| Runtime gate | Observed | Result |
| --- | ---: | :---: |
| Optimizer / logged steps | 8 / 8 | Pass |
| Generation events | 256 | Pass |
| Gradient / optimizer events | 32 / 32 | Pass |
| Missing / nonfinite gradient tensors | 0 / 0 | Pass |
| Missing / nonfinite required metrics | 0 / 0 | Pass |
| Completion clipping | 0% | Pass |
| Peak allocated / minimum free memory | 14.01 / 15.55 GiB | Pass |
| Mean safety reward | 0.9102 | Pass |
| Core W&B and safety metrics | complete and finite | Pass |

Seven of eight logged steps had nonzero gradient norm. Step 5 was a transparently retained saturated batch: all rewards were 1, group-normalized advantages were 0, and gradient norm was 0. It was not resampled and the screen was not extended. The next batch recovered a gradient norm of 0.3489, so this was a batch-level lack of contrast rather than a runtime failure.

Frozen selection candidates:

- checkpoint-2: `040fc6f69b967b29d9cd3a3c94988fb10e699b499f21b87a9a100759efafae38`
- checkpoint-5: `8d0112e0ce067ac479f7245c84be68b875e474dbb37b2df033f88d53e29c1ee3`
- checkpoint-8: `0bd419562f17bd93abfb25c8d99ac3150611df2ce893ed3accb3f1d2e16ee9b7`

W&B run: `uwxyw57s` in project `capa-planner-post-training`.

The screen proves that the run and telemetry are healthy; it does not prove model improvement. Only the preregistered 216-case selection-dev comparison may decide promotion. The 35B reference and both sealed cohorts remain unopened.
