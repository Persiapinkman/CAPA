# V12 selection-dev result

Status: **PROMOTE checkpoint-5 — sealed test authorized**

All four models completed the same 216 entity-isolated selection cases under deterministic inference. Coverage was 216/216 per model, JSON validity was 100%, completion clipping was 0%, and Planner runtime errors were 0. The larger 35B reference was not used for checkpoint selection.

| Model | Primary pass | Delta vs SFT | Entity-bootstrap 95% CI | Control pass | Control delta | Wrong-action occurrences | Promotion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| SFT | 19.44% | — | — | 43.75% | — | 32 | — |
| checkpoint-2 | 30.56% | +11.11pp | [+5.56, +16.67]pp | 68.06% | +24.31pp | 36 | Fail |
| **checkpoint-5** | **34.72%** | **+15.28pp** | **[+6.94, +23.61]pp** | **68.75%** | **+25.00pp** | **32** | **Pass / selected** |
| checkpoint-8 | 33.33% | +13.89pp | [+6.94, +20.83]pp | 67.36% | +23.61pp | 32 | Pass |

Checkpoint-5 and checkpoint-8 passed every frozen promotion gate. The preregistered tie-break selected checkpoint-5 because it had the higher primary entity-level pass rate. Checkpoint-2 failed only the wrong-side-effecting-action occurrence gate.

The frozen safety gate counts every forbidden action occurrence in a trajectory. Under that exact metric, checkpoint-5 and SFT are tied at 32, so the maximum added occurrence count is zero. A secondary, non-selection diagnostic deduplicated forbidden actions within each case and found 15 introduced cases and 12 removed cases: 32 unique case/action hits for checkpoint-5 versus 29 for SFT, net +3. This does not retroactively change the preregistered selection rule or selected checkpoint, but it is a limitation that must accompany the final claim.

Evidence:

- Selection decision: `selection_decision.json` (`e610b63a962d75f890c08fa234b1e341f4abea0fb23acaa13cba234623b4806e`)
- Secondary side-effect audit: `selection_side_effect_audit.json` (`7be5c90212c5a630dd7344fb1d9e5072cbd23cec5ceb7de59f97d062623f8236`)
- Selected predictions: `experiments/runs/20260717_qwen35_4b_v12_selection_dev/checkpoint-5/v12_checkpoint-5_run1_predictions.jsonl` (`21c1a28fb41e1baa12a44134d0b294f52b8cbd24eab502d1a66936ba898a4930`)
- Selected adapter: checkpoint-5 (`8d0112e0ce067ac479f7245c84be68b875e474dbb37b2df033f88d53e29c1ee3`)
- Sealed commitment before opening: `6e9413858d6e9cc09672d00d80dbda593e13e190457915498f5240c97b16c149`

The next allowed action is the single materialization of both committed sealed cohorts, followed by one common-protocol evaluation of SFT, checkpoint-5, and Qwen3.5-35B-A3B.
