# V12 sealed result

Status: **objective_not_met**, with statistically strong primary-task superiority.

The sealed set was materialized once from commitment `6e9413858d6e9cc09672d00d80dbda593e13e190457915498f5240c97b16c149`. Checkpoint 5 had been selected without the larger reference. SFT, GRPO, and `Qwen3.5-35B-A3B` each covered all 432 cases and 24 independent entities with zero final Planner runtime errors.

| Comparison | Primary pass rate | Delta | Entity-paired bootstrap 95% CI |
| --- | ---: | ---: | ---: |
| 4B-GRPO vs 4B-SFT | 30.56% vs 26.39% | +4.17pp | [+1.39pp, +7.64pp] |
| 4B-GRPO vs 35B-A3B | 30.56% vs 4.17% | +26.39pp | [+20.83pp, +31.25pp] |

The target therefore achieved the research core: GRPO improved the 4B initializer and made 4B strongly exceed the fixed larger model on the preregistered entity-isolated primary residual metric. This claim is limited to that metric and benchmark. Overall pass rate was 58.56% for 4B-GRPO and 67.59% for 35B because the larger model scored 99.31% on controls.

The full guarded objective failed one of four checks:

- primary GRPO > SFT: pass;
- primary GRPO > 35B: pass;
- control regression no worse than -5pp: pass; GRPO improved over SFT by +15.28pp;
- wrong side-effecting actions no greater than SFT: **fail**, 88 vs 78 occurrences.

All forbidden hits were premature `migration_advisor` calls in the three primary scenarios. On a case-unique audit, GRPO affected 86 cases versus SFT's 70: 25 were introduced, 9 removed, net +16. The change was beneficial for `current_success_step2` (10→4 affected cases) but harmful for `fresh_retry_step2` (30→40) and `post_retry_success_step3` (30→42). This is the prospective target for a new-entity V13; V12 sealed rows must not be reused for training, checkpoint selection, or another V12 claim.

The shared 35B gateway initially failed because the launcher forced a direct `NO_PROXY` route on a host that requires its SOCKS proxy. The diagnosis and correction were frozen before valid predictions. Six later empty-response runtime rows were retried exactly once under an independently committed, correctness-blind policy; all six replacements were applied unconditionally. The original and corrected prediction files remain separately hashed and committed.

Primary artifacts:

- `sealed_objective.json` (`89b05e9706e0e8c68ff17395b48bcbb9ea351319203886115326328fc8ac000d`)
- `sealed_side_effect_audit.json` (`ad26648b88d4e38c61f0cfcceb319501be12737eb92ee613c6f4201738549dbe`)
- target prediction freeze commit `4dc597a`
- runtime-clean larger prediction freeze commit `079312f`
