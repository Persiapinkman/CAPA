# Dataset Card: `planner_retry_migrate_v7_longobs`

_Frozen 2026-08-01 · seed 2026080101 · 2000 cases across 250 entities_

## 1. Story line

Real deployments never hand the planner a "retryable=true, retry_count=0" flag. What
the planner actually sees is a *tool response object* — a detector's structured
JSON payload, some server telemetry, an optional trace of prior attempts, and one
or more knowledge-base excerpts fetched by upstream retrievers. Whether to retry,
migrate to the capability-migration advisor, or terminate the query has to be
derived from that raw material.

The v6 dataset trained on a synthetic shortcut: every case carried an
observation summary of the form
`retryable=false; retry_count=0; gateway_error=detector_admission_window_full`,
which is precisely the field the planner is expected to key on. SFT hit 0.98
`action_match` in sixty optimizer steps and GRPO's support gate found
`nonzero_reward_variance_rate = 1.1%`, so the study concluded that "the core-only
step-2 split is already nearly saturated". Great label mask, wrong task.

v7 tells the same story — retry vs. migrate vs. end — but forces the planner to
read a **1.5-4k-token tool observation** and *derive* the routing signal:

| Signal | Where it lives in v7 |
|---|---|
| `retryable` (error class recoverable?) | `detector_response.error.class_label` + `error.message` free text |
| `retry_count` (budget consumed?) | `session_history[*].attempted_action / attempted_result_summary` |
| `iou_between_probes` | model must compute IoU between `objects[i].bbox_xyxy` and `second_probe_objects[i].bbox_xyxy` |
| `candidate_count`, `min_confidence` | count / min-reduce over `detector_response.objects` |
| `domain_shift` | `detector_response.meta.tags` includes `night_scene / thermal_ir / sketch_style / sar_grayscale` |
| stale-history conflict (G2) | prior attempt claims success in a different lighting / camera view; current response fails |

The `user_query` describes only the business goal (e.g. "assess whether 光伏电站巡检回路 9933 号 can reuse the existing 岩灰楔形挡片 detector"); the routing rules live only in the system prompt.

## 2. Splits and totals

| Split | Entities | Cases | Role |
|---|---:|---:|---|
| `sft_train` | 80 | 640 | SFT training |
| `sft_dev` | 20 | 160 | SFT model selection |
| `grpo_train` | 60 | 480 | GRPO optimizer pool |
| `grpo_dev` | 30 | 240 | GRPO support gate + selection |
| `test` | 60 | 480 | **sealed**, one-shot final eval |
| **Total** | **250** | **2000** | |

Splits are entity-disjoint. Each entity contributes exactly one case per scenario
(8 cases). Scenarios are matched: within a given entity the badge and detector
nuisance factors are shuffled independently of the scenario so
`MI(badge, target_action)` and `MI(detector, target_action)` both approach 0
(measured: 0.00025 and 0.000 respectively; threshold 0.02).

## 3. Scenario grid

| Scenario | Role | Target | Discriminating signal |
|---|---|---|---|
| P1 `iou_low_fresh` | primary | **retry** | Two probe bboxes disagree (IoU ≈ 0.55) even though the detector returned candidates |
| P2 `all_gates_ok` | primary | **end** | Single candidate, confidence ≥ 0.90, IoU ≥ 0.90, no domain-shift tag |
| P3 `transient_5xx` | primary | **retry** | `error.class_label ∈ {transport_5xx, gateway_timeout}` on first attempt |
| P4 `auth_quota` | primary | **migrate** | `error.class_label ∈ {authorization_denied, quota_exceeded}`; retry cannot recover |
| P5 `second_failure` | primary | **migrate** | `session_history[0]` shows a prior same-detector attempt with the same failure class; budget exhausted |
| P6 `domain_shift` | primary | **migrate** | Detector succeeded but `meta.tags` contains a domain shift signal → current asset uncovered |
| G1 `first_success_end` | **guardrail** | **end** | Clean single-shot success; planner must not reflexively migrate after every tool call |
| G2 `conflict_stale_history` | **guardrail** | **migrate** | Prior success claim in history contradicts the current failure observation; must trust current |

## 4. Observation size (measured)

| Statistic | Value |
|---|---:|
| `approx_tokens` min | 1866 |
| `approx_tokens` mean | 2421 |
| `approx_tokens` p50 | 1950 |
| `approx_tokens` p95 | 3962 |
| `approx_tokens` max | 4058 |
| min required (audit gate) | 1500 |

Approximation: 1 token per 1.8 CJK chars + 1 per 3.6 ASCII chars. Real tokenizer
count is slightly higher; the frozen minimum ensures observations always exceed
the 1500 target.

## 5. Integrity gates (all must pass at build time)

1. **No trigger substrings** in any observation:
   `retryable= / retry_count= / gateway_error= / domain_shift= /
   candidate_count= / min_confidence= / cross_prompt_iou= /
   retryable: / retry_count:`.
2. **No rule leakage** in any `user_query`:
   `retryable= / retry_count / 候选数 / 跨提示 / IoU / 域偏移 / fresh retryable`.
3. **No metadata leak**: `case_id`, `entity_id` do not appear inside the query text.
4. **Entity-disjoint** across all five splits.
5. **Nuisance MI**: `MI(badge, target_action) < 0.02` and
   `MI(detector_family, target_action) < 0.02`.
6. **Fixture SHA256** unique per entity; regenerated deterministically.

Any failure aborts the builder with a non-zero exit code — no partial or "best
effort" dataset is written.

## 6. Contract with the rollout pipeline

- Cases use the exact same schema as `planner_retry_migrate_v6` (same fields,
  same `mock_observations[after_step].observation` dict shape), so
  `training/planner_grpo_seed_v1/scripts/run_planner_grpo_rollout.py` needs no
  changes.
- The planner's `MemoryProjector` reads `observation["summary"]` first (short,
  ≤600 chars for working-memory extraction) and then json-dumps the whole dict
  into `query_trajectories` for the planner's next-step prompt. The long fields
  (`detector_response`, `session_history`, `technical_notes`) surface through
  the second path — this is why they are populated as top-level keys.

## 7. Scoring

Sealed evaluation is scored by the frozen
`training/planner_grpo_seed_v1/scripts/reward_planner_grpo.py`. The primary
metric reported per case is `mean_rule_reward` (a normalised 0-1 combination of
`action_match / argument_match / decision_type_valid / final_tool_finish /
finish_after_tool / json_valid / no_forbidden_action / no_premature_stop /
no_repeated_tool / no_skip_required_probe`). Aggregation is case-macro grouped
by `counterfactual_bundle_id`; comparisons use bootstrap paired 95% CI (2000
draws).

## 8. Files

```
data/datasets/planner_retry_migrate_v7_longobs/
├── manifest.json                # totals, sha256, MI check
└── DATASET_CARD.md              # this file

training/planner_grpo_seed_v1/cases/
├── planner_retry_migrate_v7_longobs_sft_train_cases.jsonl     640 rows
├── planner_retry_migrate_v7_longobs_sft_dev_cases.jsonl       160
├── planner_retry_migrate_v7_longobs_grpo_train_cases.jsonl    480
├── planner_retry_migrate_v7_longobs_grpo_dev_cases.jsonl      240
└── planner_retry_migrate_v7_longobs_test_cases.jsonl          480 (sealed)

examples/images/planner_retry_migrate_v7_longobs/
└── <adj>_<shape>_<code>.png    (one deterministic 256×256 fixture per entity)
```

## 9. Regeneration

Deterministic. From the repository root:

```
.venv-h20-infer/bin/python \
  training/planner_grpo_seed_v1/scripts/build_planner_retry_migrate_v7_longobs.py \
  --min-obs-tokens 1500
```

Any change that flips a byte in the output JSONL will change the SHA256 recorded
in `manifest.json`; downstream studies must re-freeze before use.

## 10. Known limitations

- Detector responses are synthesised, not sampled from a real detector service.
  IoU numbers are constructed from templates; a future v8 could replace them
  with real MSCOCO 2017-val Grounding-DINO / Qwen-VL rollouts.
- The `technical_notes` corpus is a fixed 6-entry seed pool with rotation, so
  cross-case n-gram novelty at the tech-notes level is limited; the routing
  signal never lives in `technical_notes`, so this is a distraction-quality
  issue rather than a correctness issue.
- G1 is intentionally frequency-balanced (1/8 of cases). Deployment traffic
  will be more skewed; the planner still trains on the balanced signal to
  prevent "always migrate" bias, and the sealed test measures per-scenario pass
  rates separately.
