# Planner retry/migrate V6 exploratory data analysis

_Generated 2026-07-15 from the frozen JSONL cases and Qwen3.5 stage artifacts_

---

## 📋 Executive summary

The V6 family contains **1,875 cases, 250 entities, 1,500 matched core cases, and 375 guard cases**. All five splits pass schema, trajectory, canonical-reward, fixture, cross-split isolation, and repository-overlap checks. The frozen SFT and GRPO artifacts add 1,840 rows with unique prompts and unit canonical step reward.

The main analytical result is that the intended routing feature is identifiable: every core bundle supplies both retry and migrate outcomes while holding the entity, detector, query, alias, badge, target, and fixture constant. Badge/action and alias/action mutual information are zero to floating-point tolerance. This directly removes the two shortcuts observed in V5/V5-train-v1.

## 💾 Basic information

| Artifact | Rows | Approximate size | Purpose |
| --- | ---: | ---: | --- |
| Full case JSONLs | 1,875 | 6.6 MiB | Trajectory truth and rollout evaluation |
| SFT train/dev | 1,300 | 26 MiB | Qwen3.5 non-thinking supervised warm-up |
| GRPO train/dev | 540 | 10.4 MiB | Step-2 transition optimization and validation |
| Human-review sample | 45 | Under 0.1 MiB | Independent semantic review |

Each JSONL line is a nested object. Required top-level fields include role flags, entity and bundle identifiers, query, fixture metadata, full expected decisions, mock observations, forbidden actions, and the reward specification.

## 📊 Distribution analysis

| Split | Cases | Entities | Retry / migrate / end at target | Three-step cases |
| --- | ---: | ---: | ---: | ---: |
| `sft_train` | 600 | 80 | 160 / 400 / 40 | 160 |
| `sft_dev` | 150 | 20 | 40 / 100 / 10 | 40 |
| `grpo_train` | 450 | 60 | 120 / 300 / 30 | 120 |
| `grpo_dev` | 225 | 30 | 60 / 150 / 15 | 60 |
| `test` | 450 | 60 | 120 / 300 / 30 | 120 |

The GRPO artifact intentionally includes only the core step-2 states: 360 train rows and 180 dev rows. Its action mix is 1:2 retry:migrate, arising from one fresh-retryable and two migrate states per matched bundle. Guard cases remain available for full-rollout SFT/dev/test checks rather than being mixed into the GRPO transition objective.

Post-retry outcomes are balanced exactly where split size permits. The 60-entity GRPO train and test splits each contain 40 success, 40 metric-veto, and 40 technical-error trajectories. The 30-entity GRPO dev split contains 20 of each.

## 🔍 Quality assessment

### Completeness and validity

- All 1,875 case IDs are present and unique
- Every retry trajectory has three decisions and two observations at `after_step=1` and `after_step=2`
- All other trajectories have two decisions and one observation
- Every referenced image fixture exists; all ten fixture hashes are distinct
- All 1,875 canonical full trajectories pass the strict case reward
- An independent field parser reproduces all 2,375 next-transition labels without reading scenario IDs
- All 1,840 frozen stage rows receive canonical step reward `1.0`
- JSONL round-trip row counts match the in-memory build

### Isolation

Ten pairwise split comparisons have zero overlap on all protected fields. The repository audit also reports zero overlap against the other case files on case ID, entity, project, target, normalized query, template, error alias, fixture family, fixture path, and fixture hash.

### Shortcut diagnostics

Within every core bundle, badge and alias are fixed while the action changes. Consequently:

- Badge/action mutual information is `0` or below `1.2e-16` bits across splits
- Alias/action mutual information is `0` or below `1.8e-16` bits across splits
- Every badge condition supports both retry and migrate
- Every non-sentinel error alias supports both retry and migrate

The tiny nonzero values are floating-point roundoff, not measurable dependence.

## 📈 Prompt analysis

| Stage | Rows | Min tokens | Mean tokens | P95 tokens | Max tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT train | 1,040 | 4,056 | 4,263.2 | 4,460 | 4,601 |
| SFT dev | 260 | 4,052 | 4,251.8 | 4,451 | 4,590 |
| GRPO train | 360 | 4,238 | 4,255.9 | 4,268 | 4,272 |
| GRPO dev | 180 | 4,234 | 4,248.4 | 4,257 | 4,261 |

Every prompt is under the 4,608-token hard gate. The longer SFT tail comes from step-3 and stale-history examples. Prompt audits found zero case/entity IDs, absolute paths, persistence references, training-instruction phrases, max-step mismatches, or duplicate prompt hashes.

## 🎯 Key findings

1. **The decision variable is locally identifiable.** The comparison no longer depends on learning a global entity or alias association.
2. **Retry is a real transition rather than a terminal label.** Each fresh-retry case consumes a second observation and then ends or migrates.
3. **SFT and GRPO have separate entities.** GRPO therefore measures and optimizes transfer of the rule rather than continued memorization of SFT entities.
4. **Dev and test are separate from both optimization stages.** Templates, aliases, targets, queries, and fixtures are also disjoint.
5. **Reward parity is restored.** The step scorer now treats `final_tool_finish` as applicable only when the gold final tool explicitly requires `finish_after_tool=true`.

## ⚠️ Remaining risks

- The 45-case independent human semantic review is not yet signed off
- Synthetic policy wording may overrepresent explicit threshold-following compared with production requests
- Long system prompts leave little margin under a 4,608-token context gate for future tool-schema growth
- Passing canonical reward only verifies scorer/gold consistency; it does not prove that sampled model outputs provide useful GRPO variance
- The sealed-test policy relies on operational discipline because the file is locally readable

## 🔄 Recommended next analysis

Before training, complete the independent human review and freeze its decisions. After SFT, report full-trajectory accuracy and per-transition confusion matrices on `sft_dev`, `grpo_dev`, and the untouched test protocol. Before GRPO, rerun sampling-support audits specifically on the 180 `grpo_dev` matched transitions and require nonzero support for retry in both detector families.

The complete numeric payload is available in [eda_summary.json](eda_summary.json), and hash-level provenance is in [manifest.json](manifest.json).
