# H20 三场景 3x 评测对照

_Repro root_: `/apdcephfs_hzlf/share_1227201/zkq/capa_h20/artifacts/CAPA/repro_h20`

_Eval root_: `/apdcephfs_hzlf/share_1227201/zkq/capa_h20/artifacts/CAPA/repro_h20/eval`


Deterministic settings: `temperature=0 top_p=1 seed=42 runs=3`; each row is 3-run mean.


## S1. 单步工具路由 (`planner_routing_eval_90cases`, 90 case)

| Arm | case_macro_mean | case_pass_rate | runs | source |
|---|---:|---:|---:|---|
| base_4b | — | 0.7111 | 3 | `20260801_123112_base_4b/routing90` |
| base_4b _(gateway historical)_ | — | 0.9074 ± 0.0064 | 3 | `results/planner_routing_eval/*_3x_aggregate.json` |
| sft | — | — | — | — |
| grpo42 | — | — | — | — |
| grpo43 | — | — | — | — |
| grpo44 | — | — | — | — |
| base_35b | — | 0.7148 | 3 | `20260801_125005_base_35b/routing90` |
| base_35b _(gateway historical)_ | — | 0.9444 ± 0.0000 | 3 | `results/planner_routing_eval/*_3x_aggregate.json` |

## S2. 多步工具路由 (`planner_grpo_focused_val_v3`, 31 case)

| Arm | case_macro_mean | case_pass_rate | runs | source |
|---|---:|---:|---:|---|
| base_4b | 0.8032 | 0.3871 | 3 | `20260801_123112_base_4b/multistep` |
| sft | — | — | — | — |
| grpo42 | — | — | — | — |
| grpo43 | — | — | — | — |
| grpo44 | — | — | — | — |
| base_35b | 0.8935 | 0.3656 | 3 | `20260801_125005_base_35b/multistep` |

## S3. 软边界状态 (`planner_retry_migrate_v6_grpo_dev`, 225 case)

| Arm | case_macro_mean | case_pass_rate | runs | source |
|---|---:|---:|---:|---|
| base_4b | 0.5133 | — | 3 | `20260801_123112_base_4b/softbnd_dev` |
| sft | — | — | — | — |
| grpo42 | — | — | — | — |
| grpo43 | — | — | — | — |
| grpo44 | — | — | — | — |
| base_35b | 0.5157 | — | 3 | `20260801_125005_base_35b/softbnd_dev` |

## Notes
- softbnd_dev 请按 `entity_id` / `counterfactual_bundle_id` 聚合再解读；此表按 case-macro 呈现。
- routing90 的 gateway 行是 V100 时代远端网关的 3x 基线，仅供噪声范围参考，不是 pass/fail 判据。
