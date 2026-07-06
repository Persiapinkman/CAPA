# Experiment Log

人工可读台账；每次实验一行摘要。机器可读元数据见 `manifest.jsonl`。

正式评测协议见 `EVALUATION_POLICY.md`：vLLM serving，`temperature=0`，`top_p=1`，`do_sample=false`，`seed=42`，每次评测 3 repeats 后取 aggregate。

| Date | Run ID | Type | Model | Adapter | Eval | Accuracy | Key Notes |
|---|---|---|---|---|---|---:|---|
| 2026-06-24 | `2026-06-24_base_local_t0` | baseline_eval | `qwen3.5-4b` | `none` | `results/planner_routing_eval/planner_routing_report_qwen35_4b_base_zip90_local.json` | 62/90 (0.6889) | 本地 transformers serve baseline，关闭 thinking，修复 content-list 与 V100 cuDNN 后重跑。 |
| 2026-06-25 | `2026-06-25_base_local_temp1` | baseline_eval_sampling | `qwen3.5-4b` | `none` | `results/planner_routing_eval/planner_routing_report_qwen35_4b_base_zip90_temp1.json` | 64/90 (0.7111) | 采样温度调到 1；总分与 DPO 持平，但通过集合不同，稳定性较差。 |
| 2026-06-24 | `2026-06-24_dpo_newarch` | dpo_train_eval | `qwen35-4b-newarch-dpo` | `outputs/planner-dpo-qwen35-4b-newarch-lora` | `results/planner_routing_eval/planner_routing_report_qwen35_4b_newarch_dpo_zip90.json` | 64/90 (0.7111) | 使用 DPO train seed v1 的 preference pairs 训练 LoRA。相对 base 净增 2 条，主要改善 historical 与少量 vision。 |
| 2026-07-03 | `2026-07-03_sft_chosen` | sft_train_eval | `qwen35-4b-sft-chosen` | `outputs/planner-sft-qwen35-4b-chosen-lora` | `results/planner_routing_eval/planner_routing_report_qwen35_4b_sft_chosen_zip90.json` | 62/90 (0.6889) | 只用 DPO chosen response 做 completion-only LoRA SFT；DDP 使用 gloo 后端绕过 NCCL driver/runtime 问题。 |
| 2026-06-24 | `packaged_4b` | packaged_reference_baseline | `packaged_4b` | `none` | `training/planner_dpo_train_seed_v1/eval/planner_routing_report_Qwen3.5-4B_90cases_baseline_summary.json` | 68/90 (0.7556) | Packaged reference summary from training bundle; no per-case elapsed timing available. |
| 2026-06-24 | `packaged_9b` | packaged_reference_baseline | `packaged_9b` | `none` | `training/planner_dpo_train_seed_v1/eval/planner_routing_report_Qwen3.5-9B_90cases_arch_rescored.json` | 76/90 (0.8444) | Packaged reference summary from training bundle; no per-case elapsed timing available. |
| 2026-06-24 | `packaged_35b_a3b` | packaged_reference_baseline | `packaged_35b_a3b` | `none` | `training/planner_dpo_train_seed_v1/eval/planner_routing_report_Qwen3.5-35B-A3B_90cases_baseline_summary.json` | 82/90 (0.9111) | Packaged reference summary from training bundle; no per-case elapsed timing available. |

## Timing Policy

- `run_planner_routing_eval.py` schema `1.1` 起，每个 case 写入 `started_at`、`finished_at`、`elapsed_ms`。
- `summary.timing` 记录评测总耗时、case 总耗时、平均/最小/最大 case 耗时。
- 旧报告作为 `historical_import` 保留，因当时未记录逐 case 耗时，台账中 `eval_efficiency.source=historical_import_no_eval_timing`。
