# Qwen3.5-4B DPO LoRA newarch

## Summary

- Run ID: `2026-06-24_dpo_newarch`
- Type: `dpo_train_eval`
- Model: `qwen35-4b-newarch-dpo`
- Base model: `/mnt/zkq/models/Qwen3.5-4B`
- Adapter: `outputs/planner-dpo-qwen35-4b-newarch-lora`
- Eval report: `results/planner_routing_eval/planner_routing_report_qwen35_4b_newarch_dpo_zip90.json`
- Accuracy: `64/90 = 0.7111`

## Notes

使用 DPO train seed v1 的 preference pairs 训练 LoRA。相对 base 净增 2 条，主要改善 historical 与少量 vision。

## Efficiency

- Train efficiency: `{'train_runtime_s': 861.8, 'train_samples_per_second': 0.118, 'train_steps_per_second': 0.015, 'train_loss': 0.6513}`
- Eval efficiency: `{'source': 'historical_import_no_eval_timing', 'elapsed_ms': None, 'case_elapsed_ms_total': None, 'case_elapsed_ms_avg': None}`

## Next Step

补充 general_answer > rag_answer hard negatives 后再训 v2。
