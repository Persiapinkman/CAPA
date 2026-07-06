# Qwen3.5-4B SFT on DPO chosen

## Summary

- Run ID: `2026-07-03_sft_chosen`
- Type: `sft_train_eval`
- Model: `qwen35-4b-sft-chosen`
- Base model: `/mnt/zkq/models/Qwen3.5-4B`
- Adapter: `outputs/planner-sft-qwen35-4b-chosen-lora`
- Eval report: `results/planner_routing_eval/planner_routing_report_qwen35_4b_sft_chosen_zip90.json`
- Accuracy: `62/90 = 0.6889`

## Notes

只用 DPO chosen response 做 completion-only LoRA SFT；DDP 使用 gloo 后端绕过 NCCL driver/runtime 问题。

## Efficiency

- Train efficiency: `{'train_runtime_s': 113.7, 'train_samples_per_second': 0.897, 'train_steps_per_second': 0.185, 'train_loss': 2.081}`
- Eval efficiency: `{'source': 'historical_import_no_eval_timing', 'elapsed_ms': None, 'case_elapsed_ms_total': None, 'case_elapsed_ms_avg': None}`

## Next Step

该 SFT 不优于 DPO；后续不要沿用单纯 chosen SFT，除非扩充 general_answer 数据。
