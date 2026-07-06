# Qwen3.5-4B local baseline temp=0

## Summary

- Run ID: `2026-06-24_base_local_t0`
- Type: `baseline_eval`
- Model: `qwen3.5-4b`
- Base model: `/mnt/zkq/models/Qwen3.5-4B`
- Adapter: `none`
- Eval report: `results/planner_routing_eval/planner_routing_report_qwen35_4b_base_zip90_local.json`
- Accuracy: `62/90 = 0.6889`

## Notes

本地 transformers serve baseline，关闭 thinking，修复 content-list 与 V100 cuDNN 后重跑。

## Efficiency

- Train efficiency: `not_applicable`
- Eval efficiency: `{'source': 'historical_import_no_eval_timing', 'elapsed_ms': None, 'case_elapsed_ms_total': None, 'case_elapsed_ms_avg': None}`

## Next Step

排查与 packaged 4B baseline 的 general_answer 差异。
