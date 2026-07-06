# Qwen3.5-4B local baseline temp=1

## Summary

- Run ID: `2026-06-25_base_local_temp1`
- Type: `baseline_eval_sampling`
- Model: `qwen3.5-4b`
- Base model: `/mnt/zkq/models/Qwen3.5-4B`
- Adapter: `none`
- Eval report: `results/planner_routing_eval/planner_routing_report_qwen35_4b_base_zip90_temp1.json`
- Accuracy: `64/90 = 0.7111`

## Notes

采样温度调到 1；总分与 DPO 持平，但通过集合不同，稳定性较差。

## Efficiency

- Train efficiency: `not_applicable`
- Eval efficiency: `{'source': 'historical_import_no_eval_timing', 'elapsed_ms': None, 'case_elapsed_ms_total': None, 'case_elapsed_ms_avg': None}`

## Next Step

不要作为主 baseline；可补测 temperature=0.7/0.8 验证采样敏感性。
