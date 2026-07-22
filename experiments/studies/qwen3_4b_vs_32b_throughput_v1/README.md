# Qwen3.5-4B vs Qwen3-32B throughput study

最终结论与发布边界见
[`reports/QWEN3_4B_VS_32B_THROUGHPUT_20260721.md`](../../../reports/QWEN3_4B_VS_32B_THROUGHPUT_20260721.md)。

主要有效数据：

- `raw_4b_hf_static_formal.json`：Qwen3.5-4B、1×V100、Transformers 静态 batch；
- `raw_32b_formal.json`：Qwen3-32B、4×V100 TP、vLLM 服务；
- `analysis/SUMMARY_TABLES.md`：自动生成统计；
- `environment_manifest.json`：环境、模型和文件哈希。

`diagnostic_invalid_4b_vllm_formal.json` 是输出语义无效的 vLLM 兼容性诊断，不得用于
模型吞吐排名。原因和替代设计见 `PROTOCOL_AMENDMENT_VALIDITY.md`。
