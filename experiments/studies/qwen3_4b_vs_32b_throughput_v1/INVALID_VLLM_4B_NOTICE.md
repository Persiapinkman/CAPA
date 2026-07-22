# Qwen3.5-4B vLLM 数据有效性通知

`diagnostic_invalid_4b_vllm_formal.json` 及 `smoke_4b*.json` 仅用于诊断本地 vLLM
兼容路径。尽管请求成功、长度正确且吞吐稳定，自然语言 completion/chat sanity 输出出现明显
重复乱码，因此不得进入可发布模型吞吐对比。有效 4B 正式数据以
`raw_4b_hf_static_formal.json` 为准。
