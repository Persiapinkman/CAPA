---
title: 4B 推理路径有效性修订
date: 2026-07-21
status: frozen-before-replacement-formal-measurement
amends: PROTOCOL.md
---

# 4B 推理路径有效性修订

## 1. 修订触发条件

原协议中的 Qwen3.5-4B vLLM 0.10.2 V0 路径完成了 30 个 trial，但随后执行的自然语言
completion 与 chat sanity check 均出现明显重复乱码，例如连续重复 `user` 和编号片段。该路径
满足 HTTP、长度和稳定性检查，却不满足“有效模型输出”这一更高优先级条件。因此：

- 原 4B vLLM 性能数据降级为兼容性诊断，不进入吞吐名次；
- 32B vLLM 数据已通过自然语言 sanity check，继续作为有效对照；
- 不根据 4B/32B 的快慢调整任何工作负载水平或正式 block 数。

根因定位表明原始 4B 权重没有损坏。相同权重通过 Transformers
`AutoModelForCausalLM` 加载为 `Qwen3_5ForCausalLM`、FP16、SDPA 后能给出正常回答。
无效输出来自 Qwen3.5 multimodal 包装配置与本地 vLLM generic Transformers backend 的
兼容路径。

## 2. 替代 4B 路径

有效 4B 结果改用单卡 Transformers 同步静态 batch：

| 字段 | 修订后 4B 配置 |
|---|---|
| 源检查点 | `/raid/zkq/models/Qwen3.5-4B` |
| 服务检查点 | `/raid/zkq/artifacts/CAPA/bench_models/Qwen3.5-4B-text-only-fp16` |
| 导出契约 | `Qwen3_5ForCausalLM`；4,205,751,296 参数；无 visual module；加载 missing/unexpected/mismatched 均为 0 |
| 框架 | Transformers 5.13.0.dev0、PyTorch 2.8.0+cu128 |
| 精度与 attention | FP16、SDPA、cuDNN disabled |
| GPU | 1 × V100-SXM2-32GB，物理 GPU 0 |
| 执行方式 | 每个 wave 将同步到达的 C 个等长请求组成一个静态 batch；每 trial 两个 wave |

尝试过 Transformers 原生 continuous batching，但它在初始化 paged cache 时明确报错
`Invalid group type: linear_attention`；该失败发生在正式替代采样前并保留日志，不计数据。

## 3. 保持不变的设计

- ISL 仍为 256、2,048；OSL 仍固定 128；
- C 仍为 1、4、16。32B 中 C 表示同时到达的 HTTP 请求数；4B 中 C 表示同步静态 batch
  大小。两者每 trial 都处理 `2 × C` 个请求，构成两个 wave；
- 使用相同生成语料、模型各自 tokenizer 下精确相同 token 长度、相同随机种子；
- greedy、忽略 EOS、固定生成 128 token；
- 正式 block 仍为 5，block 内条件顺序仍按 seed `20260721` 随机化；
- 吞吐、每 GPU 吞吐、功率和能耗仍按 trial 级 bootstrap 统计。

## 4. 修订后的结论边界

主要比较是“有效且当前可运行的两套本地推理栈”在同步工作负载下的容量：

- 4B：Transformers 静态 batch，1 GPU；
- 32B：vLLM continuous batching / TP4，4 GPU。

框架和调度器是处理的一部分，因此仍不能解释为纯模型参数规模效应。4B 静态 batch 没有流式
请求层，故不发布 4B TTFT，也不进行跨模型 TTFT 排名；可比较指标限于 output tok/s、
tok/s/GPU、完整 batch/request latency、功率与能耗。

替代 4B 路径必须在正式采样前通过以下附加门槛：

1. 模型类严格为 `Qwen3_5ForCausalLM`；
2. 加载 missing、unexpected、mismatched、error keys 全为空；
3. 自然语言 sanity 输出长度至少 20 字符且 token unique ratio 不低于 0.20；
4. 每个正式序列严格生成 128 token；
5. 30/30 trial 有效且无 CUDA/OOM/非有限输出错误。
