# Qwen3-32B on CAPA Planner V15：只读扩展对照协议

_冻结时间：2026-07-22 07:22:44 UTC · Study ID：`qwen3_32b_v15_posthoc_comparison_v1`_

## 结论边界

本实验把本地 `/raid/zkq/models/Qwen3-32B-vllm` 加到已经打开并完成的 V15
confirmation 数据上，回答“32B 在完全相同的 24 个 case、Planner prompt、mock
observation、parser、verifier 和加权公式下表现如何”。它是结果已知后的第五臂扩展，
不是原始 V15 的预注册臂，不改变原四臂 confirmation 的状态，也不能用于重新筛题、训练或
checkpoint 选择。

## 冻结输入与协议

- 数据：V15 全部 24 cases，SHA-256
  `37d9a739585c012041397fba77b995d5842359ba5e23a8e8f8f604578e4ade78`；禁止 case 级筛选。
- 原四臂来源：原 V15 `final_report.json`，SHA-256
  `4bc4a819cf6f2291ac125c33b0a42133c679d1dbbb21d1dc54bb0337db8dcfd7`。
- 32B：Qwen3-32B FP16，vLLM 0.8.5.post1，4×V100 tensor parallel；模型元数据哈希见
  `config.json`。
- 每轮固定 24 cases，共 3 轮；沿用原 35B 的 4 shards × 6 cases 并行布局。
- `temperature=0`、`top_p=1`、`seed=42`、`do_sample=false`、最大 3 Planner steps、
  每步最大 4096 tokens、请求与步骤 timeout 均为 600 秒。
- 和原 V15 35B 一致，模型请求不携带真实图片 payload；图片语义通过文本任务与 mock
  observation 测试 Planner 路由。
- 主分数严格沿用 V15：
  `(111 × metric-veto pass rate + 14 × current-success pass rate) / 125`。

服务上下文设为 16,384，而不是吞吐实验的 4,096。这只是为容纳约 4k–5k 输入加
`max_tokens=4096` 的请求上界；不会改变单请求解码参数。

## 校准与停止规则

正式运行前只对 offset 0 的一个固定 case 做一次 calibration。校准只检查 endpoint 模型
身份、HTTP 成功、JSON schema 可解析、prediction/reward 文件完整且没有 length finish；
正确或错误动作都不得用于改 prompt、模板、解码或服务参数。

正式输出目录必须不存在。正式开始后不允许补某一 case、shard 或 run。若发生明确的基础设施
失败，保留失败目录和日志，修复必须与模型正确性无关，并在新的目录从 72 条顶层预测整体重跑。
如果覆盖完整但模型输出错误，则结果照常计错，不得重跑。

## 有效性检查

最终比较只有在以下条件满足时标为 valid：每轮精确 24 行且 case ID 集合与源数据一致；三轮
共 72 行；无 prediction runtime error；无 `finish_reason=length`；评分报告和预测行一一对应。
同时报告两类各 12 case 的通过数、三轮加权分数、均值、极差，以及相对四个原始 V15 臂的
百分点差。Qwen3 与 Qwen3.5 不是同代模型，32B 与 35B-A3B 也不是同架构，因此比较只描述
本场景的实际检查点表现，不解释为纯参数规模因果效应。
