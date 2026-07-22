# Qwen3-32B on V15：结果摘要

_完成时间：2026-07-22 07:52 UTC · 状态：`valid_posthoc_extension`_

本地 Qwen3-32B 已在原 V15 的全部 24 cases 上完成三轮固定评测，共 72 条顶层预测。
覆盖率为 72/72，prediction runtime error、JSON parse retry、长度截断和服务端 fatal error
均为 0。

## V15 主加权分

| 模型 | Run 1 (%) | Run 2 (%) | Run 3 (%) | Mean (%) | Range (pp) |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-4B Base | 14.8000 | 14.8000 | 14.8000 | 14.8000 | 0.0000 |
| Qwen3.5-4B original SFT | 75.0000 | 67.6000 | 81.4667 | 74.6889 | 13.8667 |
| **Qwen3-32B local FP16 TP4（事后扩展）** | **74.0000** | **81.4000** | **88.8000** | **81.4000** | **14.8000** |
| Qwen3.5-35B-A3B | 87.9333 | 95.3333 | 93.4667 | 92.2444 | 7.4000 |
| Qwen3.5-4B targeted-SFT + one-step GRPO | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 0.0000 |

主分排序为：

`4B Base 14.80 < 4B SFT 74.69 < Qwen3-32B 81.40 < 35B-A3B 92.24 < 4B GRPO 100.00`

32B 相对 Base、SFT、35B 和 GRPO 的均值差分别为 `+66.6000pp`、`+6.7111pp`、
`-10.8444pp` 和 `-18.6000pp`。它只有 Run 3 超过 85%，三轮极差 14.8pp，不能视为
本场景中稳定超过 85% 的 larger-model 参考。

## 权重敏感性

V15 主分按 `metric-veto:current-success = 111:14` 加权。32B 在 metric-veto 上很强，
但 current-success 严格通过为 0，因此必须同时报告不加权结果：

| 模型 | 不加权 strict | Current-success | Metric-veto |
|---|---:|---:|---:|
| Qwen3.5-4B Base | 6/72 = 8.3333% | 0/36 = 0.0000% | 6/36 = 16.6667% |
| Qwen3.5-4B original SFT | 53/72 = 73.6111% | 26/36 = 72.2222% | 27/36 = 75.0000% |
| **Qwen3-32B** | **33/72 = 45.8333%** | **0/36 = 0.0000%** | **33/36 = 91.6667%** |
| Qwen3.5-35B-A3B | 54/72 = 75.0000% | 19/36 = 52.7778% | 35/36 = 97.2222% |
| Qwen3.5-4B GRPO | 72/72 = 100.0000% | 36/36 = 100.0000% | 36/36 = 100.0000% |

所以“32B 高于 SFT”只对预注册的主加权分成立；按每个 case 等权，32B 反而比 SFT 低
`27.7778pp`。不能脱离 111:14 场景权重只发布 81.4%。

## 32B 错误结构

- 36 次 current-success 中，25 次正确选择 `end`，但把 `end_reason` 写成
  `recheck_done`，而严格 gold 为 `memory_hit`；另外 11 次错误调用 `migration_advisor`。
- 36 次 metric-veto 中通过 33 次。3 次失败包括：1 次在第三步再次调用 detector，2 次在
  第二步应重试 detector 时提前迁移。
- 三轮各产生 60 个 Planner decisions；没有空 decision、parse retry、length finish、HTTP
  error、OOM 或服务进程异常。

这说明 32B 已较好学会“错误后重试、指标否决后迁移”的主路径，但没有稳定遵守该 Demo 的
精确结束语义。专门训练的 4B GRPO 在两类上均为 100%，显著优于通用 32B。

## 证据边界与路径

32B 是 V15 已打开后的只读扩展臂，不是原始预注册的第五臂，不能改变原 V15 confirmation
结论。Qwen3-32B 与 Qwen3.5 模型也不是纯参数规模对照。

- 冻结协议：[PROTOCOL.md](PROTOCOL.md)
- 冻结配置：[config.json](config.json)
- 机器报告：[comparison_report.json](comparison_report.json)，SHA-256
  `4ce0cd1e5c1ba6e8e1794a52d456b9acceff82adf1ef6cfd8a4eda3dad5cdb90`
- 表格：[comparison_table.md](comparison_table.md)，SHA-256
  `03c598bad3f19a65f776e5c0a29b0fba42003bcc7afe83a6fae35a33752eabac`
- 原始证据：`/raid/zkq/artifacts/CAPA/evals/qwen3_32b_v15_posthoc_comparison_v1/formal_20260722T072244Z`
- 完整中文报告：[QWEN3_32B_V15_POSTHOC_COMPARISON_20260722.md](../../../reports/QWEN3_32B_V15_POSTHOC_COMPARISON_20260722.md)
