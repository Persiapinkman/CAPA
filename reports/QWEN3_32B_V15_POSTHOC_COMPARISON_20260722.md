# Qwen3-32B 在 CAPA Planner V15 同集评测中的对比结果

_2026-07-22 UTC · 结论状态：有效的只读事后扩展，不属于原 V15 预注册四臂_

## 可发布结论

在完全相同的 24 个 V15 Planner cases、三轮完整运行、相同 prompt、mock observation、
JSON schema、parser、strict verifier 和 111:14 场景加权下，本地 Qwen3-32B FP16
（4×V100 tensor parallel）获得 `74.0 / 81.4 / 88.8%`，均值 `81.4%`。它在 V15 主分上
高于原始 4B-SFT `6.7111pp`，但低于 35B-A3B `10.8444pp`，也低于 4B-GRPO
`18.6000pp`。

因此加入 32B 后的主分排序是：

`4B Base < 4B SFT < Qwen3-32B < 35B-A3B < 4B GRPO`

32B 的三轮极差为 `14.8pp`，且只有第三轮超过 85%，所以它不能替代 35B 作为“本场景稳定
超过 85%”的参考模型。4B-GRPO 三轮均为 100%，仍然超过两个大模型。

## 主结果表：V15 预注册加权正确率

| 模型 | Run 1 (%) | Run 2 (%) | Run 3 (%) | Mean (%) | Range (pp) |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-4B Base | 14.8000 | 14.8000 | 14.8000 | **14.8000** | 0.0000 |
| Qwen3.5-4B original SFT | 75.0000 | 67.6000 | 81.4667 | **74.6889** | 13.8667 |
| **Qwen3-32B local FP16 TP4** | **74.0000** | **81.4000** | **88.8000** | **81.4000** | **14.8000** |
| Qwen3.5-35B-A3B | 87.9333 | 95.3333 | 93.4667 | **92.2444** | 7.4000 |
| Qwen3.5-4B targeted-SFT + one-step GRPO | 100.0000 | 100.0000 | 100.0000 | **100.0000** | 0.0000 |

主分计算式为：

`(111 × post_retry_metric_veto_step3 pass rate + 14 × current_success_step2 pass rate) / 125`

## 必须同时披露的不加权正确率

V15 主分有意把 metric-veto 权重设为 current-success 的 7.93 倍。32B 的两个子场景非常
不均衡，因此只报 81.4% 会掩盖明显缺陷。

| 模型 | 三轮 strict 合计 | Current-success | Metric-veto |
|---|---:|---:|---:|
| Qwen3.5-4B Base | 6/72 = 8.3333% | 0/36 = 0.0000% | 6/36 = 16.6667% |
| Qwen3.5-4B original SFT | 53/72 = 73.6111% | 26/36 = 72.2222% | 27/36 = 75.0000% |
| **Qwen3-32B** | **33/72 = 45.8333%** | **0/36 = 0.0000%** | **33/36 = 91.6667%** |
| Qwen3.5-35B-A3B | 54/72 = 75.0000% | 19/36 = 52.7778% | 35/36 = 97.2222% |
| Qwen3.5-4B GRPO | 72/72 = 100.0000% | 36/36 = 100.0000% | 36/36 = 100.0000% |

按 case 等权时，排序变为：

`4B Base 8.33 < Qwen3-32B 45.83 < 4B SFT 73.61 < 35B 75.00 < 4B GRPO 100.00`

这不是两个统计结果互相矛盾，而是目标函数不同：32B 在高权重的 metric-veto 上达到
91.67%，把加权均值抬到 81.4%；SFT 在两个类别更均衡，因此不加权分更高。

## 32B 逐类错误复核

### Current-success：0/36

每个 case 都先正确执行 detector，问题出在读取成功 observation 后的严格收口：

- 25/36 次选择了正确的 `decision_type=end`，但输出 `end_reason=recheck_done`；gold 要求
  `memory_hit`。这些是单字段 near miss，单 case verifier score 为 `0.956522`，但 strict
  pass 仍为 false。
- 11/36 次忽略“所有阈值均已满足，应结束”的结构化条件，错误进入
  `migration_advisor`。

因此 0/36 并非接口或 JSON 失败，而是模型对 Demo 特有的两种结束语义区分不稳定。

### Metric-veto：33/36

- Run 1：10/12；一条在第三步应迁移时再次调用 Qwen detector，另一条在第二步应重试 Rex
  detector 时提前迁移。
- Run 2：11/12；同一 Rex case 再次提前迁移。
- Run 3：12/12。

三轮相差的两个 strict case 每个贡献 `7.4pp` 加权分，恰好形成 32B 的 `14.8pp` 极差。

## 实验有效性

| 检查项 | 结果 |
|---|---:|
| 顶层预测覆盖 | 72/72 |
| 每轮 case ID | 24/24，精确匹配 V15 |
| Planner decisions | 60/60/60 |
| Prediction runtime error | 0 |
| JSON parse retry | 0 |
| `finish_reason=length` | 0 |
| 服务端 traceback / OOM / HTTP 500 | 0 / 0 / 0 |
| 输入 / 输出 token | 770,446 / 30,730 |
| 正式运行墙钟时间 | 约 22 分 56 秒 |

vLLM 0.8.5.post1 记录 `do_sample` 扩展字段被忽略；实际 SamplingParams 为
`temperature=0, top_p=1, top_k=-1, seed=42`，仍是 greedy。服务上下文设为 16,384，
只是为了容纳约 4k–5k prompt 加相同的 `max_tokens=4096`；三轮均无截断。

## 解释与使用建议

1. 32B 的通用规模优势主要体现在复杂的 retry/metric-veto 状态转移，不自动保证项目自定义的
   `memory_hit`/`recheck_done` 精确协议。
2. 4B-GRPO 的 100% 说明窄域后训练可以胜过更大的通用模型；这里证明的是 V15 严格 Planner
   路由，不是通用能力或真实视觉精度。
3. 如果线上业务比例接近 V15 的 111:14，应使用主加权分；如果两类任务同等重要，应看不加权
   strict 表，此时 32B 明显低于 SFT。
4. 32B、35B-A3B 与 4B 属于不同 Qwen 代际/架构，部署也不同，不能把差异归因于参数量本身。
5. 这是已打开 V15 上的事后扩展。可用于模型画像和工程选型，但不得改写成原始 sealed
   confirmation 的预注册第五臂。

## 复查入口

- 研究目录：[qwen3_32b_v15_posthoc_comparison_v1](../experiments/studies/qwen3_32b_v15_posthoc_comparison_v1/RESULT.md)
- 冻结协议：[PROTOCOL.md](../experiments/studies/qwen3_32b_v15_posthoc_comparison_v1/PROTOCOL.md)
- 机器报告：[comparison_report.json](../experiments/studies/qwen3_32b_v15_posthoc_comparison_v1/comparison_report.json)
- 原始输出：`/raid/zkq/artifacts/CAPA/evals/qwen3_32b_v15_posthoc_comparison_v1/formal_20260722T072244Z`

关键 SHA-256：

| 资产 | SHA-256 |
|---|---|
| V15 cases | `37d9a739585c012041397fba77b995d5842359ba5e23a8e8f8f604578e4ade78` |
| 原 V15 final report | `4bc4a819cf6f2291ac125c33b0a42133c679d1dbbb21d1dc54bb0337db8dcfd7` |
| 32B comparison report | `4ce0cd1e5c1ba6e8e1794a52d456b9acceff82adf1ef6cfd8a4eda3dad5cdb90` |
| 32B Run 1 predictions | `d0d9aac76640d55330043a96a02c5196a6b7015f28927d491020324d618739d7` |
| 32B Run 2 predictions | `f11433fe93acbafc81f25508a10e8a5e10fb44f15fd5bb4337c5ac95d1c97a8b` |
| 32B Run 3 predictions | `e3dc35dc6694cfae4409fcf16493e522afe05679dbf339e57edd275702733621` |
