---
title: Qwen3.5-4B 与 Qwen3-32B 本地部署吞吐预注册协议
date: 2026-07-21
status: superseded-by-validity-amendment
study_id: qwen3_4b_vs_32b_throughput_v1
---

# Qwen3.5-4B 与 Qwen3-32B 本地部署吞吐预注册协议

> **有效性更新（2026-07-21）：** 原预注册的 4B vLLM 路径虽然完成性能采样，但自然语言
> sanity check 出现重复乱码，不能作为可发布模型吞吐。原始协议与数据保留供审计；主要结论
> 改按 [PROTOCOL_AMENDMENT_VALIDITY.md](PROTOCOL_AMENDMENT_VALIDITY.md) 执行。

## 1. 研究问题与结论边界

本研究比较同一台服务器上的两套**实际可部署配置**：单卡 Qwen3.5-4B 与四卡张量并行
Qwen3-32B。主要问题是在固定输入、固定输出和固定并发下，两套服务分别能提供多少请求吞吐
和 token 吞吐，以及这些吞吐在按 GPU 数量归一化后有何差异。

这不是严格的“只改变参数规模”的因果实验。两个检查点属于相邻但不同的 Qwen 代际和架构，
并且由于本机软件兼容性约束，4B 与 32B 使用不同 vLLM 小版本。因此，允许发布的结论是
“在本文记录的硬件和部署栈上，配置 A 与配置 B 的服务性能差异”，不能外推为所有硬件、
所有推理引擎或所有 Qwen 4B/32B 模型的固有差异。

## 2. 被比较的部署

| 字段 | 4B 部署 | 32B 部署 |
|---|---|---|
| 原始检查点 | `/raid/zkq/models/Qwen3.5-4B` | `/raid/zkq/models/Qwen3-32B-vllm` |
| 服务检查点 | 独立兼容副本；仅移除未使用的 `mtp.*` 推测头并改写 architecture 声明 | 原目录只读加载 |
| 模型系列 | Qwen3.5-4B | Qwen3-32B |
| 精度 | FP16 | FP16 |
| GPU | 1 × Tesla V100-SXM2-32GB，物理 GPU 0 | 4 × Tesla V100-SXM2-32GB，物理 GPU 0–3 |
| 并行方式 | 单卡 | tensor parallel = 4 |
| 服务引擎 | vLLM 0.10.2 V0，Transformers model implementation | vLLM 0.8.5.post1 V0，原生 Qwen3 implementation |
| PyTorch | 2.8.0+cu128 | 2.6.0+cu124 |
| 最大上下文 | 4,096 tokens | 4,096 tokens |
| eager mode | 开启 | 开启 |
| prefix cache | 关闭 | 关闭 |
| chunked prefill | 关闭 | 关闭 |
| max batched tokens | 8,192 | 8,192 |
| max sequences | 16 | 16 |
| GPU memory utilization | 0.85 | 0.85 |

选择不同 vLLM 版本是预先确定的兼容性约束：Qwen3.5 需要项目中已验证的 0.10.2
Transformers backend；32B 的四卡配置必须使用 CUDA 12.4 兼容的 0.8.5.post1，原因是
主机 Driver 550.163.01 无法支持 `.venv-train` 中 PyTorch 2.8/cu128 的多卡 NCCL 路径。
该差异将作为部署栈的一部分报告，而不是隐藏为模型差异。两边都固定使用 V0 scheduler；
4B 的 V1 calibration 仅用于发现“V1 强制打开 chunked prefill”的行为，不进入正式数据。

## 3. 工作负载矩阵

所有请求均调用本机回环地址的 OpenAI-compatible `/v1/completions`，发送 token ID 数组，
绕过客户端分词时间。prompt 由固定自然语言语料确定性生成，逐请求不同；每个模型使用自己的
tokenizer 截取到精确长度。关闭 prefix cache，避免重复前缀产生虚高结果。

| 因素 | 水平 |
|---|---|
| 输入长度（ISL） | 256、2,048 tokens |
| 输出长度（OSL） | 固定 128 tokens；`min_tokens=128`、`max_tokens=128`、`ignore_eos=true` |
| 并发数 | 1、4、16 |
| 解码 | greedy，`temperature=0` |
| 每个 trial 的请求数 | `2 × concurrency`，即两个闭环请求波次 |
| 正式区组数 | 5 |
| 条件数 | 2 ISL × 3 concurrency = 6 |
| 每模型正式 trial 数 | 30 |

每个模型启动后先执行不进入结果的健康检查、固定长度校验和每个条件一个预热波次。正式阶段
在每个 block 内随机打乱六个条件，随机种子固定为 `20260721`。模型顺序因 32B 加载成本而
按部署分区执行；32B 完成后重新启动 4B，对 256/128 的 concurrency 1 和 16 做锚点复测，
用于检查时间漂移，但锚点不并入主要估计。

## 4. 观察指标

### 4.1 主要指标

1. 服务输出吞吐：成功请求的 completion tokens / trial wall time，单位 output tok/s。
2. 每 GPU 输出吞吐：服务输出吞吐 / 分配 GPU 数量，单位 output tok/s/GPU。
3. 在测试并发范围内的峰值吞吐：每个 ISL 下三个并发均值中的最大值。它不声称是未经测试
   更高并发下的全局饱和值。

### 4.2 次要指标

- request/s、prompt tok/s、total tok/s；
- 请求端到端延迟 P50/P95；
- TTFT P50/P95；
- TPOT，其中 `TPOT = (E2E - TTFT) / (completion_tokens - 1)`；
- 分配 GPU 的平均/峰值利用率、显存、功率与 trial 能耗；
- 服务加载至健康检查成功的时间；
- HTTP 错误率、固定 token 长度符合率。

端到端延迟分位数使用全部请求作描述性统计。吞吐推断的独立单位是 trial/block，不把同一
trial 内的请求或 token 伪装成独立重复。

## 5. 统计计划

每个 `model × ISL × concurrency` 单元有 5 个 trial。报告 trial 均值、样本标准差以及对
trial 进行 20,000 次有放回重采样得到的 percentile 95% bootstrap CI。4B/32B 吞吐比通过
分别重采样两组 trial 后取均值之比获得 95% CI。固定随机种子为 `20260721`。

不把“置信区间未重叠”当作显著性检验，也不以请求级样本量计算虚假的窄区间。结果以效应量
和不确定性为主。由于只有 5 个 block，区间应解释为本机短时重复性，而不是跨机器总体区间。

## 6. Trial 有效性与停止规则

正式采样开始后不根据观察到的模型输赢增加或删除条件。固定完成 5 个 block。一个 trial
只有在下列条件全部满足时有效：

- HTTP 成功率为 100%；
- 每个请求的服务端 usage 显示 prompt tokens 等于目标 ISL；
- 每个请求 completion tokens 恰为 128；
- trial 期间服务未重启、未 OOM、未发生 GPU Xid；
- 采集器记录到至少两个 GPU 遥测时间点。

无效 trial 原样保留并标注原因；仅允许在相同配置下整 trial 重跑一次。若同一条件连续两次
无效，则停止该模型的正式采样并报告阻塞，而不是静默调参。

## 7. 环境控制

- 两个模型顺序运行，不共享 GPU，不同时压测；
- 使用同一 NUMA 域内的 GPU 0–3；4B 使用其中 GPU 0；
- 每次启动前确认 GPU 无其他 compute process；
- 请求客户端与服务在同一主机，通过 `127.0.0.1` 通信；
- server stdout/stderr、命令、包版本、模型配置哈希、原始请求级数据和 GPU 遥测全部归档；
- 正式结果不引用历史 vLLM 日志中的滚动吞吐，因为那些日志没有冻结工作负载。

## 8. 预先声明的发布措辞

如果数据完整，结论应使用以下形式：

> 在 8×V100 节点中的固定资源配置上，Qwen3.5-4B（1×V100）与 Qwen3-32B
> （4×V100 TP）在 ISL/OSL = X/128、并发 C 下分别达到 A 与 B output tok/s；比值为
> R（trial-bootstrap 95% CI ...）。按 GPU 归一化后分别为 ...。该结果衡量记录的软件栈和
> 硬件配置，不是纯参数规模效应。

若固定长度校验失败、输出出现非有限 logits、服务不稳定或数据不足，则不得发布吞吐名次，
只能发布兼容性诊断。
