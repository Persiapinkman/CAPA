# Planner retry/migrate residual V7 支持度门禁结果

日期：2026-07-16
最终状态：**support gate failed；GRPO optimizer steps = 0；sealed test 未物化。**

## 结论

V7 成功解决了 V6 的“奖励几乎无方差”问题，但没有同时满足预注册的金标准动作支持门槛。checkpoint-100 在 216 个实体隔离开发集 prompt 上各采样 4 次，共得到 864 个样本；全部 JSON 合法、无截断、无非有限 reward，216 个 prompt 组全部完整。

主残差场景有 35/144（24.31%）个非零 reward 方差组，明显高于 10% 门槛；fresh-retry 在 Qwen/Rex 两侧分别有 4/2 个非零方差组，金标准支持率为 83.33%/75.00%，也都通过各自门槛。唯一失败的硬检查是主残差金标准动作支持率：110/144（76.39%），低于预注册的 80%。因此 `optimizer_authorized=false`，五步 canary 没有启动，W&B 也没有创建一条伪装成训练的 V7 run。

## 正式硬门槛

| 检查 | 观察值 | 门槛 | 结果 |
|---|---:|---:|---|
| samples | 864 | 864 | pass |
| 完整 prompt 组 | 216 | 216 | pass |
| JSON 有效率 | 100.00% | ≥99.00% | pass |
| 截断率 | 0.00% | ≤1.00% | pass |
| 主残差金标准动作支持率 | **76.39%** | **≥80.00%** | **fail** |
| 主残差非零 reward 方差率 | 24.31% | ≥10.00% | pass |
| fresh-retry Qwen 非零方差组 | 4 | ≥2 | pass |
| fresh-retry Qwen 金标准支持率 | 83.33% | ≥50.00% | pass |
| fresh-retry Rex 非零方差组 | 2 | ≥2 | pass |
| fresh-retry Rex 金标准支持率 | 75.00% | ≥50.00% | pass |

## 场景诊断

| 场景 | 金标准支持率 | 非零方差组 | 方差率 | 自适应场景规则 |
|---|---:|---:|---:|---|
| fresh_retry_step2 | 79.17% | 6/24 | 25.00% | pass |
| post_retry_error_step3 | 87.50% | 7/24 | 29.17% | pass |
| current_success_step2 | 66.67% | 8/24 | 33.33% | pass |
| conflicting_state_step2 | 62.50% | 7/24 | 29.17% | pass |
| post_retry_success_step3 | 79.17% | 5/24 | 20.83% | fail：Rex 仅 1 个方差组 |
| post_retry_metric_veto_step3 | 83.33% | 2/24 | 8.33% | fail：Qwen 0 个方差组且总体低于 10% |

这里的“自适应场景规则 pass”不等于获准训练。预注册协议要求先通过所有全局硬门槛；由于主残差支持率失败，最终 accepted-scenario 文件为空，不能选择其中四类绕过全局失败。

三个稳定性 control 合计金标准支持率为 70.83%，非零方差率为 33.33%。这进一步说明模型在 V7 的新实体、别名和结构化状态组合上确实存在可学习残差，而不是格式失败；但正确动作尚未在足够多的采样组中出现，直接做 GRPO 风险过高。

## 数据与隔离

- train/dev 分别为 24/12 个全新实体，每个实体与 Qwen、Rex 及 9 类反事实场景成束配对；独立单位是实体。
- query style、badge、fixture、detector、error alias 分别平衡，避免 V6 中由同一 index 取模引入的 nuisance 混杂。
- train 为 432 行，dev 为 216 行；step 2/3 混合监督已通过冻结 manifest 和哈希校验。
- test 承诺为 216 行，SHA-256 为 `8b2dcfe9c66af3936592e37aafc6b52e8820da009110f6f58f4e1ae716cfc820`，当前 `materialized=false`。本轮没有查看 test。
- 独立人工复核仍为 pending，因此该结论只适用于规则 oracle 下的合成实验。

## 下一版最合理的设计

不要在 V7 上重采样、放宽 80% 阈值或事后挑选四个场景训练。应把 V7 作为 pilot，预注册实体完全隔离的 V8：

1. 将 `fresh_retry_step2` 与 `post_retry_error_step3` 作为第一阶段核心 residual arm；二者已经同时表现出正确动作支持与足够方差。
2. 对 `current_success_step2` 和 `conflicting_state_step2` 先做一轮小规模、全新实体的 residual SFT augmentation，或设为独立实验臂；它们当前的 66.67%/62.50% 支持率是本次全局失败的主要来源。
3. `post_retry_success_step3` 与 `post_retry_metric_veto_step3` 暂不进入同一 GRPO optimizer dataset：前者 Rex 方差不足，后者 Qwen 基本饱和。
4. V8 使用新的 train/dev/test 实体与文本，不复用 V7-dev 做确认；每个场景×detector 单独设最低支持和方差组数，再设全局完整性门槛。
5. 只有新门禁通过才创建 W&B 训练 run；看板继续强制记录 Train Reward Statistics、Policy Entropy、Gradient Norm 与 Mean Advantage Estimate。

## 证据

- 预注册：`preregistration.json`
- 正式门禁：`support_decision.json`
- 合并样本：`experiments/runs/20260716_qwen35_4b_residual_v7_grpo_dev_support4x_ckpt100/samples.jsonl`
- 合并摘要：`experiments/runs/20260716_qwen35_4b_residual_v7_grpo_dev_support4x_ckpt100/summary.json`
- 数据清单：`data/datasets/planner_retry_migrate_residual_v7/manifest.json`
- 机器可读结果：`final_result.json`
