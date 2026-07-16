# Qwen3.5-4B 在 planner_retry_migrate_v6 上的训练与真实评测

日期：2026-07-16
最终状态：SFT 已完成；GRPO 因奖励支持门控失败而未启动；sealed test 已完成。

## 结论

这次 SFT 是有效的，但模型还没有真正学稳“显式结构化状态优先于词面干扰”的策略。checkpoint-100 在实体隔离 SFT-dev 上将动作准确率从 base 的 77.31% 提高到 97.69%，但在冻结后才解封的 60 个全新实体 test 上，780 个去重决策的动作准确率为 94.49%，450 条完整轨迹全对率只有 90.89%。

当前 V6 的 GRPO split 不应该直接训练。checkpoint-100 只有 2/180 个 prompt 组产生非零奖励方差，备选 checkpoint-75 也只有 5/180；绝大部分 batch 会得到零优势，而且这个 core-only step-2 split 不包含 test 暴露出的全部残差边界。按门控协议，本次 GRPO optimizer steps 为 0。

## 数据与隔离

- SFT：train 1,040 行，dev 260 行。
- GRPO：train 360 行，dev 180 行；当前只覆盖 core step 2。
- sealed test：450 个 case、60 个新实体，展开为 780 个唯一 prompt；按原始 case 展开后为 1,020 个决策。
- test 与四个 train/dev split 的 `case_id`、`entity_id`、`project_entity`、`target_entity`、`counterfactual_bundle_id` 和格式化 prompt hash 均为零交叉。
- test 在 checkpoint 选择和两个 GRPO 门控结论冻结后才格式化解封；没有再用 test 比较其他 checkpoint。
- 独立人工复核仍为 pending，因此所有结论是规则 oracle 下的合成数据结论，不是生产验收。

## SFT 结果

训练使用 Qwen3.5-4B 的 LoRA，训练参数 14,376,960，100 个 optimizer step，耗时 4,731.7 秒。训练过程没有非有限日志值或非有限权重。

| 模型 | SFT-dev 动作准确率 | 平均规则奖励 | JSON 有效率 |
|---|---:|---:|---:|
| base | 77.31% | 80.68% | 100% |
| checkpoint-75 | 95.77% | 96.36% | 100% |
| checkpoint-100 | **97.69%** | **98.04%** | 100% |

checkpoint-100 按预先冻结的主指标选中，相对 base 提升 20.38 个百分点，动作错误相对减少 89.83%。最终 eval loss 为 0.2191，token accuracy 为 97.41%。训练 loss 已从 1.4070 降到 0.00151；继续无条件延长当前 SFT 更可能增加记忆和捷径，而不是解决组合泛化。

## 为什么没有硬跑 GRPO

| initializer | prompt 组 | gold action support | 非零奖励方差组 | 方差率 | 结论 |
|---|---:|---:|---:|---:|---|
| checkpoint-100 | 180 | 100.00% | 2 | 1.11% | fail |
| checkpoint-75 | 180 | 96.67% | 5 | 2.78% | fail |

checkpoint-100 的 720 个采样动作准确率已达 99.58%，core step-2 基本饱和。checkpoint-75 的完整审计中，Qwen/Rex 只有 3/2 个方差组，retry 为 0/60；此前 14-prompt 小面板显示的 14.29% 方差率在全量 180-prompt 审计中降到 2.78%，说明小面板明显高估了可学习支持。

因此，启动 GRPO 只会满足“流程上跑过”，不会针对剩余错误产生稳定的策略梯度。本次将它记录为门控失败，而不是训练失败。

## Sealed test

| 口径 | 数量 | 动作准确率/全轨迹通过率 | 平均规则奖励 |
|---|---:|---:|---:|
| 去重决策 | 780 | 94.49% | 95.44% |
| 展开到原始 case 的决策 | 1,020 | 95.78% | 96.52% |
| 完整轨迹 | 450 | **90.89%** | 90.89% 全规则通过 |

JSON 有效率为 100%，JSON 后额外文本为 0%，说明失败不是格式或截断问题，而是策略路由错误。按展开决策计，step 1 为 450/450，step 2 为 410/450，step 3 为 117/120；主要瓶颈是 step 2。

完整轨迹分层如下：budget-exhausted 120/120，nonretryable 120/120，fresh-retry 84/120；conflicting-state 13/15，initial-success 12/15，其余四类 guardrail 全对。43 个错误决策中，35 个是“应继续 Qwen/Rex detector，却过早调用 migration_advisor”；另外有 5 个应 end 却 migrate、2 个应 migrate 却调用 Qwen detector、1 个应 migrate 却 end。

Qwen 与 Rex 的完整轨迹率分别为 90.67% 和 91.11%，detector family 不是主因。更明显的是 nuisance shortcut：带“请对最新故障回执作最终判定”的 policy-4 模板在两个 detector 上的决策准确率只有 84.76%/85.71%，其他 policy 模板为 96.19%–100%。红色 badge 的完整轨迹率为 88.00%，低于 amber 的 93.33% 和 missing 的 91.33%。不同 test error alias 的完整轨迹率为 80%–100%。这些字段与动作标签的边际互信息虽然被构造为接近零，但模型仍在交互组合上使用了捷径。

test 的两种 fixture family 轨迹率为 96.00% 与 85.78%，但当前构造用 `entity_index % 2` 分 fixture、`entity_index % 4` 分 query style，二者没有完全交叉，不能把差异解释为视觉 fixture 因果效应。这个混杂本身说明下一版评测设计还需要改进。

## 真正正确的下一版数据

1. SFT、GRPO-support dev、checkpoint-selection dev、sealed test 继续保持实体与内容隔离；sealed test 只在所有选择冻结后解封。
2. 将 query style、fixture/目标、badge、error alias family、detector、retry outcome 做完全或平衡部分因子交叉，不能再用同一个 entity index 的取模同时分配多个 nuisance。
3. 主评测指标改为完整轨迹通过率；step-level 动作、参数、finish 和 JSON 只作为诊断指标。每个关键 stratum 都单独报告，不只报告总体均值。
4. 下一轮 SFT 数据应增加 held-out hard wording，尤其是包含“故障、预算、耗尽、失联”等容易诱发 migrate 的表述，并在完全相同词面下配对 fresh-retry、nonretryable、budget-exhausted 三种结构化状态。
5. 下一轮 GRPO 数据必须在 SFT 后按残差挖掘，而不是事先固定 core step 2。优先覆盖 fresh-retry step 2、重试后 success/error/metric-veto 的 step 3、conflicting-state 和 current-success；全部使用新实体。
6. 在任何 optimizer step 前，重新做每 prompt 多采样支持审计。要求 gold action 在每个 detector/target stratum 都有支持，同时至少 10% prompt 组有非零规则奖励方差，并设置每个关键 stratum 的最小方差组数；更理想的学习组应让正确动作采样率落在约 20%–80%，避免全错或全对。

## 证据

- 数据清单：`data/datasets/planner_retry_migrate_v6/manifest.json`
- SFT 结果：`experiments/runs/20260716_qwen35_4b_planner_v6_sft_seed42_v1/capa_qwen35_planner_v6_sft_result.json`
- checkpoint 选择：`experiments/studies/planner_retry_migrate_v6_qwen35_4b_v1/sft_selection.json`
- GRPO 门控：`grpo_support_checkpoint100_decision.json`、`grpo_support_checkpoint75_decision.json`
- sealed-test 汇总：`experiments/runs/20260716_qwen35_4b_planner_v6_sft_ckpt100_sealedtest_eval/report.json`
- 完整轨迹与分层分析：`experiments/runs/20260716_qwen35_4b_planner_v6_sft_ckpt100_sealedtest_eval/trajectory_analysis.json`
- 机器可读总结果：`final_result.json`
