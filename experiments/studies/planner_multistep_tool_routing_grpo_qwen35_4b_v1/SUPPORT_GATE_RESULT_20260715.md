# Qwen3.5-4B V5 train-v1 GRPO support gate 结果

日期：2026-07-15
结论：`FAIL / OPTIMIZER_NOT_STARTED`

## 审计对象

- base policy：`/raid/zkq/models/Qwen3.5-4B`
- 训练池：`planner_multistep_grpo_value_v5_train_v1`
- 固定 support pool：80 cases，8 个 scenario × 10，migrate/retry = 48/32
- 采样：每题 8 次，`temperature=0.7`，`top_p=0.9`，`seed=42`
- 生成上限：320 tokens
- 总计：80 groups / 640 samples

support pool SHA256：
`f24debec5ecc8b428afe3d8583626b01435c42af2e3f9313ee949650dcb957dc`

gate spec SHA256：
`9154f71b8937ab41c05d626f841c5cbccf4e1e06a135fdcbdbcf3cff474c98d0`

## 冻结门禁结果

| 检查项 | 观测值 | 门槛 | 结果 |
|---|---:|---:|---|
| migrate nonzero reward std rate | 0.8750 | >= 0.80 | PASS |
| migrate usable support rate | 0.8750 | >= 0.80 | PASS |
| migrate exact-action support rate | 1.0000 | >= 0.80 | PASS |
| migrate mean distinct valid actions | 1.0000 | >= 1.40 | **FAIL** |
| migrate fully saturated rate | 0.6875 | <= 0.25 | **FAIL** |
| retry exact-action support rate | 0.5625 | >= 0.95 | **FAIL** |
| completion clipped rate | 0.0390625 | <= 0.01 | **FAIL** |

总 gate 为 `false`。根据预注册方案，未进入 G3/G4/G5，optimizer steps = 0。

## 解读

48 个 migrate group 都至少采到一次正确 `migration_advisor`，但所有组的有效 action
多样性都只有 1，且 33/48 组八次采样全部高分。其 reward std 主要来自格式尾部差异或
截断，而不是路由边界；将它当作主学习集会优化输出长度，不能清晰验证 GRPO 的
路由价值。

32 个 retry group 只有 18 个至少出现一次正确 detector route。256 个 retry 采样中，
`migration_advisor` 占 189 个，`qwen_detection` 52 个，`rexomni_detection` 12 个。这说明
retry 才是当前 base policy 的真正难点，与原先“retry 是 anchor、migrate 是 target”的前提相反。

25/640 个 completion 没有在 320 tokens 内生成 EOS；其中 migrate 为 23/384
（5.99%），retry 为 2/256（0.78%）。因此单题 320-token 探针不能代表整个训练分布。

## 建议的下一版

1. 不用当前 480 条原样本直接开 GRPO，也不在观察结果后放松本 gate。
2. 把 retry 重新定义为主学习边界：构造能让 4B 在 8 次采样中同时出现 detector 与
   migration 的样本，目标每组正确路由采样率约 25%–75%。
3. 将已饱和 migrate 下调为 10%–20% 保持集；新增真正会在 detector/migrate 间摆动的
   migrate hard negatives，而不是仅修改 query 表达。
4. 对 14/32 完全采不到正确 retry 的难度桶，先用完全 disjoint 数据做最短 SFT
   warm-up；对照必须是同一 initializer 不训练与加 GRPO 两个 arm。
5. 新数据冻结后另取不重复 support pool，重做 8 次采样门禁；长度先用 384/512
   无训练探针测出全分布 p99/max，再单独重跑 G2 显存门。

## 机器可读证据

- `support320/gate.json`
- `support320/combined/summary.json`
- `support320/combined/groups.jsonl`
- `support320/combined/samples.jsonl`

combined groups SHA256：
`012009edecb54fd20f119baafef47225097007a64e38f8459b31862d9a530231`

combined samples SHA256：
`3cda50a0150900892224cf46da6dca6ba5d3cd04a8df5cc5d6eadaa43316d0c5`

gate result SHA256：
`f1674d85c1c7456cb10d81988ab912119ac0bb8cd1725c108c79278737c1faa1`
