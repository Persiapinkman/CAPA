# v7 GRPO：从零信号到首次通过 support 门（2026-08-03/04）

_范围：定位并修复"GRPO 无法训练"的根因，重建 initializer，机械通过 support 硬门，启动第一次有资格执行的 GRPO。_

## 一句话结论

**这个项目此前跑的 GRPO 全部是数学意义上的零更新空转**，根因是一个从未实现的
数据开关；修复后`nonzero_variance` 从 `0.0000` 升到 `0.6500`，
`grad_norm` 从恒等于 0 变成 0.054，support 硬门首次通过。

## 1. 起点：三份并发空转

2026-08-03 21:10 有**三份同一 seed43 的 GRPO 并发**跑在同 4 卡上
（`20260803_2110{47,05,24}`）。遥测（step 22/100）：

```text
reward = 6.0 (满分)         reward_std = 0.0
frac_reward_zero_std = 0.994advantage/std = 0.0
grad_norm = 0.0  于 21/22 步（仅 step 12 = 0.1266）
route_exact = argument_exact = stop_exact = no_forbidden = 1.0
entropy ≈ 0.006
```

因为 `A_i = (r_i − μ_g)/σ_g` 在 `σ_g = 0` 时无梯度，这些运行不可能学到任何东西。
已于 21:58 终止，遥测与日志保留为证据。

## 2. 根因：`CAPA_STRIP_ROUTING_HINT` 只存在于注释里

`build_planner_retry_migrate_v7_longobs.py` 往 `observation.summary` 注入一句
显式路由提示（`决策提示：… 下一步请调用 migration_advisor …`），用于让 base 模型
零样本通过"35B ≥ 0.85"的数据集健康门。其 docstring 声明训练侧可用
`CAPA_STRIP_ROUTING_HINT=1` 剥离——**但该开关在 `prepare_v7_longobs_stage_data.py`
中从未实现**，全局搜索只命中那行注释本身。

于是 480/480 条 GRPO 行、1280条 SFT 行全部带 hint 出厂，`h20_experience.md`
第 5.3 节记录的"训练时 mask 掉 hint 保持学习价值"从未生效。

hint 只点名**动作**，从不提**参数**，因此模型行为被劈成两半：

- hint 覆盖的部分（选哪个工具）→ 背下来 → 满分 → 零方差 → GRPO 无梯度
- hint 未覆盖的部分（`finish_after_tool`）→ 没学会 → 占dev 残差的 93%

饱和度随 SFT 变强单调恶化，恰好解释了为什么"越训越训不动"：

| run | initializer | `frac_reward_zero_std` | `task_reward` |
|---|---|---:|---:|
| `20260801_grpo_v7_seed42_r3` | 老 SFT（fp16 栈） | 0.789 | 0.627 |
| `20260802_224145_seed42` | hint-SFT ckpt-100 | 0.864 | 0.979 |
| `20260803_2110*_seed43` | 同上 | **0.994** | ~1.000 |

## 3.顺带修掉的verifier bug

`reward_planner_grpo.py` 的 `arg_contains` 曾用ALL 语义（同义词集变成逻辑与陷阱），
提交 `432c186` 已修，但 08-02 的三场景评测跑在修复之前且未重跑。原始
`*_predictions.jsonl` 都在盘上，用修复后的 verifier 零推理成本重打分：

| Arm | mean_score 旧 → 新 | pass_all_runs 旧 → 新 |
|---|---|---|
| 4B base | 0.7634 → 0.7714 | 0.0000 → 0.0000 |
| 35B base | 0.8525 → **0.8614** | 0.0000 → **0.1833** |
| 4B SFT ckpt-100 | 0.9704 → **0.9813** | 0.5125 → **0.7625** |

三个研究不等式与 35B ≥ 0.85 健康门仍成立。但**`pass_headroom` 是 `mean_headroom`
的 12.7 倍**（0.2375 vs 0.0187）：以 `mean_score` 为 GRPO 主指标时只剩 1.87% 空间，
且恰在饱和区。主指标应为 `pass_all_runs`。

## 4. 2×2 support 对照矩阵

剥离 hint 是唯一变量（`case_id` / `expected_step` / `reward_spec` /
`forbidden_actions` 逐字节一致，prompt token 均值 6513→6478，`prompt_sha256`
重合 0）。用 trainer 的解码设置（T=0.7, top_p=0.9, G=4）各采 120 组：

| initializer | pool | `gold_support` | `nonzero_variance` | 门 |
|---|---|---:|---:|---|
| hint-SFT ckpt-100 | hint | 1.0000 | **0.0000** | FAIL |
| hint-SFT ckpt-100 | nohint | 0.7479 | 0.2333 | FAIL |
| noHint-SFT ckpt-50 | nohint | 0.6208 | **0.8667** | FAIL |
| noHint-SFT ckpt-100 | nohint | 0.7354 | 0.2750 | FAIL |
| **noHint-SFT ckpt-50** | **hint** | **0.9729** | **0.6500** | ✅ **PASS** |

一个反直觉但重要的发现：**hint 泄漏并没有把 SFT 变成纯粹的"读 hint 机器"**。
hint-SFT 在剥离 hint 后 gold support仍有 0.7479，与 noHint-SFT ckpt-100 的
0.7354 基本相当。hint 的真正危害是**锁死策略熵**：

| initializer | `eval_loss` | `eval_entropy` |
|---|---:|---:|
| hint-SFT ckpt-100 | 1.8e-4 | ≈0.006 |
| noHint-SFT ckpt-50 | 0.467 | **1.049** |
| noHint-SFT ckpt-100 | 0.023 | 0.869 |

熵差 166 倍。GRPO 需要的不是"更高的分数"，是"还没被压平的分布"。

## 5. 为什么选 noHint-ckpt-50 + hint 池

这是唯一通过门的配置，且它在科学上自洽：

- initializer 在 hint-stripped prompts 上训练 → 不依赖 hint 字面量，保留熵
- optimizer pool 保留 hint → 与 dev/test 评测条件一致；同时改变 initializer
  和评测协议会让比较无法解释
- 残差集中在 `finish_after_tool`（156/190 component failures）—— 一个单布尔
  决策边界，正是"SFT 装支持、GRPO 移边界"的适用对象

no-hint 池未过门的原因也定位清楚了：G1_first_success_end（gold 0.267）与
P2_all_gates_ok（gold 0.150）两个 `end` 分支类别在无 hint 时gold support 崩溃，
其余 6 类均 ≥ 0.717。四条件合取判断（候选唯一 ∧ 置信度∧ IoU ∧ 无域偏移 → end）
是真实能力缺陷，按playbook 规矩只能补 SFT，不能靠 GRPO 修，也不能降门槛。

## 6. 首次健康的 GRPO 首步

`20260804_*_qwen35_4b_v6_grpo_seed42`，step 1：

| 指标 | seed43（饱和池） | 本轮 |
|---|---:|---:|
| `grad_norm` | 0.0000 | **0.0540** |
| `reward_std` | 0.0000 | **0.3844** |
| `frac_reward_zero_std` | 0.994 | **0.500** |
| `advantage/std` | 0.0000 | **0.4326** |
| `entropy` | 0.006 | **0.244** |
| `completions/clipped_ratio` | 0.0 | 0.0 |
| `reward` | 6.000（满分饱和） | 5.516 |

## 7. 交付物

**新增工具**

| 文件 | 作用 |
|---|---|
| `pipelines/eval/diagnose_grpo_headroom.py` | 主指标从 `mean_score` 切到 `pass_all_runs`，逐失败签名归因 |
| `pipelines/eval/audit_grpo_support.py` | trainer-faithful support 审计，4 条硬门，支持多池对照 |

**修改**

| 文件 | 改动 |
|---|---|
| `prepare_v7_longobs_stage_data.py` | 实现 `CAPA_STRIP_ROUTING_HINT`；变体写`_nohint` 后缀独立路径；manifest 记录 strip 统计 |
| `train_qwen35_4b_grpo.py` | 新增 `VanishingSignalCallback`：连续 N 步 `frac_reward_zero_std≥0.99` 且 `grad_norm==0` 即中止并落 `vanishing_signal_abort.json` |
| `run_h20_repro.sh` | 新增 `phase_support`；`phase_grpo`硬依赖 `status/support.done`；`all-train` 串入 support |
| `h20_experience.md` | 新增第 0 节记录根因；第 7 节标注旧数字的适用条件 |

`VanishingSignalCallback` 用 08-03 真实日志回放验证：在第 22 步中止（原本会白跑
100 步 × 3 份）；健康 run（`zero_std=0.75, grad>0`）不误杀。

**报告**

- `reports/V7_GRPO_HEADROOM_DIAGNOSIS_20260803.md` — 残差定位
- `reports/V7_GRPO_SUPPORT_AUDIT_20260803.md` — hint 泄漏根因与类别分层
- `reports/grpo_headroom_{2026-08-03,rescored_2026-08-03}.json`
- `reports/grpo_support_audit_{20260803,nohint_sft50_20260804,nohint_sft100_20260804}.json`

**新产物**

- `sft/20260803_230824_qwen35_4b_planner_v6_sft/` — no-hint SFT，400 步，
  checkpoint-50…400 全存，`checkpoint-{50,100}_merged`
- `training/.../sft_data_..._nohint/{train,dev}.jsonl` + `step_data/*_step2_nohint.jsonl`
- `repro_h20/support/support_20260804_005034.json` + `status/support.done`

## 8. 下一步

1. GRPO seed42 跑完 → `grpo-eval` 三场景 3× → 用
   `diagnose_grpo_headroom.py` 以 `pass_all_runs` 为主指标比较
2. 若 seed42 有正向且副作用不增 → seed43/44（support门已授权同一池）
3. reward 权重重配：待 GRPO 后的新残差分布确定再做。当前不动的理由是
   残差分布在两个池之间差异很大（hint 池 93% 在 `finish_after_tool`，
   no-hint 池均匀分布在 action/decision_type/argument），此时改权重是过早优化
4. `--no-forbidden-action-reward-weight` 暂不打开：当前
   `forbidden_group_rate = 0.0000`，打开只会稀释信号
5. G1/P2 的 `end` 分支能力缺陷需要独立的 SFT 补强实验，不在本轮范围
