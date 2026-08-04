# v7 GRPO Support 审计：hint 泄漏是零方差的根因

_日期：2026-08-03 · 工具：`pipelines/eval/audit_grpo_support.py` · 采样：SFT ckpt-100 merged / vLLM / T=0.7 top_p=0.9 G=4 / 120 组 × 2 池 = 960 completion · 本轮 `optimizer_steps = 0`_

## 结论先行

`planner_retry_migrate_v7_longobs` 的 GRPO optimizer pool 在**数学上不可训练**，
根因不是 GRPO 超参，而是数据管线的一个未实现的开关：

`build_planner_retry_migrate_v7_longobs.py` 往 `observation.summary` 注入一句
显式路由提示（`决策提示：… 下一步请调用 migration_advisor …`），其docstring 声明
"SFT/GRPO training may strip it via `CAPA_STRIP_ROUTING_HINT=1` in
`prepare_v7_longobs_stage_data.py`"。**该开关在 `prepare_v7_longobs_stage_data.py`
中从未实现**，全局搜索只能命中那一行注释本身。因此 480/480 条 GRPO 行、
1280 条 SFT 行全部带hint 出厂。

`h20_experience.md` 第 5.3 节记录的"训练时用环境变量 mask 掉 hint 保持学习价值"
这一设计意图，实际上从未生效。

## 1. 同一 initializer，两个池，对照实验

剥离 hint 是**唯一变量**：`case_id`、`expected_step`、`reward_spec`、
`forbidden_actions`、类别分布逐字节一致；prompt token 均值 6513 → 6478（−34）；
`prompt_sha256` 重合数 = 0。

| 池 | `json_valid` | `clipped` | `gold_support` | `nonzero_variance` | `reward_mean` |
|---|---:|---:|---:|---:|---:|
| **hint**（现役产物） | 1.0000 | 0.0000 | **1.0000** | **0.0000** | **1.0000** |
| **nohint**（本轮新建） | 1.0000 | 0.0000 | 0.7479 | **0.2333** | 0.7979 |

hint 池的 `nonzero_variance = 0.0000` 是**全部 120 组、8/8 类别**的结果，不是均值
掩盖：每一组的 4 个 completion reward 完全相同且均为满分。

`A_i = (r_i − μ_g)/σ_g` 在 `σ_g = 0` 时无定义/为零，因此 hint 池上的任何
GRPO 更新都必然为零。这与 08-03 seed43 训练遥测精确吻合：

```text
frac_reward_zero_std = 0.994 (mean over 22 steps)
grad_norm = 0.0 on 21 / 22 steps  (仅 step 12 = 0.1266)
reward = 6.0 (满分)   reward_std = 0.0   advantage/std = 0.0
```

审计工具在 16 组小样本上即独立复现了 `frac_reward_zero_std = 1.0`，说明这不是
训练框架缺陷，而是池的固有性质。

## 2. 为什么 hint 泄漏会同时制造"高分"和"零信号"

hint 只点名**动作**，从不提**参数**。于是模型的行为被劈成两半：

- hint 覆盖的部分（选哪个工具）→ 背下来→ 满分 → 零方差
- hint 未覆盖的部分（`finish_after_tool`）→ 没学会 → 成为全部残差

这解释了同日残差诊断的发现（`V7_GRPO_HEADROOM_DIAGNOSIS_20260803.md`）：
grpo_dev 上 42/43 个 P1+P6 失败是同一个签名—— 动作、`use_image`、
`use_visual_probe`、`user_query` 全对，**只有 `finish_after_tool` 输出 false**。

因此"SFT softbnd mean_score = 0.9813"这个数字衡量的主要是**照抄 hint 的能力**，
不是从 `detector_response` / `session_history` / `technical_notes` 推理的能力。

## 3. 剥离 hint 后的类别分层：真实能力地图

| 类别 | `nonzero_variance` | `gold_support` | 判定 |
|---|---:|---:|---|
| P1_iou_low_fresh | 0.533 | 0.700 | **TRAINABLE** |
| P6_domain_shift | 0.267 | 0.867 | **TRAINABLE** |
| G1_first_success_end | 0.733 | **0.300** | GOLD_BROKEN |
| P2_all_gates_ok | 0.333 | **0.117** | GOLD_BROKEN |
| G2_conflict_stale_history | 0.000 | 1.000 | SATURATED |
| P3_transient_5xx | 0.000 | 1.000 | SATURATED |
| P4_auth_quota | 0.000 | 1.000 | SATURATED |
| P5_second_failure | 0.000 | 1.000 | SATURATED |

三类各有不同含义：

- **SATURATED（4 类，60/120 组）**：错误路径（5xx / 认证配额 / 二次失败 / 冲突历史）。
  剥离 hint 后依然满分，说明模型**真的**从 `error.class_label` 学到了规则。
  这是 v7 数据设计成功的部分，但对 GRPO 零贡献。
- **TRAINABLE（2 类，30 组）**：成功回执但指标不合格（IoU 偏低 / 域偏移）。
  同时有 gold support 和组内方差 —— 这是 GRPO 唯一有效的作用域。
- **GOLD_BROKEN（2 类，30 组）**：`end` 分支。gold support 崩到 0.300 / 0.117，
  即模型在无 hint 时**基本采不到正确动作**。这是 hint 泄漏最严重的地方：
  SFT 学的是"读到『请直接输出 end』就输出 end"。

按 playbook 的V7 教训（"Reward 有方差但 gold support 低 → 变化主要是不同错误
→ 小规模 SFT 补支持，不放宽门槛"），GOLD_BROKEN 类别不能靠 GRPO 修，必须先补 SFT。

若只用 TRAINABLE 两类构成 optimizer pool：

```text
groups = 30    nonzero_variance = 0.4000    gold_support = 0.7833
```

方差门（≥0.25）通过，gold support 门（≥0.80）仍差 0.017。

## 4. 门禁判定：本轮不授权任何 optimizer step

| 门 | 阈值 | hint 池 | nohint 池 | nohint 仅 TRAINABLE |
|---|---|---|---|---|
| `json_valid` | ≥0.99 | ✅ 1.000 | ✅ 1.000 | ✅ |
| `clipped` | ≤0.01 | ✅ 0.000 | ✅ 0.000 | ✅ |
| `gold_support` | ≥0.80 | ✅ 1.000 | ❌ 0.748 | ❌ 0.783 |
| `nonzero_variance` | ≥0.25 | ❌ 0.000 | ❌ 0.233 | ✅ 0.400 |
| **总判定** | | **FAIL** | **FAIL** | **FAIL** |

三种配置全部未过门，因此 `optimizer_steps = 0`。

**不允许的补救**（playbook 明文）：把 `gold_support` 门从 0.80 降到 0.75、
把方差门从 0.25 降到 0.23、或重采到通过为止。

## 5. 因果链复盘

```text
hint 未被strip（开关只存在于注释）
  → SFT 在含 hint 数据上训练，学会照抄 hint 指定的动作
  → hint 覆盖动作但不覆盖参数
     ├─ 动作维度：满分 →组内零方差 → GRPO梯度恒为 0（seed42/43 空转）
     └─ 参数维度：finish_after_tool 未学会 → 占据 dev 残差的 93%
  → mean_score=0.9813 看似接近上限，实际主要衡量"照抄能力"
  → 以 mean_score 为主指标 → 只剩 1.87% headroom 且恰在饱和区
  → 三份 seed43 并发空转 2.5 小时，grad_norm 21/22 步为 0
```

## 6. 下一步被允许做什么

唯一科学正确的路径是**先修initializer，再谈 GRPO**：

1. **用 no-hint 数据重跑 SFT**。当前 ckpt-100 是在泄漏数据上训练的，
   其能力声明（`4B SFT ≥ 35B base`）的有效性受污染，必须重建。
   预期效果有两个方向且都有利：
   - GOLD_BROKEN 的 `end` 分支重新获得真实 support（不再依赖 hint 字面量）
   - 任务变难 → SATURATED 类别可能重新出现方差 → optimizer pool 扩大
2. **重跑 support 审计**，只有 `gold_support ≥ 0.80` 且 `nonzero_variance ≥ 0.25`
   同时成立才授权 GRPO。
3. **reward 权重重配**（已量化，待support 通过后生效）：
   当前 `action_match` 占 56.5% 权重却只贡献 2% 残差；`finish_after_tool` +
   `final_tool_finish` 占 13% 权重却贡献 93% 残差。信号密度与残差严重错配。
4. **保留 hint 版本作为评测基线**。35B base ≥ 0.85 的数据集健康门是在含 hint
   条件下测得的，评测协议不应单方面改变；新增的 no-hint 评测必须作为独立
   arm 报告，不能与旧数字混用。

## 7. 复查入口

```bash
# support审计（本报告数据来源）
python3 pipelines/eval/audit_grpo_support.py \
  --pool nohint=training/planner_grpo_seed_v1/step_data/planner_retry_migrate_v7_longobs_grpo_train_qwen35_4b_nothinking_step2_nohint.jsonl \
  --pool hint=training/planner_grpo_seed_v1/step_data/planner_retry_migrate_v7_longobs_grpo_train_qwen35_4b_nothinking_step2.jsonl \
  --model sft_ckpt100 --prompts 120 --generations 4 --seed 42 \
  --out reports/grpo_support_audit_20260803.json

# no-hint 数据重建（幂等；写入带 _nohint 后缀的独立路径，不覆盖既有产物）
CAPA_STRIP_ROUTING_HINT=1 \
CAPA_QWEN35_TOKENIZER_DIR=<4B 权重目录> \
  .venv-h20-infer/bin/python \
  training/planner_grpo_seed_v1/scripts/prepare_v7_longobs_stage_data.py
```

产物：

- `reports/grpo_support_audit_20260803.json`（960 completion 的逐组 reward）
- `training/planner_grpo_seed_v1/step_data/*_step2_nohint.jsonl` + manifest
- `training/planner_grpo_seed_v1/sft_data_..._nothinking_nohint/{train,dev}.jsonl`
- 空转证据：`repro_h20/grpo/20260803_2110*/telemetry/`、
  `logs/train/resume_grpo_20260803_2110*.log`
