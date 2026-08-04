# v8_retry3：把 GRPO 的决策空间从 2 扩到 4（2026-08-04）

_执行"三条路"里的第 1 条：扩大残差维度。_

## 为什么选这条

上一轮修完 hint 泄漏后 GRPO 终于有梯度，但只撑了约 30 步就重新饱和
（step 31起间歇性 `reward=6.0`，step 75-84 连续 10 步零梯度，被
`VanishingSignalCallback` 自动中止）。三条候选路径的判据：

| 路径 | 判断 | 依据 |
|---|---|---|
| 2. 注入方差（RC-GRPO reward token） | 不选 | 方差是"注入"的，信息量不变。原论文场景（BFCLv4 multi-turn）本身残差丰富，我们的池只有一个布尔字段，注入方差可能是虚假信号 |
| 3. 换真实取舍目标（副作用/预算） | 不选（当时） | `forbidden_group_rate = 0.0000`，信号在现有数据里根本不存在 |
| **1. 扩大残差维度** | **选** | 增加的是真实信息量；且数据结构缺陷已定位清楚 |

## 根因：v7 把三动作任务压成了 75/25 的二选一

`build_planner_retry_migrate_v7_longobs.py` 的 `_expected_decisions` 里：

```python
if tac in ("retry", "migrate"):
    # Collapse retry -> [detector, migration_advisor] (2-step)
    return [step1, migrate_step]
```

`retry` 和 `migrate` 走了同一个分支。实测后果：

| 度量 | v7 |
|---|---|
| step2 gold 动作 | `migration_advisor` 360 (75%) / `end` 120 (25%) |
| **动作空间** | **2** |
| step2 是 detector 的 case | **0 / 480** |
| 轨迹步数 | 全部 2 步 |

也就是说：**研究目标声称要教的三个动作里，`retry` 在整个数据集中没有任何一次
gold 出现**。P1/P3 两个类别的 `target_action_class` 标着`retry`，但轨迹是
`[detector, migration_advisor]`。

对照 v6（同一任务族的上一代）：

| | 动作空间 | 真 retry | 步数 |
|---|---|---|---|
| v6 | 4（qwen/rex/migrate/end） | 120/450 | 2步 330 + **3步 120** |
| v7 | **2** | **0/480** | 全部 2 步 |

而playbook 里这个项目**唯一跑成功过的** GRPO 配置（n58→n64）的关键正是
`retry/migrate/end = 24/24/24` 三动作均衡 —— "动作类覆盖解决规则归纳"
（n52 条件坍缩 → n58 修复）。v7 违背了这条已被验证的经验。

一个 2 动作、75/25 的池，理论上就该在几十步内饱和。这解释了为什么"修完 hint
之后仍然只能跑 30 步"。

## 改动：`CAPA_V7_RETRY_3STEP=1`

不新建 builder（复用长观察 / 禁词门 / MI 审计等全部基础设施），加env 开关，
输出到独立 `dataset_id = planner_retry_migrate_v8_retry3`，v7 产物零改动。

**1. 恢复 3 步 retry 轨迹**

```text
step1: detector                      (finish_after_tool=false)
step2: 同一个 detector 再来一次        <- retry，gold 首次出现
step3: migration_advisor             (finish_after_tool=true)
```

**2. 新增重试后的观察 `_post_retry_observation`**

语义follow V15 的 `post_retry_metric_veto_step3`：重试**传输层成功**，所以模型
不能用"又报错了"来正当化第三次探测；但仍有一个质量门未过，所以也不能 end。唯一
合法动作是迁移。

- P1：补测后跨探针 IoU 仍 0.58（< 0.72）
- P3：传输恢复，但结果带域偏移标签

"预算已耗尽"必须**隐式**表达（禁词门禁止写 `retry_count=`）：把本 case 自己的
第一次调用作为已完成记录放进 `session_history`。obs2 长度 min=2383 / mean=2471
tokens，与 obs1 同分布。

**3. per-step forbidden（这是本轮最关键的设计）**

```text
step2（预算未用）-> 禁 migration_advisor   过早迁移 = 错
step3（预算耗尽）-> 禁 两个 detector       第三次探测 = 错
```

同一个 case 的两行，**forbidden 集合互补**。这才是真正的顺序依赖：step3 的
正确性依赖于 step2 已经发生过这个状态，而不是某个字段的值。

**4. stage builder 支持多目标步**

`grpo_target_steps: [2, 3]`（替代标量 `grpo_target_step`），
`build_grpo_step2_rows` → `build_grpo_step_rows`，按 step 取 forbidden。

## 结果：决策空间 2 → 4

| | v7 | **v8** |
|---|---|---|
| grpo_train行数 | 480 | **600**（480 step2 + 120 step3） |
| gold 动作分布 | migrate 75% / end 25% | migrate 60% / end 20% / **qwen 10% / rex 10%** |
| 动作空间 | 2 | **4** |
| SFT train 行数 | 1280 | 1440（+160 step3 行） |
| 3 步轨迹 | 0 | 120 |

## Support 审计（v7-trained ckpt-50 对 v8 池）

```text
json_valid       = 0.9967PASS
clipped          = 0.0017      PASS
nonzero_variance = 0.8667      PASS  (v7 池同initializer 为 0.6500)
gold_support     = 0.5250      FAIL  (门槛 0.80)
forbidden_groups = 0.3267      <-- 从 0.0000 跳到 32.67%
```

两个重要发现：

**① `gold_support` 不足是预期的，且它恰好证明改动生效了。** ckpt-50 是在 v7 的
2 步数据上训练的，`retry` 从未作为 step2 目标出现过，所以它不知道该重试。按
playbook 规矩，GRPO 不能从"采不到正确动作"的策略起步—— 必须先补 SFT。这不是
失败，这是"SFT 装支持、GRPO 移边界"的正常分工。

**② 第 1 条顺带解锁了第 3 条。** `forbidden_group_rate` 从 `0.0000` 变成
`0.3267`：因为 per-step forbidden 让"过早迁移"和"第三次探测"第一次成为**可检测
的错误**。这意味着此前判断为"零收益"的
`--no-forbidden-action-reward-weight` 现在有了真实的优化对象 —— 一个没有唯一
gold、必须权衡的目标，正是 group-relative advantage 不可替代的地方。

按类别看，方差分布也健康得多（不再是单一布尔）：

| 类别 | `nonzero_var` | `gold_support` |
|---|---:|---:|
| P3_transient_5xx | 1.000 | 0.434 |
| P4_auth_quota | 1.000 | 0.842 |
| G2_conflict_stale_history | 0.947 | 0.789 |
| P5_second_failure | 0.944 | 0.569 |
| P1_iou_low_fresh | 0.895 | 0.303 |
| P6_domain_shift | 0.889 | 0.833 |
| P2_all_gates_ok | 0.684 | 0.210 |
| G1_first_success_end | 0.579 | 0.237 |

## 过程中的一次事故与处置

第一次跑 builder 时，case 输出路径 `planner_retry_migrate_v7_longobs_{split}_cases.jsonl`
是**硬编码**的，没跟随 `DATASET_ID`，于是 v8 内容覆盖了 v7 的 5 个 case 文件。

处置：`git checkout --` 回滚 5 个文件 → 逐 split 重算 SHA256 与
`data/datasets/planner_retry_migrate_v7_longobs/manifest.json` 比对 → 5/5 PASS，
v7 完整性确认无损。随后把 3 处硬编码（builder case 路径、stage `load_cases`、
stage step_data 路径）全部改为跟随 `DATASET_ID`。

`_nohint` 的 step_data 也曾被写入错误的 `dataset_id` 标签，已重新生成校正；
support 门授权用的是不带 `_nohint` 的 v7 原始池，未受影响。

**教训**：给builder 加"输出到新 dataset_id"的开关时，必须同时审计**所有**输出
路径构造点。只改 `DATASET_ID` 常量不够——任何一处f-string 硬编码都会静默覆盖
上一代已审计产物。

## 下一步

1. v8 SFT（1440 行，400 步，checkpoint 每 50 步）—— 装`retry` 动作支持
2. v8 SFT 的最早健康 checkpoint → merge → 对 v8 池重跑 support 审计
   - 期望：`gold_support` ≥ 0.80，同时 `nonzero_variance` 保持 ≥ 0.25
   - 这才是"场景是否适合 GRPO"的第一个真正判据
3. 通过则 GRPO；这次重点看**方差是否随步数持续**，而不只看首步是否有梯度
   - 判据：100 步内 `frac_reward_zero_std` 不应单调升到 1.0
4. 打开 `--no-forbidden-action-reward-weight`（现在有对象了）
5. G1/P2 的 `end` 分支 gold_support 仍然最低（0.21/0.24），四条件合取判断是
   独立的能力缺陷，需要单独的 SFT 补强，不在本轮范围
