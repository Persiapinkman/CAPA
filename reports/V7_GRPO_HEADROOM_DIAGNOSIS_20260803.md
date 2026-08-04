# v7 GRPO 残差定位报告（第 1 步：改指标，不改算法）

_日期：2026-08-03· 范围：零推理成本重打分 + 残差归因 · 不含训练_

##摘要

用现有 3× 评测的**原始预测文件**重新打分并把主指标从 `mean_score` 切到
`pass_all_runs`，得到三个结论：

1. 之前的 reward 报告受一个已修复但未重跑的 verifier bug 影响，低估了全部三个 arm。
2. 修正后 SFT 的真实残差是 `pass_all_runs = 0.7625`，而 `mean_score = 0.9813`；
   **pass headroom 是 mean headroom 的 12.7 倍**。
3. 残差成分极度纯净：**42/43 个 P1/P6 失败是同一个签名** —— 最后一步
   `migration_advisor` 的 `finish_after_tool` 输出 `false`，gold 要求 `true`；
   动作、`use_image`、`use_visual_probe`、`user_query` 全部正确。

这解释了为什么 08-03 的 seed43 GRPO 在 `mean_score` 目标下会100% 零方差空转：
它优化的那 1.87% mean headroom，SFT 已经吃干净了。

## 1. verifier bug：`arg_contains` 的 AND trap

`reward_planner_grpo.py` 的 `arg_contains` 曾用 "ALL" 语义，导致同义词集变成
逻辑与陷阱。提交 `432c186fix(reward): arg_contains uses ANY semantics for
synonym lists` 已修复，但 **08-02 的三场景评测跑在修复之前**，报告未重跑。

失败信息长这样（期望列表里明明包含实际值）：

```text
arg 'end_reason' expected to contain
  ['recheck_done','memory_hit','resolved','done','complete','ok','success','confirmed'],
  got 'memory_hit'
```

影响面：G1_first_success_end 与 P2_all_gates_ok 两个类别的 `end_reason` 被全量误判，
各90 次occurrence。

**处理方式**：不重跑推理。原始 `*_predictions.jsonl` 全部在盘上，直接用修复后的
verifier 重打分，产物落在
`repro_h20/eval/rescored_anyfix/{sft,base4b,base35b}_softbnd/`。

## 2. 重打分前后对照（v7 grpo_dev，240 case × 3 run，temperature=0）

| Arm | mean_score 旧 | mean_score 新 | pass_all_runs 旧 | pass_all_runs 新 |
|---|---:|---:|---:|---:|
| 4B base | 0.7634 | 0.7714 | 0.0000 | 0.0000 |
| 4B SFT ckpt-100 | 0.9704 | **0.9813** | 0.5125 | **0.7625** |
| 35B-A3B base | 0.8525 | **0.8614** | 0.0000 | **0.1833** |

研究目标的三个不等式不受影响，且数据集健康门槛仍然成立：

- `4B base < 4B SFT`：0.7714 → 0.9813
- `4B SFT >= 35B base`：0.9813 vs 0.8614
- 35B base `>= 0.85` 硬门槛：0.8614 ✅

## 3. mean headroom 与 pass headroom 的分裂

| Arm | mean_headroom | pass_headroom | 比值 |
|---|---:|---:|---:|
| 4B SFT | 0.0187 | 0.2375 | **12.7×** |
| 35B base | 0.1386 | 0.8167 | 5.9× |
| 4B base | 0.2286 | 1.0000 | 4.4× |

**对 GRPO 的含义**：以 `mean_score` 为目标时，可优化空间只有 1.87%，且这部分
恰好是 SFT 已饱和的区域 →组内 reward 方差消失。以 `pass_all_runs` 为目标时，
可优化空间是 23.75%，且其中 17个 case 是 flaky（同一 case 3 次重复里有过有对有错），
天然带组内方差。

## 4. 残差归因（4B SFT，重打分后）

`pass_all_runs` 未通过的 57 个 case：

| 类型 | 数量 | 类别分布 |
|---|---:|---|
| stable_fail（3 次全错） | 40 | P1_iou_low_fresh 19；P6_domain_shift 18；G2 3 |
| flaky（有对有错） | 17 | P6 6；G2 4；P1 4；P4 3 |

失败签名 occurrence（3 run 合计）：

| 签名 | 次数 |
|---|---:|
| `step2:finish_after_tool` | 142 |
| `final_tool_not_finished_hit` | 139 |
| `step1:argument_match` | 3 |
| `step2:action_match` | 3 |
| `step2:argument_match` | 3 |
| `repeated_tool_hit` | 3 |

逐 case 检查 P1/P6 的 43 个失败，step-2 的实际输出签名：

| `action` | `finish_after_tool` | `use_image` | `use_visual_probe` | 数量 |
|---|---|---|---|---:|
| `migration_advisor` | **false** | true | true | **42** |
| `qwen_detection` | false | — | — | 1 |

即：**42/43 是单布尔字段错误**。模型选对了工具、选对了两个视觉开关、`user_query`
也复制正确，只在"这是最后一步，应当收口"上判错。

SFT gold 侧无问题，v7 SFT train 里P1/P6 的 step-2 canonical completion 是：

```json
{"action":"migration_advisor",
 "action_input":{"finish_after_tool":true,"use_image":true,"use_visual_probe":true,
                 "user_query":"山地风电塔筒外附件9816号 米黄梯形校准器"}}
```

所以这是**模型真实的决策边界残差**，不是数据缺陷、不是 verifier 缺陷。

## 5. 与 08-03 seed43 空转事故的因果关系

08-03 21:10 同时启动了 3 份 seed43 GRPO（`20260803_2110{47,05,24}`），共享 4 卡。
训练遥测（step 22/100）：

```text
reward = 6.0 (满分)      reward_std = 0.0
frac_reward_zero_std = 0.994 (mean)advantage/std = 0.0
grad_norm = 0.0 (22 步里21 步为 0，仅 step 12 有 0.1266)
route_exact = argument_exact = stop_exact = no_forbidden_action = 1.0
entropy ≈ 0.006
```

对照三次 run 的饱和度演进：

| run | initializer | `frac_reward_zero_std` | `task_reward` mean |
|---|---|---:|---:|
| `20260801_grpo_v7_seed42_r3` | 老 SFT（fp16 栈） | 0.789 | 0.627 |
| `20260802_224145_..._seed42` | `20260802_155804/ckpt-100` | 0.864 | 0.979 |
| `20260803_2110*_..._seed43` |同上 | **0.994** | ~1.000 |

结论：GRPO 的 optimizer pool（`planner_retry_migrate_v7_longobs_grpo_train`，
480 行、`step_index` 全为 2）已被 SFT ckpt-100 完全饱和。按 playbook 的 support
门规则，此时 `optimizer_steps` 必须为 0，因此三份进程已于21:58 终止，4 卡释放。
遥测与日志保留，作为「饱和 → 零梯度」的证据，不删除。

## 6. 下一步被允许做什么

1. **主指标切换**：gate 与 compare 的 primary 改为 `pass_all_runs`，并把
   `final_tool_not_finished_hit` / `premature_stop_hit` / `forbidden:*` 的
   occurrence 作为独立硬门（不得增加）。
2. **reward 权重重配**：当前 `action_match=0.65`、`finish_after_tool=0.10`，而
   99% 的残差在 `finish_after_tool`。权重与残差严重错配，需要把信号密度移到
   实际出错的分量上；同时 `--no-forbidden-action-reward-weight` 当前为 `0.0`，
   应启用。
3. **initializer 降强度**：ckpt-100 已在 optimizer pool 上100% 饱和；
   ckpt-200/300/400 也在盘上，但方向相反。需要更早的 checkpoint 或
   在 optimizer pool 中换入未饱和的 residual。
4. **不允许**：在饱和 pool 上继续加 seed、加步数、升温，或以 `mean_score`
   为主指标宣布 GRPO 成功。

## 7. 复查入口

```bash
# 残差诊断器（只读，纯重打分）
python3 pipelines/eval/diagnose_grpo_headroom.py \
  --arm sft=<eval>/rescored_anyfix/sft_softbnd \
  --arm base4b=<eval>/rescored_anyfix/base4b_softbnd \
  --arm base35b=<eval>/rescored_anyfix/base35b_softbnd \
  --out reports/grpo_headroom_rescored_2026-08-03.json
```

- 重打分前：`reports/grpo_headroom_2026-08-03.json`
- 重打分后：`reports/grpo_headroom_rescored_2026-08-03.json`
- 原始预测：`repro_h20/eval/20260802_172017_sft/softbnd_dev/sft_run{1,2,3}_predictions.jsonl`
- 空转证据：`repro_h20/grpo/20260803_2110*/telemetry/rank*.jsonl`、
  `artifacts/CAPA/logs/train/resume_grpo_20260803_2110*.log`
