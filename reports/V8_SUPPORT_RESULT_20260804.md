# v8 support 结果：方差问题解决了，暴露出真正的瓶颈（2026-08-04）

## TL;DR

第 1 条（扩大决策空间）**成功解决了方差问题**，但没有产出一个可以立刻开跑 GRPO
的池。它把问题从"没有信号"推进到了一个更精确、可诊断的位置：

**`gold_support` 与 `nonzero_variance` 在类别层面呈负相关，r = −0.598。**

gold 高的地方方差低（学会了 → 采样一致 → 无梯度），方差高的地方 gold 低
（没学会 → 采不到正确动作 → GRPO 无法凭空发现规则）。support硬门要求两者
同时成立，而当前 SFT 拿不到这个交集。

## 三个 checkpoint 的完整对照（v8 池，150 组 × 4 gen）

| ckpt | eval_loss | gold | var | forbidden | step2 gold/var | step3 gold/var |
|---:|---:|---:|---:|---:|---|---|
| 50 | ~1.1 | 0.5117 | **0.9000** | 0.3800 | 0.509 / 0.895 | 0.531 / 0.938 |
| **100** | 0.507 | **0.6183** | 0.4200 | **0.3333** | 0.578 / 0.448 | **0.953** / 0.188 |
| 150 | ~0.09 | 0.5767 | 0.6067 | 0.6667 | 0.534 / 0.649 | 0.938 / 0.250 |

门槛：`gold_support ≥ 0.80`、`nonzero_variance ≥ 0.25`。三个 checkpoint 都因
gold 不足而 FAIL。

## 与 v7 的对比：方差问题确实解决了

| | v7 池 | v8 池 |
|---|---|---|
| 动作空间 | 2（migrate 75% / end 25%） | 4（migrate 60/end 20/qwen 10/rex 10） |
| `nonzero_variance` | 0.0000（hint-SFT）→ 0.6500（noHint-SFT ckpt50） | **0.42 – 0.90（全部 checkpoint）** |
| GRPO 可跑步数 | ~30 步后饱和，84 步被 guard 中止 | 未跑（gold 门未过） |
| `forbidden_group_rate` | 0.0000（信号不存在） | **0.33 – 0.67（信号出现）** |

v7 的核心病是"方差随策略变强而消失"。v8 上这个病没了：即使
`eval_loss` 降到 0.09（ckpt-150），方差仍有 0.607。**决策空间从 2 扩到 4 是有效的。**

## 新瓶颈：gold 与 var 此消彼长

ckpt-100 按类别（8 类，19 组/类）：

| category | gold | var | 判定 |
|---|---:|---:|---|
| G2_conflict_stale_history | 1.000 | 0.000 | SATURATED |
| P4_auth_quota | 0.921 | 0.263 | **TRAINABLE** |
| P5_second_failure | 0.917 | 0.167 | SATURATED |
| P1_iou_low_fresh | 0.645 | 0.474 | GOLD_BROKEN |
| P6_domain_shift | 0.583 | 0.778 | GOLD_BROKEN |
| P2_all_gates_ok | 0.329 | 0.789 | GOLD_BROKEN |
| G1_first_success_end | 0.316 | 0.737 | GOLD_BROKEN |
| P3_transient_5xx | 0.250 | 0.158 | DEAD |

**只有 1 个类别（P4）同时满足两个条件。** 任何子集组合都过不了门：

- gold 最高的三类（G2+P4+P5）：gold 0.946，但 var 仅 0.143
- var 最高的三类（P2+P6+G1）：var 0.768，但 gold 仅 0.409

## 根因：SFT 监督里detector 占 55.6%

v8 SFT train 1440 行的动作分布：

```text
step1 (640 行) 全部是 detector          <- trivial：任何 case 第一步都是先探测
step2 (640 行) detector 160 / migrate 320 / end 160
step3 (160 行) migrate 160
------------------------------------------------------------
合计  qwen 400 (27.8%) + rex 400 (27.8%) = detector 55.6%
      migration_advisor 480 (33.3%)
      end 160 (11.1%)
```

`step1` 是无信息量的监督（所有轨迹的第一步都相同），却占了 44% 的训练行，并且
全部指向 detector。后果在checkpoint 序列上看得很清楚：

- 训得少（ckpt-50）：还没学会 step2 该retry，gold 0.512
- 训得多（ckpt-150）：把 detector 过度泛化到所有类别 —— **P6_domain_shift 的
  gold 从 ckpt-50 的 0.694 崩到 ckpt-150 的 0.000**，`forbidden_group_rate`
  升到 0.667（该 migrate 时去retry，命中 step2 的 forbidden 集）

这是playbook 里n52 记录的**条件坍缩**的重演：一个分支学满，另一个分支归零。
n58 的解法是**机械平衡目标动作类**（retry/migrate/end = 24/24/24）。

## 下一步（明确且有依据）

1. **重建 v8 SFT 数据，去掉或大幅降采样 step1 监督。**
   step1 不需要 640 行来教"先调detector"；把它降到 ~80 行，让step2 的三个
   动作类在监督中接近均衡。预期把 detector 占比从 55.6% 压到 ~30%。
2. 重训 SFT → 在 v8 池上重跑 support，目标是把 P3/P6/G1/P2 的 gold 抬起来而
   不牺牲方差。
3. `P3_transient_5xx` 已是 DEAD（gold 0.250 且 var 0.158），需要单独检查它的
   observation 是否真的可判别—— 传输层5xx 与 P4 的 auth/quota 在 NL 错误
   消息层面可能过于相似。
4. `G1/P2` 的 end 分支（四条件合取判断）仍是独立的能力缺陷，两轮下来gold 始终
   在 0.21–0.47之间，需要专门的 SFT 补强。

## 方法论收获

**support 审计必须按 step 和 category 双重分层。** 池级平均会同时隐藏两种病：

- ckpt-100 池级 `var=0.420` 看起来健康，但 step3 只有 0.188（已饱和）
- ckpt-100 池级 `gold=0.618` 看起来"差一点"，实际是 1 个类别 1.000 + 1 个类别
  0.250 的混合，没有任何单一干预能同时修好

本轮已给 `pipelines/eval/audit_grpo_support.py` 加上 `by_step` 分层输出。

**"最早健康 checkpoint"要先满足健康。** v7 上 ckpt-50 是对的（eval_loss 0.467），
但 v8 任务更难，ckpt-50 的 eval_loss 是 ~1.1，远未装好支持。同一条规则在不同
难度的数据集上对应不同的 step 数，不能照搬数字。
