# H20 v7 三场景 3× 评测对照

_数据集_: `planner_retry_migrate_v7_longobs`（240 case，注入显式 routing hint 让 base 可读）

_Repro root_: `/apdcephfs_hzlf/share_1227201/zkq/capa_h20/artifacts/CAPA/repro_h20`

_评测参数_: `temperature=0 top_p=1 seed=42 runs=3`；vLLM `max_model_len=32768`，planner `max_tokens=512`。


## 核心场景：softbnd_dev（240 case × 3 run）

| Arm | mean_score | pass_rate | Δ 相对 4B base | 备注 |
|---|---:|---:|---:|---|
| **4B base** | **0.7634 ± 0.0008** | 0/240 | — | 起点 |
| **35B base** | **0.8525 ± 0.0055** | 0/240 | +8.9 pp | 数据集判据 ≥ 0.85 ✓ |
| **4B SFT ckpt-100** | **0.9704 ± 0.0015** | 131/240 (54.9%) | **+20.7 pp** | 超越 35B base +11.8 pp |
| 4B SFT+GRPO ×3 | — | — | — | GRPO 无学习信号（SFT 饱和，见 EXECUTION_STATUS） |

## 按类别的 SFT ckpt-100 vs 35B base 对比（3-run mean）

| Category | 35B base | 4B SFT | Δ (SFT − 35B) |
|---|---:|---:|---:|
| P3_transient_5xx | 0.8804 | **1.0000** | +11.96 pp |
| P5_second_failure | 0.8498 | **1.0000** | +15.02 pp |
| P4_auth_quota | 0.8312 | **0.9971** | +16.59 pp |
| G2_conflict_stale_history | 0.7895 | **0.9870** | +19.75 pp |
| G1_first_success_end | 0.8854 | 0.9565 | +7.11 pp |
| P2_all_gates_ok | 0.8674 | 0.9565 | +8.91 pp |
| P6_domain_shift | 0.8348 | 0.9382 | +10.34 pp |
| P1_iou_low_fresh | 0.8819 | 0.9283 | +4.64 pp |

## 辅助场景（非 v7 训练数据，仅 SFT eval 记录）

| Scenario | 4B base | 4B SFT | Δ |
|---|---:|---:|---:|
| routing90 (90 case) accuracy | 0.7111 | 0.6289 ± 0.0052 | −8.22 pp（负迁移，非 v7 分布） |
| multistep focused_val_v3 (31 case) | 0.8032 | 0.890 | +8.7 pp |

routing90 负迁移：SFT 用 v7 长观测 + hint 训练，泛化到 90-case（不同 prompt 风格）时权重被拉偏。这是训练领域外的预期代价，不影响用户目标（softbnd 场景）。

## 数据 / 编排关键修复

见 `H20_V7_EXECUTION_STATUS.md` §三、§五.3。
