# V10 selection-dev 结论

状态：`no_promotion`。V10 sealed-test A/B 均保持未物化，35B reference 未参与 checkpoint 选择。

## 冻结门禁结果

选择集包含 12 个独立实体、216 个 case；SFT 与 checkpoint-2/5/10 均完整覆盖 216/216，JSON valid rate 为 100%，clipping 与 Planner runtime error 均为 0。

| 模型 | Primary pass | 相对 SFT | Entity bootstrap 95% CI | Control pass | 相对 SFT | Wrong side-effect actions | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SFT | 8.33% | — | — | 45.83% | — | 35 | reference |
| checkpoint-2 | 9.72% | +1.39pp | [0.00pp, +4.17pp] | 50.00% | +4.17pp | 35 | primary < +5pp |
| checkpoint-5 | 9.72% | +1.39pp | [-2.78pp, +5.56pp] | 62.50% | +16.67pp | 41 | primary 与副作用门禁失败 |
| checkpoint-10 | 16.67% | +8.33pp | [+1.39pp, +15.28pp] | 65.28% | +19.44pp | 37 | 副作用净增 2，拒绝 promote |

V10 的 control replay 解决了 V9 的灾难性遗忘：checkpoint-10 不仅保住 control，还比 SFT 提高 19.44 个百分点。它也通过了 primary、control、JSON 和 clipping 门禁；唯一失败项是预注册的“不得新增错误副作用动作”。因此不能用整体 pass/reward 的提升覆盖安全失败。

## 副作用定位

checkpoint-10 与 SFT 共有 14 个 case 的 forbidden-hit 集合发生变化：移除 6 个、引入 8 个，净增 2 个。

| 场景 | SFT hits | checkpoint-10 hits | 引入 | 移除 | 净变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `current_success_step2` | 16 | 12 | 1 | 5 | -4 |
| `fresh_retry_step2` | 9 | 13 | 4 | 0 | +4 |
| `post_retry_success_step3` | 10 | 11 | 2 | 1 | +1 |
| `nonretryable_step2` | 0 | 1 | 1 | 0 | +1 |

新增错误主要是成功重试后误触 `migration_advisor`；`nonretryable_step2` 的单个新增错误是从 Qwen 路由到 Rex detector，而不是应当调用 `migration_advisor`。完整 case/action 证据保存在 `selection_side_effect_audit.json`。

## 对下一版的约束

V11 不得放宽任何 promotion gate，也不得打开 V10 sealed test。下一版应使用新的 entity-disjoint selection/sealed commitment，并在 optimizer reward 中给现有 `no_forbidden_action` 诊断项非零权重；训练 initializer、步数、primary/control replay 比例与候选 checkpoint 必须在新 support sampling 和 optimizer step 前预注册。
