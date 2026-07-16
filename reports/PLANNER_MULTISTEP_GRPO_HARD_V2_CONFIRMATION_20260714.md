# Qwen2.5-7B 多步工具路由 V2 新评测结果

日期：2026-07-14
范围：方向一，Planner 多步工具路由；本阶段只构造/评测数据，未启动 SFT 或 GRPO。

## 结论

`planner_multistep_grpo_hard_v2` 满足本轮模型差分目标，并以整版方式通过冻结的
confirmation gate：

| Arm | run1 | run2 | run3 | pass-rate mean | pass-all | pass-any | mean score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-35B-A3B | 593/600 | 594/600 | 591/600 | 98.7778% | 583/600 = 97.1667% | 100% | 0.999496 |
| Raw Qwen2.5-7B-Instruct | 146/600 | 147/600 | 147/600 | 24.4444% | 136/600 = 22.6667% | 156/600 = 26.0000% | 0.724724 |

pass-all 差值为 74.5pp。以 75 个 `entity_id` 为 cluster 做 20,000 次 paired
bootstrap：

- 35B pass-all 95% CI：95.50%–98.67%；
- raw 7B pass-all 95% CI：18.17%–27.17%；
- `35B - 7B` 差值 95% CI：69.67–79.33pp；
- paired cases：35B-only 449、7B-only 2、both 134、neither 15。

冻结门要求 overall 35B >=95%、7B <=60%、gap >=30pp；每个 family 35B >=90%、
7B <=75%；600 cases、3 repeats、零最终错误、零空 decision、零未恢复截断。所有
检查均通过，没有对 confirmation 做逐题删除。

## 数据构造与 family calibration

V1 calibration 有 288 cases / 12 families，仅 2 个 family 达到能力差分门，因此
整版失败且未生成 V1 confirmation。V2 使用 V1 未暴露的新实体重新构造：

| Split | cases | decisions | entity clusters | families |
|---|---:|---:|---:|---:|
| V2 calibration | 384 | 640 | 32 | 12 |
| V2 confirmation | 600 | 1,200 | 75 | 8 |

V2 calibration 上，35B 为 383/384（99.74%），raw 7B 为 148/384（38.54%）。
完整 family gate 准入 8 个两步 family；准入 slice 合计 256 cases，35B 为
255/256（99.61%），7B 为 20/256（7.81%）。四个 direct image migration
family 因两模型均为 100% 被整族排除。

Confirmation 使用冻结 allowlist、新实体、新模板和新 fixture family，一次性生成
75 x 8 = 600 cases。它与 calibration 的 case ID、entity ID、template ID、规范化
query 和 fixture family 均零交叉；与仓库已有 case 文件的 case ID/规范化 query
审计也为零交叉。

文件 hash：

- calibration cases：`9deb1457f15ecaa5aa0642731ad0b5e479659e7b5750cba6364cc04f8992ad24`
- confirmation cases：`42201f057366411fdfab77621f6b54e3f4c16d7fcdc8109616065e6212cc3008`
- calibration steps：`4378974617020be8244d6f6d44e7fbe178f7d7d2d05b6e55bec2ace57db78fee`
- confirmation steps：`9490cabf0c27c22e5d8d4e858bbdddcb9bc81be36a3fb9ab9d4a64ce5929ab22`

## Confirmation family 结果

下表均为 3x strict pass-all：

| Family | 35B | raw 7B | gap |
|---|---:|---:|---:|
| `qwen_box_variance_to_migration` | 100.00% | 30.67% | 69.33pp |
| `qwen_empty_result_to_migration` | 97.33% | 6.67% | 90.67pp |
| `qwen_domain_shift_to_migration` | 98.67% | 56.00% | 42.67pp |
| `rex_box_variance_to_migration` | 94.67% | 12.00% | 82.67pp |
| `rex_empty_result_to_migration` | 94.67% | 2.67% | 92.00pp |
| `rex_domain_shift_to_migration` | 93.33% | 4.00% | 89.33pp |
| `qwen_confident_stop_guardrail` | 98.67% | 56.00% | 42.67pp |
| `rex_confident_stop_guardrail` | 100.00% | 13.33% | 86.67pp |

35B 每轮生成 1,200 个决策，三轮均无 API error 或空 decision。2,048-token
正式协议下，35B 有 5 次首调用达到长度上限，但全部由 Planner parser retry
恢复；retry 本身没有截断。raw 7B 没有 retry 或截断。该稳定性缺陷单独报告，
不与 strict correctness 混合。

## GRPO sampling support

在 V2 calibration 的 accepted families 上抽取 128 个 step-level prompts，每题对
raw 7B 采样 8 次，共 1,024 completions（temperature=0.7, top_p=0.9）。

把饱和 step1 与 evaluation-only stop guardrail 混在一起的严格全局门失败：

- nonzero reward variance：54.69%（门槛 80%）；
- usable support：44.53%（门槛 80%）；
- near-exact task support：57.03%（门槛 80%）；
- fully saturated：43.75%（上限 25%）。

按设计角色只看六个 primary migration family 的 48 个待学习 step2：

- nonzero reward variance：91.67%；
- usable dense-reward support：83.33%；
- exact gold-action support：85.42%；
- mean distinct valid actions：1.8125；
- fully saturated：0%。

因此状态转移场景具有 GRPO 所需的探索和组内 reward 方差，但完整 action+typed
args+finish 的 exact support 只有 2/48（4.17%）。执行结论不是“raw base 可直接
GRPO”，而是“先做短 SFT 建立精确参数支持，再复验，过门后才做 GRPO”。

长度审计：calibration/confirmation 最大 prompt 为 3,976/3,938 tokens；canonical
completion 最大 73/72 tokens。随机 support completion 最大 211 tokens，174/1,024
超过旧计划的 128，因此未来训练 completion 上限应为 256。

## 运行协议与有效性

- local base：`/raid/zkq/models/Qwen2.5-7B-Instruct`，fp16/SDPA，本地服务；
- reference：`Qwen3.5-35B-A3B`，调用前加载 `/raid/zkq/projects/CAPA/init_env.sh`；
- `temperature=0`, `top_p=1`, `do_sample=false`, `seed=42`, `max_steps=3`；
- final protocol：`max_tokens=2048`, timeout/openai-timeout 300 秒；
- `CAPA_ENABLE_ADELA=0`；
- strict named Qwen/Rex action、strict JSON boolean、typed args、exact stop；
- 23 项本轮相关 data/verifier/eval 单元测试全部通过。

384-token 首次尝试因 4 次首答截断和 3 个 timeout 被整批作废；768-token 重跑因
1 次首答截断被整批作废。两批均在查看 correctness 前停止并保留于
`invalid_t384` / `invalid_t768`，没有用于指标。最终协议与修订记录见
`protocol_amendment_t768.json` 和 `protocol_amendment_t2048.json`。

## 产物

- 数据卡：`data/datasets/planner_multistep_grpo_hard_v2/DATASET_CARD.md`
- build report：`data/datasets/planner_multistep_grpo_hard_v2/build_report.json`
- calibration gate：`data/datasets/planner_multistep_grpo_hard_v2/calibration_family_gate.json`
- confirmation gate：`data/datasets/planner_multistep_grpo_hard_v2/confirmation_gate.json`
- cluster bootstrap：`data/datasets/planner_multistep_grpo_hard_v2/confirmation_cluster_bootstrap.json`
- support global gate：`data/datasets/planner_multistep_grpo_hard_v2/grpo_support_gate.json`
- support primary-step2 gate：`data/datasets/planner_multistep_grpo_hard_v2/grpo_support_gate_primary_step2.json`
- base aggregate：`/raid/zkq/artifacts/CAPA/evals/20260714_planner_multistep_grpo_hard_v2/confirmation/base_combined/qwen25_7b_base_v2_confirmation_t0_3x_aggregate.json`
- 35B aggregate：`/raid/zkq/artifacts/CAPA/evals/20260714_planner_multistep_grpo_hard_v2/confirmation/qwen35_combined/qwen35_a3b_v2_confirmation_t0_3x_aggregate.json`

## 下一步（未执行）

1. 业务人员完成 600/600 gold 的 100% review 并签字；自动检查不能替代业务审核。
2. 使用全新实体/模板/fixture 构建 train/SFT split；禁止使用 confirmation gold。
3. 从 raw Qwen2.5-7B 做短 SFT initializer；复验 strict full-decision support。
4. 只有复验过门才启动 primary step2 的 dense-reward GRPO；step1/stop 作为
   SFT/评测 guardrail。
5. 若训练人员读取 confirmation 逐题 bad case 来调数据或超参，该集必须降级为
   test-visible，并另建外部隔离的 final sealed test。

本轮没有生成批准的 train split，也没有启动任何训练。
