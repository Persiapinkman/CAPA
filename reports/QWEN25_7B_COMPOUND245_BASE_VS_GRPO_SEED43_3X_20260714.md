# Qwen2.5-7B Base vs Runtime-Probe GRPO Seed43：compound245 3x 回归评测

日期：2026-07-14

## 结论

两臂均已完成 `245 cases x 3 repeats`。固定 candidate 相对 raw base 严格多过
5 个 case，pass-all 从 `112/245 = 45.7143%` 提升到
`117/245 = 47.7551%`，mean verifier score 提升 `+0.018788`。三个 repeat 的
pass 集合完全一致，没有 empty decision、timeout 或 rollout error。

该模型仍远未达到历史 Qwen3.5-35B-A3B：35B pass-all 为
`235/245 = 95.9184%`，candidate 仍少通过 118 case，差 `-48.1633` 个百分点。
两个核心多步类别 `probe_then_migration` 和
`migration_feasibility_with_image` 的 strict pass 仍均为 0%。

## 固定模型

- Base：`/raid/zkq/models/Qwen2.5-7B-Instruct`
- Candidate initializer：`/raid/zkq/artifacts/CAPA/outputs/merged-qwen25-7b-sft-v3-chatml`
- Candidate adapter：`/raid/zkq/artifacts/CAPA/outputs/planner-grpo-qwen25-7b-runtime-probe-curriculum-v1-seed43/checkpoint-80`

seed43 在读取 compound245 结果前依据已有 dev 证据固定：它是 seeds 42/43/44
中 dev case-macro 最高、且没有 seed44 新增 Flux 回退的模型。本轮没有用 245
选择 seed 或 checkpoint。

## 协议

- 数据：`planner_grpo_compound245_eval_cases.jsonl`
- SHA256：`cd0f03acdb15f586386c50936b4f997b7b1226732b37eb1a3409076ca2e02388`
- `temperature=0`, `top_p=1`, `do_sample=false`, `seed=42`
- `max_tokens=128`, `max_steps=3`, step timeout `180s`
- legacy action space：`CAPA_ENABLE_ADELA=1`
- 本地 Transformers fp16 服务，V100-32GB
- run1/run2 在主服务顺序执行；repeat3 在同配置副本 GPU 并行执行
- 只执行 Planner + mock observation，不执行真实工具，也没有外部模型调用

Candidate 使用其 SFTv3 产物自带 tokenizer；服务环境对该 tokenizer 给出了
`fix_mistral_regex` 兼容警告。本轮为保持既有 artifact 原样没有临时修改 tokenizer，
因此该警告和 base/candidate tokenizer 不同均属于结果适用边界，后续 clean-lineage
训练前需要单独消解。

## 总体结果

| Arm | 每轮通过 | pass-rate mean | pass-all | mean score | score stdev | decisions/轮 | error/empty |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw base | 112 / 112 / 112 | 0.457143 | 0.457143 | 0.910996 | 0.000273 | 306 | 0 / 0 |
| SFTv3 + GRPO seed43 | 117 / 117 / 117 | 0.477551 | 0.477551 | 0.929784 | 0.001256 | 306 | 0 / 0 |
| Delta | +5 / +5 / +5 | +0.020408 | +0.020408 | +0.018788 | — | 0 | 0 |

配对结果为 5 个 improvement、0 个 strict regression。case bootstrap 95% 区间为
`[+0.004082, +0.040816]`；McNemar 双侧 exact `p=0.0625`。由于 compound245
是已暴露的 legacy regression，这些统计量只做描述，不作为泛化显著性证据。

## 类别结果

| Category | Base pass-all | Candidate pass-all | Delta | Base score | Candidate score | Score delta |
|---|---:|---:|---:|---:|---:|---:|
| `adela_eval` | 0.5000 | 0.5000 | +0.0000 | 0.9500 | 0.9500 | +0.0000 |
| `full_detection_eval` | 0.4000 | 0.8000 | +0.4000 | 0.9100 | 0.9800 | +0.0700 |
| `general_answer` | 0.0000 | 0.0000 | +0.0000 | 0.7000 | 0.7000 | +0.0000 |
| `historical_asset_qa` | 1.0000 | 1.0000 | +0.0000 | 1.0000 | 1.0000 | +0.0000 |
| `intent_ambiguity` | 1.0000 | 1.0000 | +0.0000 | 1.0000 | 1.0000 | +0.0000 |
| `migration_feasibility` | 0.7105 | 0.7895 | +0.0789 | 0.9711 | 0.9789 | +0.0079 |
| `migration_feasibility_with_image` | 0.0000 | 0.0000 | +0.0000 | 0.8232 | 0.8353 | +0.0121 |
| `probe_then_migration` | 0.0000 | 0.0000 | +0.0000 | 0.8469 | 0.9016 | +0.0547 |
| `single_image_probe` | 1.0000 | 1.0000 | +0.0000 | 1.0000 | 1.0000 | +0.0000 |

新增严格通过的 5 条 case：

- `GRPO-PIPELINE-003`
- `GRPO-EXP-BOUNDARY-005`
- `GRPO-MIG-002`
- `GRPO-EXP-MIG-TXT-011`
- `GRPO-EXP-MIG-TXT-018`

## 多步失败变化

按三轮平均 failure count，candidate 确实学到了部分状态转移：

| Failure | Base | Candidate | Delta |
|---|---:|---:|---:|
| 应转 `migration_advisor`、实际重复 `qwen_detection` | 19.3 | 11.0 | -8.3 |
| `use_visual_probe=true`、实际 false | 32.3 | 13.7 | -18.7 |
| 缺失 `use_image=true` | 19.3 | 11.0 | -8.3 |
| 缺失 `use_visual_probe=true` | 19.3 | 11.0 | -8.3 |
| step1 false 输出成字符串 `"false"` | 9.0 | 0.0 | -9.0 |
| 最终 `finish_after_tool` 应 true、实际 false | 113.7 | 115.7 | +2.0 |

也就是说，GRPO candidate 更常在第二步选择 migration，并修复了布尔字符串和
部分视觉字段，但最终停止位仍系统性错误。由于 strict pass 要求 action、参数和
停止全部正确，多步 dense score 明显上升，却没有转化为任何多步 strict pass。

## Repeat 稳定性

两臂 245 个 case 的 pass/fail 在三轮间 100% 一致。结构化动作序列完全一致的
case 为 base `244/245`、candidate `242/245`；忽略 thought/timing 后核心 decision
完全一致的 case 为 base `238/245`、candidate `239/245`。因此 pass 稳定，但少量
argument/action 文本存在 GPU/repeat 级波动。

## 数据血缘边界

GRPO 阶段使用的 540 条 `planner_runtime_probe_curriculum_v1` train cases 与
compound245 的 case_id 和规范化 exact query 均为 0 交叉，符合本轮指定条件。

但最终 candidate 并不是从 raw base 直接做 GRPO；它从 SFTv3 initializer 开始。
SFTv3 的 123 条 train split 与 compound245 有 69 条 case_id/query 精确重合。
因此准确表述是“GRPO 新数据零交叉”，不能表述为“candidate 全训练血缘与 245
零交叉”，也不能把 base-to-candidate 的全部增量归因于 GRPO。

## 产物

- Base aggregate：`/raid/zkq/artifacts/CAPA/evals/20260714_compound245_base_vs_grpo3x/base/qwen25_7b_base_compound245_t0_3x_20260714_aggregate.json`
- Base case audit：同目录 `qwen25_7b_base_compound245_t0_3x_20260714_case_audit.csv`
- Candidate aggregate：`/raid/zkq/artifacts/CAPA/evals/20260714_compound245_base_vs_grpo3x/grpo_seed43/qwen25_7b_sftv3_grpo_runtime_probe_seed43_compound245_t0_3x_20260714_aggregate.json`
- Candidate case audit：同目录 `qwen25_7b_sftv3_grpo_runtime_probe_seed43_compound245_t0_3x_20260714_case_audit.csv`
- Machine-readable comparison：`experiments/runs/20260714_qwen25_7b_compound245_base_vs_runtime_probe_grpo_seed43_3x/metrics.json`

本轮仅评测，所有本地模型服务已停止，没有启动训练。
