# Qwen3.5 4B→35B 能力阶梯：V15 最终交接

_最终更新：2026-07-20 UTC · 主目标已达成 · Canonical artifact：`planner_retry_ladder_v15_n67/final_open_once`_

## 最终结论

在同一套全新、一次性封存的 CAPA Planner V15 场景上，四臂三次完整运行得到：

| 模型 | Run 1 (%) | Run 2 (%) | Run 3 (%) | Mean (%) | Range (pp) |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-4B Base | 14.8000 | 14.8000 | 14.8000 | **14.8000** | 0.0000 |
| Qwen3.5-4B original SFT | 75.0000 | 67.6000 | 81.4667 | **74.6889** | 13.8667 |
| Qwen3.5-35B-A3B | 87.9333 | 95.3333 | 93.4667 | **92.2444** | 7.4000 |
| Qwen3.5-4B targeted-SFT + one-step GRPO（LR 2e-8） | 100.0000 | 100.0000 | 100.0000 | **100.0000** | 0.0000 |

因此严格均值排序成立：

`4B Base 14.80 < 4B SFT 74.69 < 35B 92.24 < 4B GRPO 100.00`

用户要求的门槛全部满足：Base `<65%`；35B 均值和每一次运行都 `>85%`；GRPO 超过 35B `7.7556pp`。四个相邻/阈值约束中的最小余量为 35B 最弱一次相对 85% 下限的 `2.9333pp`。

需要保留一个审计限定：机器报告状态为 `fail`，唯一原因是研究中额外预注册了更严格的 `35B run range <=5pp`，实际为 `7.4pp`。这不改变上述表格和用户门槛，但不能把额外稳定性门称为通过。对本项目，“35B 稳定超过 85%”成立；“35B 三轮极差不超过 5pp”不成立。

## 场景定义

V15 测的是 Planner 对结构化工具 observation 的多步路由，不是真实视觉检测精度。每个 case 要完成严格的完整轨迹：

- `post_retry_metric_veto_step3`：先调用指定 detector；仅在可重试错误且 `retry_count=0` 时重试同一 detector；重试后的最新 metric 任一未过阈值时进入 `migration_advisor`
- `current_success_step2`：detector 最新回执同时满足 candidate count、置信度、跨提示一致性和低域偏移时结束
- detector 包含 Qwen 与 Rex-Omni 两类；每类场景×detector 精确 6 条
- 6 个新实体，每个实体覆盖 `2 scenarios × 2 detectors`；总计 24 条
- 新实体词表、新 error alias、新 fixture 名称与两张新图片；20 个保护词扫描 558 个历史/训练文件后精确重合为 0
- 无 case 级筛选、无结果后改比例、无 confirmation 参与训练或 checkpoint 选择

主分数为严格完整轨迹通过率的预注册加权：

`score = (111 × metric-veto pass rate + 14 × current-success pass rate) / 125`

所有模型共享 temperature `0`、top-p `1`、`do_sample=false`、最大 3 个 Planner step、最大 4096 新 token、相同 parser/verifier。每臂 3 次完整运行，每次精确 24 行；最终共 288 条顶层预测，prediction runtime error 为 0。

## 模型与训练血缘

| 臂 | 冻结资产 | 核心标识 |
|---|---|---|
| Base | `/raid/zkq/models/Qwen3.5-4B` | 模型文件 hash 见 V15 config |
| SFT | `experiments/runs/20260716_qwen35_4b_planner_v6_sft_seed42_v1/checkpoint-100` | adapter `0c094520…a60` |
| 35B | gateway model ID `Qwen3.5-35B-A3B` | 固定 endpoint descriptor `290c5fd5…c44` |
| GRPO | `/raid/zkq/artifacts/CAPA/arbor/ladder_n64/grpo_lr_grid_20260720T0745Z/lr2e-8/checkpoint-1` | adapter `a962c637…0c2` |

最终 GRPO 臂不是旧 V12 checkpoint。它的实际血缘是：

`Qwen3.5-4B → targeted SFT warm-start (n58 checkpoint-6) → 1 个真实 GRPO optimizer step (n64, LR 2e-8)`

该 GRPO step 使用冻结的 32-row residual optimizer set、seed 42、temperature 1.3、top-p 0.95、32 个 completion、4-GPU DDP；reward 权重为 task `0.75`、format `0.05`、no-forbidden-action `0.20`。健康证据为 clip `0`、reward std `0.196158`、advantage std `0.432900`、grad norm `0.136726`，且 adapter SHA 从 `088f1ef3…123` 变为 `a962c637…0c2`，所以不是名义上的零更新。

## 实验演进与关键否证

```mermaid
flowchart LR
    accTitle: Qwen3.5 capability-ladder experiment progression
    accDescr: The study moved from a V13 development ladder through a failed disjoint V14 confirmation, checkpoint and learning-rate diagnostics, and finally a fresh V15 confirmation that met the requested ordering and thresholds.

    v13["V13 开发场景<br/>排序首次成立"] --> v14["V14 全新词表<br/>排序失败"]
    v14 --> cp["n63 checkpoint 复查<br/>cp6 最稳"]
    cp --> lr["n64 小学习率 GRPO<br/>2e-8 入选"]
    lr --> repeat["n65 双开发集 3x<br/>识别固定残差"]
    repeat --> seal["n66 预注册并封存 V15"]
    seal --> result["n67 单次开封<br/>目标阶梯成立"]

    classDef positive fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef negative fill:#fee2e2,stroke:#dc2626,color:#7f1d1d
    classDef process fill:#dbeafe,stroke:#2563eb,color:#1e3a5f
    class v13,lr,result positive
    class v14 negative
    class cp,repeat,seal process
```

| 节点 | 结果 | 决策价值 |
|---|---|---|
| n41 | 旧 GRPO 在 V13 confirmation 上失败 | 不能靠事后换场景解释旧 checkpoint |
| n48–n57 | 多轮 SFT/GRPO 深度、replay 与 clipping 尝试未形成阶梯 | 明确过深 SFT 和大更新会破坏边界稳定性 |
| n58–n60 | targeted SFT + 一步 GRPO 在 V13 开发集达到 100%，固定表为 39.47/79.02/96.27/100 | 找到首个候选，但尚未证明跨词表泛化 |
| n62 / V14 | Base 29.60、SFT 100、35B 94.11、n60 GRPO 85.20 | 关键否证：V13 成功未泛化；原 SFT 在 V14 措辞上饱和 |
| n63 | cp6 为 92.6/92.6/92.6，cp12 均值更高但波动，cp18 退化 | 选择零范围 cp6，不追逐单次峰值 |
| n64 | 一步 GRPO LR grid：5e-9=70.4、1e-8=92.6、2e-8=100 | 选定真实非零的 LR 2e-8 更新 |
| n65 | n64 候选 V13=100/100/100；V14=100/92.6/92.6 | 仅在 V13 稳定结构上封存下一 confirmation |
| n66–n67 / V15 | 全新实体/词表一次性四臂 3x，得到最终表 | 用户目标成立；额外 5pp range 门失败需披露 |

完整 68-node 探索树保存在 [`.arbor/tree.json`](.arbor/tree.json)。n63–n65 的详细记录分别见 [checkpoint screen](N63_V14_OPEN_DEV_CHECKPOINT_SCREEN.md)、[small-LR grid](N64_SMALL_LR_GRPO_GRID.md) 和 [multi-repeat validation](N65_MULTI_REPEAT_DEVELOPMENT_VALIDATION.md)。

## Canonical 证据与哈希

| 证据 | 路径 | SHA-256 |
|---|---|---|
| V15 generation spec | `experiments/studies/planner_retry_ladder_v15_confirmation_v1/generation_spec.json` | `93c4ace6954b2cd2614118d8881e4bbb8d7e866cdb7defe1f89000c5033796fb` |
| V15 cases | `experiments/studies/planner_retry_ladder_v15_confirmation_v1/sealed_data/v15_confirmation_cases.jsonl` | `37d9a739585c012041397fba77b995d5842359ba5e23a8e8f8f604578e4ade78` |
| V15 manifest | `experiments/studies/planner_retry_ladder_v15_confirmation_v1/sealed_manifest.json` | `eabf39b49c5c589f53156558a90a9dbc30c6c0a734525bb58fd540f06a6d992f` |
| V15 config | `configs/eval/qwen35_v15_final_ladder.json` | `063932854510fafe7b7952ed8a3f2d937bbdbf98f78710b3dec849b8ebfacae0` |
| Opening receipt | `/raid/zkq/artifacts/CAPA/final/planner_retry_ladder_v15_n67/final_open_once/opening_receipt.json` | `1c15df9c3939f2bcad4b9f274196813ff2f184256a906f9f74498fb1a0c8ab1a` |
| Final report | `/raid/zkq/artifacts/CAPA/final/planner_retry_ladder_v15_n67/final_open_once/final_report.json` | `4bc4a819cf6f2291ac125c33b0a42133c679d1dbbb21d1dc54bb0337db8dcfd7` |
| Final table | `/raid/zkq/artifacts/CAPA/final/planner_retry_ladder_v15_n67/final_open_once/final_table.md` | `389e2b5d51d0cb383fa8021bdcac47b26eb88b152ea559c5d063c905015f2207` |

V15 的仓内结果摘要见 [RESULT.md](../planner_retry_ladder_v15_confirmation_v1/RESULT.md)，冻结配置见 [qwen35_v15_final_ladder.json](../../../configs/eval/qwen35_v15_final_ladder.json)。

## 复查与复现

V15 是 open-once confirmation，正式 runner 会拒绝复用现有输出目录；不要删除目录后重跑并把它仍称为 V15。只读复查：

```bash
ROOT=/raid/zkq/artifacts/CAPA/final/planner_retry_ladder_v15_n67/final_open_once
jq '{status,table,hard_gates,grpo_minus_larger_mean_pp}' "$ROOT/final_report.json"
sha256sum "$ROOT/opening_receipt.json" "$ROOT/final_report.json" "$ROOT/final_table.md"
```

仓内冻结资产与几何测试：

```bash
/raid/zkq/projects/CAPA/.venv-qwen35-grpo/bin/python -m pytest -q \
  tests/test_planner_retry_ladder_v15_confirmation.py \
  tests/test_qwen35_v15_final_ladder.py
```

执行入口为 `scripts/run_qwen35_v15_final_all_scopes.sh`，但它只用于一个全新的、尚不存在的 output root。若未来要求把 35B 极差也压到 `<=5pp`，应建立 V16、预注册更大的独立实体样本或新的稳定性统计合同；不得挑选 V15 run 或删掉失败的额外门。

## 后续人工审查要点

1. 最终表的 4B-SFT 指原始 V6 SFT checkpoint-100；GRPO 臂包含 targeted-SFT warm-start，不能写成“从该表 SFT 臂只做一步 GRPO”。
2. 100% 是 24-case 场景内的 strict Planner trajectory accuracy，不应外推为通用 Planner 或视觉模型 100%。
3. 三次 deterministic inference 验证的是 checkpoint 推理稳定性，不等价于 seed 42/43/44 的训练复现。
4. 用户要求的 35B 稳定下限通过；额外 `range<=5pp` 未通过。这两个结论必须同时出现。
5. V14 是有效失败，不应删除：它证明措辞家族会改变 SFT 饱和和 GRPO 泛化，直接促成了 n64/V15。
