# CAPA H20 —— 性能对比 + 下一版软边界数据集设计

_2026-08-01 起草，先不启动训练_

## 1. 4B vs 35B-A3B 性能画像（H20 实测）

单条数据源：`capa_h20/artifacts/CAPA/repro_h20/eval/20260801_{123112_base_4b,125005_base_35b}/`。GPU：NVIDIA H20，97 GiB/卡；4B TP=1（GPU 0），35B-A3B TP=4（GPU 0-3）；bf16；vLLM 0.8.5.post1，`temperature=0 top_p=1 seed=42 runs=3`。

### 1.1 显存与容量

| 项 | Qwen3.5-4B (TP=1) | Qwen3.5-35B-A3B (TP=4) |
|---|---|---|
| 权重 shard 数 | 3 | 16 |
| 磁盘占用 | 7.6 GiB | 57 GiB |
| 服务时单卡显存 | ~16 GiB（`gpu_memory_utilization=0.9` 下预留了 KV 缓冲） | 4 张 × ~16 GiB（同 profile）|
| 剩余卡数 | **3 张空闲** | 0 张空闲 |
| GPU KV cache | 314,704 tokens | 4 × ~2.75 M tokens |
| 8k prompt 最大并发 | ≈38.4× | 4 × ≈335.4× ≈ **1340×** |
| torch.compile 首次 | 已包含在 <10 s 加载 | 51–52 s（4 rank 并发） |
| 冷启动到 ready | ~10 s | ~9 min（加载 shard + compile） |

### 1.2 单请求延迟与吞吐（routing90, prompt≈3.4k / output≈88 tokens）

| 指标 | 4B | 35B-A3B | 相对 |
|---|---:|---:|---:|
| `api_call_ms_avg` | 963 ms | 648 ms | 35B 快 **33%** |
| `api_call_ms_min` | 586 ms | 346 ms | |
| `api_call_ms_max` | 1608 ms | 3576 ms | 35B 尾延迟更差 |
| `case_elapsed_ms_avg` | 984 ms | 670 ms | |
| 单 run 90 case 墙钟 | 89.0 s | 60.7 s | |

### 1.3 场景级 3-run 总墙钟

| Scenario | 4B 3× (s) | 35B 3× (s) | 35B/4B |
|---|---:|---:|---:|
| routing90 (90 case) | 273 | 178 | 0.65 |
| multistep (31 case × ≤3 steps) | 161 | 131 | 0.82 |
| softbnd_dev (225 case × ≤3 steps) | 667 | 567 | 0.85 |

35B 相对 4B **单请求快 20-35%**，但把 4 张卡都占了；4B 只吃 1 张卡还剩 3 张可训练/评测。

### 1.4 GPU-hour 折算（H20 单卡 · 单小时为单位 = 1 gpu·h）

| 阶段（一次全量） | 4B | 35B-A3B |
|---|---:|---:|
| 三场景 3× eval | 1 卡 × 1101 s ≈ **0.31 gpu·h** | 4 卡 × 876 s ≈ **0.97 gpu·h** |
| SFT 400 step（历史 6×V100 79 min 参考，4×H20 bf16 约 30-40 min）| **~2.5 gpu·h** | 需要 8×H20 或 ZeRO-3；至少 **20 gpu·h** 且需换机 |
| GRPO 100 step | ~2 gpu·h/seed × 3 = **6 gpu·h** | ≥ 30 gpu·h × 3 = **90 gpu·h** |
| 单次完整流水线（SFT+GRPO×3+全套 eval） | **≈10 gpu·h** | **≈120 gpu·h**（且训练不可行） |

结论：**35B-A3B 在 4×H20 上只能做推理，做不了训练**（KV+梯度+优化器状态 ~250 GB 已经超过 4×97=388 GB 里能给训练用的部分，且训练要求 attention/MoE gate 都能 backward）。所以"4B SFT+GRPO ≥ 35B base"是唯一可能的路径——4B 侧付得起训练开销，35B 侧付不起。

### 1.5 精度 vs 成本速览

| 场景 | 4B_base | 35B_base | Δ | 是否值得用 35B |
|---|---:|---:|---:|---|
| routing90 accuracy | 0.7111 | 0.7148 | +0.4 pp | 否（几乎无差） |
| multistep case_macro | 0.803 | 0.894 | +9.1 pp | 部分（多步优势主要在这类） |
| softbnd_dev case_macro | 0.513 | 0.516 | +0.3 pp | **否——两模型都没掌握** |

**核心结论**：软边界是"模型规模无法解决"的能力缺口，规模换不来 pass 率——必须靠 SFT/GRPO 把决策规则内化进去。这就是设计新数据集的必要性。

## 2. 现有 v6 数据集的缺陷（下一版要修的病）

一个真实 `mock_observation`：

```json
{
  "status": "gateway_error",
  "success": false,
  "summary": "当前结构化状态：candidate_count=NA；min_confidence=NA；cross_prompt_iou=NA；domain_shift=unknown；gateway_error=detector_admission_window_full；retryable=false；retry_count=0。说明：展示字段不覆盖结构化值。"
}
```

问题：

1. **observation 是规则字段的字符串拼接**，模型只要正则出 `retryable=false` 就能给出正确 action；这不是"读长上下文做决策"，是"读结构化 flag 跳转"。
2. **上下文长度只有 ~4.2k tokens**，主要被 system prompt 撑起，实际检测/工具返回可能只占几十 token；长上下文能力完全没被压测。
3. `user_query` 里**已经把规则完整复述了一遍**（"技术错误下仅 fresh retryable 状态可以复用…"），等于把答案抄在题面上——SFT 就是背 query 里的规则子串。
4. 三支路由的判别 `retryable ∈ {true,false}` × `retry_count ∈ {0, ≥1}` 是**离散低维空间**，SFT 已在 sft_dev 上 `action_match=0.977`，`nonzero_reward_variance=2/180` → GRPO 采样几乎全是零优势，无法进一步优化。

**这解释了 2026-07-16 那次 GRPO 被 support gate 主动跳过的根因**：数据的判别信号被规则字段直接给出，没有"需要探索"的动作分布。

## 3. 新数据集设计：`planner_retry_migrate_v7_longobs`

### 3.1 目标（可验证的成功条件）

1. **观测长度**：每条 case 的**平均 tool observation 长度 ≥ 1500 tokens**，含 detector JSON 输出（bbox + confidence + tags）、上一次 detector 的历史 trace（用于识别 stale）、迁移顾问需要的资产片段（RAG 检索命中若干段规范）。整个 planner prompt 目标平均 ≥ 8000 tokens、p95 ≥ 12000 tokens。
2. **规则字段隐藏**：`retryable / retry_count / gateway_error / domain_shift / candidate_count / min_confidence / cross_prompt_iou` 都**不出现在 observation summary**，模型必须从 observation 里**导出**它们。例如：`retryable` 通过错误码类型识别（timeout/overload → 可重试；auth_failed/quota_exhausted → 不可重试）；`candidate_count` 通过数 bbox 数组长度；`cross_prompt_iou` 通过对比两次结构化返回。
3. **user_query 干净**：只描述业务目标（要检测什么、下游想拿什么结论），**不能再复述规则**。规则只在 system prompt 里说明一次。
4. **判别路径丰富**：核心状态从 3 支扩到 **8 支**（见 §3.3），并加入**多步依赖**（step 3 决策依赖 step 2 输出，不是从 setup 直接可读）。
5. **验证式训练价值**：SFT 后在新 grpo_dev 上 **`nonzero_reward_variance_rate ≥ 15%`**（v6 是 1.1%），确保 GRPO 有采样空间；这是 support gate 的 preregistered 判据。

### 3.2 目标能达成什么（预注册的量化目标）

以 `unique-decision action_match_rate` 与 `full_trajectory_match` 为主指标，`mean_rule_reward` 为辅（都按 counterfactual_bundle_id 做 case-macro，2000 bootstrap 出 95% CI）：

| Arm | unique_action_match | full_trajectory_match |
|---|---:|---:|
| 4B base | ≥ 0.35（下限锚点） | ≥ 0.15 |
| 4B + SFT | ≥ 0.80，且 **> base+15pp 95% CI 排除 0** | ≥ 0.60 |
| 4B + SFT + GRPO ×3 seeds | ≥ 0.90，且 case-macro paired 95% CI 相对 SFT 排除 0，同时 **wrong side-effecting actions ≤ SFT** | ≥ 0.75 |
| **4B + SFT + GRPO vs 35B base** | ≥ 35B base，95% CI 下界 ≥ 0 | 同 |

35B base 预期 ~0.55–0.65（比 v6 的 0.52 略高但仍受限于没训练过软边界）——4B SFT+GRPO 越过 35B base 是明确目标。

### 3.3 状态空间：从 3 支扩到 8 支

以 `(observation_class, retry_budget)` 双维展开，前 6 支是核心 primary，后 2 支是 guardrail（防止训练学成"检测后必迁移"）：

| # | Observation class | Retry budget | 期望动作 | 学习难点 |
|---|---|---|---|---|
| P1 | detector 成功但 `iou_between_prompts < 0.72` | 0 | 再重试同 detector 一次 | 需要对比两个 observation 内 bbox 集合，非规则字段 |
| P2 | detector 成功且四项 gate 均过 | 0 | end（不迁移） | 需要在 detector JSON 里数 candidate 数并读 confidence 分布 |
| P3 | detector timeout / gateway 5xx | 0 | 重试同 detector | 需要从 `error_message` 分类 retryable |
| P4 | detector 返回 `auth_failed` / `quota_exhausted` | 0 | migration_advisor | 同上，属于非可重试大类 |
| P5 | detector 二次返回仍不达标 | ≥1 | migration_advisor | 需要看历史 trace 识别"已经试过一次" |
| P6 | detector 成功但 `domain_shift=high`（隐含在 tags 里） | 0 | migration_advisor | 需要从 tags 集合推断域偏移 |
| G1 | 首步 detector 一次就完美达标 | 0 | end | 防止模型学成"看到 detector 结果一律 migrate" |
| G2 | 冲突历史（上一 query 说成功、本次 observation 说失败） | 0 | migration_advisor | 长上下文里跨轮次一致性判断 |

每支各 60 counterfactual bundle × 3 nuisance rotation（badge / detector family / entity）= **每支 180 case，总 1440 case**；按 4:1:1 分成 train / dev / sealed_test（960 / 240 / 240）。

### 3.4 真实 observation 的构造方法（关键难点）

不能真的调外部 detector 服务（成本 + 稳定性），但也不能像 v6 那样纯字符串拼。折衷：**从公开数据集 + 规则模板生成 detector 的 JSON 输出**，同时**保留自然语言噪声**。

- **视觉源**：`COCO 2017 val` (5000 图) + `Objects365 val` (~30k 图)，选出至少含目标类别的 3000 图。目标类别按 v6 的 fixture family 映射（"绿色弧顶隔离柱"→ COCO `traffic_light` 系列等，或用 GroundingDINO 类别对齐）。
- **detector 输出模板**：模拟 Qwen-VL / RexOmni 的 OpenAI-兼容响应 —— `choices[0].message.content = <JSON: {objects: [...], meta: {...}}>` + 20-50 行系统 telemetry（版本号、耗时、queue length、warning 消息、模型 fingerprint）。
- **retry_budget 的隐式化**：不写 `retry_count=1`，而是给出**上一步的 detector response 完整 dump**（500-800 tokens），加上时间戳；模型需要从"这是同一个 detector 的第二次调用响应"推导出预算耗尽。
- **stale history**：guard G2 类，`session_history` 字段塞 3-5 条历史 query 与其 observation，其中最后一条**主动矛盾**当前 observation（例如上一 query 说 "已成功"、本次 observation 是 timeout）。
- **迁移顾问需要的 RAG 命中**：迁移分支的 observation 里包含**从公开产品文档 chunk 出的 3-5 段规范描述**（可以直接用 `sources/` 下的中文技术文档），每段 200-400 tokens。这样 `migration_advisor` 的正确性也能被 label 检查。
- **长度控制**：每 case 从上述元件按预设 recipe 拼接，用 tokenizer 计数保证总长在 [7000, 13000] tokens，超出的随机 drop telemetry chunk。

**审计**：数据集构建脚本（沿用 `build_planner_retry_migrate_v6.py` 的 `write_json/audit_report/eda_summary` 结构）额外做：
- observation 里**不允许**出现 `retryable=` / `retry_count=` / `gateway_error=` / `domain_shift=` / `candidate_count=` / `min_confidence=` / `cross_prompt_iou=` 任一子串（gate=hard fail）。
- prompt 长度 5th / 50th / 95th percentile 分位 + `over_limit=0`。
- badge × action 与 error_alias × action 的互信息 ≤ 0.01（entity-disjoint，禁 shortcut）。
- 分 split 计算 fixture 内容 SHA256，零交叉。

### 3.5 数据集实体隔离与 leakage 控制

- 250 个 entity（沿用 v6 命名生成器），**train 80 / sft_dev 20 / grpo_train 60 / grpo_dev 30 / test 60**，entity-disjoint，测试集 sealed。
- entity_id / case_id / normalized_query / template_id / fixture_family / fixture_sha256 六字段跨 split 零重叠。
- 与本仓库现存所有 planner cases 做**全量 SHA256 与 fixture path 交叉检查**（builder 已实现，直接复用）。
- COCO/Objects365 图片只用**图内容 SHA256 判重**，避免 URL 同、内容不同的情况；进入仓库的图片全部本地化到 `examples/images/planner_retry_migrate_v7/`，避免运行时依赖网络。

### 3.6 评测指标（用户强调"正确普适的计算方式，不要编造"）

**沿用仓库既有 `train_planner_grpo` 打分器**，不新造。主指标严格按以下路径产生：

1. **`action_match_rate`**：`predicted_action == gold_action`（gold 来自 `expected_decisions[step].action`）；决策位级、按 case_id 分组再按 counterfactual_bundle_id case-macro 一次。
2. **`mean_rule_reward`**：`reward_planner_grpo.py` 里定义的组合分（`action + required_args + arg_contains + forbidden_actions + finish_after_tool`），0-1；同上聚合。
3. **`full_trajectory_match`**：一条完整轨迹的每一步都 pass action_match 且 finish 契合。
4. **`nonzero_reward_variance_rate`**：在 grpo_dev 上给每个 prompt 采 4 samples，reward std > 0 的比例；GRPO support gate 用。
5. **95% CI**：`counterfactual_bundle_id` 分组、2000 次 bootstrap paired（baseline vs candidate）；下界 > 0 才算显著。
6. **副作用监控**：`wrong_side_effecting_actions_mean`（错误的 `flux-image-generation` / `pipeline_eval` / `qwen_detection with side effect flag` 数量），H20 上仍按 V100 门规则不许上升。

### 3.7 计算与人力预算（H20 4 卡）

| 阶段 | 内容 | 时长（估） |
|---|---|---|
| 数据构造 | builder + audit + 300 case 独立人工抽检 20% | 6-10 gpu·h（tokenizer 计数、图片 SHA256）+ 1-2 人日 |
| SFT | 4B LoRA, 4×H20 bf16, 600 step | ~1.5 gpu·h |
| SFT eval | 三场景 × 3 runs | ~0.3 gpu·h |
| GRPO ×3 seeds | 100 step / seed, 采样 4 gen/prompt | ~6 gpu·h |
| GRPO eval + compare + gate | | ~1 gpu·h |
| 35B base 三场景对照 | 已有，重跑一次以对齐 v7 dev cases | ~1 gpu·h |
| **合计** | | **≈10 gpu·h + 1-2 人日** |

## 4. 与目标的对齐说明

用户目标：**`4B base < 4B SFT < 4B SFT+GRPO ≥ 35B base`**。

- `4B base < 4B SFT`：v6 上已实证（0.7731 → 0.9769 action_match，89.8% 相对误差减少）；新数据只要长 observation 能给出足够可学习信号，SFT 一定能拉高一档，且 §3.2 已把它设为门。
- `4B SFT < 4B SFT+GRPO`：**v6 上失败**（GRPO 采样零优势）。新数据 §3.3 的 8 支 + §3.4 长 observation 让判别信号从"读字段"变成"读上下文推断"，SFT 在 dev 上不会瞬间打满；GRPO 有采样空间。§3.6 的 `nonzero_reward_variance_rate ≥ 15%` 是可测门。
- `4B SFT+GRPO ≥ 35B base`：因为 35B 没在软边界上训练过，v6 上它就与 4B base 并列（0.516 vs 0.513）；新数据里 8 支中至少 4 支需要**规则内化**（P3/P4 的错误类别推断、P5 的历史 trace 识别、G2 的跨轮一致性），这些是 SFT+GRPO 能明确学会、而 35B zero-shot 靠 in-context 推理很难稳定过 0.8 的类别。

## 5. 未纳入范围（避免范围蠕变）

- **不做**多模态 planner。仍是"文本 planner + 结构化 detector JSON 塞进 observation"，与 v6 保持接口一致（`CAPA_OMIT_MODEL_IMAGE_PAYLOAD=1`）。
- **不做**真实工具执行；observation 是 mock，但结构与真检测服务的 JSON 契约一致。
- **不做** GRPO reward model；仍用规则打分器（可复现、可审计）。
- **不改** v6 的 sealed test。它作为独立回归集保留，v7 训练不允许拿它做选择。

## 6. 落地前需用户确认的三点

1. **视觉源**：使用 COCO/Objects365 生成 mock detector 输出可接受吗？若限公开数据合规问题，可回退到"完全合成的 bbox + tag 集合"（数据可控但少了自然语言噪声）。
2. **迁移顾问 RAG 片段来源**：是否可用仓库 `sources/` 下现有中文技术文档？若不方便，用 GPL 中文 wiki 段落做占位。
3. **35B base 是否重跑一次**：目前的 35B base 数字是在 v6 grpo_dev 上跑的，v7 上限需要在 v7 dev 上重新跑一版 base 才能公平对齐 —— 计入 §3.7 的 1 gpu·h 预算，是否 OK？

—— 以上为方案，不启动训练；等确认后再落地 `build_planner_retry_migrate_v7_longobs.py` 与相应 study 目录。
