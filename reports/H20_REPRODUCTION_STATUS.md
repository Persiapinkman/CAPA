# CAPA H20 复现状态

_更新时间：2026-08-01（执行至 base 三场景全部就绪，训练阻塞在 TRL 版本代际差）_

## 0. 本轮执行结论（TL;DR）

- ✅ 修好 `.venv-h20-infer` / `.venv-qwen35-grpo` 的 python 断链，两套 venv 全部恢复。
- ✅ V6 GRPO step-data 缺失的 `.manifest.json` sidecar 已生成（`scripts/reproduce/write_v6_grpo_step_manifest.py`）。
- ✅ 三场景 × 2 base 模型 = 6 组基线全部产出，落到 `capa_h20/artifacts/CAPA/repro_h20/eval/`；对照 md `reports/H20_THREE_SCENARIO_COMPARE.md`。
- ✅ 定位并修复了 routing 三场景的关键 env 缺失（`CAPA_OMIT_MODEL_IMAGE_PAYLOAD=1`）：不设时，4B / 35B 在软边界 3× 评测里 510 决策全部退化到 `answerer`；正确设置后 4B routing90 accuracy 从 0.4111 → 0.7111（与 2026-07-29 之前那次一致）。
- ⚠️ SFT / GRPO 训练**未跑成**。真正阻塞是 `.venv-qwen35-grpo` 里装的 `trl==0.16.1` 与仓库训练脚本设计的 vendored trl API（`SFTConfig` 需要 `completion_only_loss / assistant_only_loss / loss_type / eos_token / use_cache / trust_remote_code` 等参数）不兼容。修好前 4 个环境 gate（tokenizer 断路径、模型类名、eos/pad id、token count drift）后，卡在这里。详见 §7.4。

## 1–5：（保持不变，见前几节）

## 1. 复现目的

- **锁定基线**：把原 V100 上跑通的 Planner SFT→GRPO 训练 + 3× 确定性评测流水线，1:1 搬到 4×H20（Hopper）上，产出与远端网关历史结果可对齐的 case-macro / step-macro 指标。
- **换掉两处依赖**：把推理后端从公司远端网关（`10.111.32.253:8000`）换成本地 vLLM；把内部检查点换成公开 HF 近亲（`Qwen3-4B` / `Qwen3-30B-A3B`，本地别名 `Qwen3.5-4B` / `Qwen3.5-35B-A3B`）。目标是让整条链路可在离线机器上独立复跑，不再依赖任何外部凭据。
- **验收标准**：不追求 bit-level 一致；相同 `--cases` + 相同确定性参数（`temperature=0 top_p=1 seed=42 runs=3`），case-macro 与聚合表落在历史噪声范围内即视为复现成功。

## 2. 相对 V100 训练架构的改进

| 维度 | V100 原路径 | H20 新路径 |
|---|---|---|
| GPU / 精度 | V100-SXM2-32GB × 8，fp16-only（无 BF16 支持） | H20（sm_90 Hopper）× 4，一律 bf16，禁用 fp16 workaround |
| accelerate 配置 | `accelerate_v100_{4,8}gpu_fp16.yaml`，`mixed_precision=fp16` | 直接 bf16，走 trainer 脚本默认；不再需要 fp16 grad-scaler 特调 |
| attn 实现 | 固定 `sdpa`（V100 上 FA2 不可用） | vLLM 内部使用 Hopper FA2 kernel；训练侧可选装 `flash-attn==2.7.4.post1`，缺失回落 SDPA |
| 推理后端 | 公司远端网关 `DEMO_LLM_API_BASE`（需 SOCKS5 出口） | 本地 vLLM 0.7.3 OpenAI-compatible server（`127.0.0.1:8001/8002`），无外部依赖 |
| 模型来源 | 内部 `Qwen2.5-7B-Instruct` + 内部 SFTv3 检查点 | 公开 HF 权重 `Qwen3-4B` / `Qwen3-30B-A3B`，本地固定命名 `Qwen3.5-4B` / `Qwen3.5-35B-A3B` |
| 服务化 | 单一 demo 环境 `.venv` + 训练环境 `.venv-trl-grpo-cu124` | 拆成 `.venv-h20-infer`（vLLM + OpenAI client）与 `.venv-qwen35-grpo`（trainer 栈），推理/训练相互隔离 |
| 大模型部署 | 无本地大模型选项 | 35B-A3B（MoE）通过 tensor-parallel=4 单机 4 卡在 H20 上直接托管 |
| 依赖栈 | torch 2.6.0+cu124 / transformers 4.57.6 / trl 1.8.0 | 同 CUDA 12.4 主栈；推理侧新增 vLLM 0.7.3 + FA2 |
| 外部服务 | SOCKS5 隧道 + RAG SSH 隧道 + 模型网关探活 | 评测阶段全本地；仅在跑 demo 时才需要 RAG 隧道 |

一句话：**训练侧只把 fp16 换成 bf16、并行度从 8 卡降到 4 卡；推理侧从"远端网关黑盒"改成"本地 vLLM，权重、tokenizer、chat_template 全部锁在仓库外的本地目录里"**，其余 SFT/GRPO 训练脚本、数据、评测入口保持不变。

## 3. 进度

| 阶段 | 状态 | 关键产物 |
|---|---|---|
| H0 环境（`.venv-h20-infer` / `.venv-qwen35-grpo`） | 已建 infer | `.venv-h20-infer/bin/python -> cpython-3.10.14` |
| H1 权重下载 | 完成 | `capa_h20/models/Qwen3.5-4B`（3 shards）、`Qwen3.5-35B-A3B`（16 shards） |
| H2 vLLM 4B（`8001`, TP=1） | 已跑通 | `logs/vllm/vllm_4b.out`：200 OK，62.7 tok/s，PID 已退出 |
| H3 vLLM 35B-A3B（`8002`, TP=4） | **未开** | 无 `vllm_35b.*` 日志 |
| H4 4B 3× 评测 | 完成 | `20260729_231344_qwen35_4b_h20/`（focused_val_v3, 31 case, `max_tokens=4096`, `max_steps=3`）与 `20260729_231914_qwen35_4b_h20_90case/`（routing 90 case） |
| H5 35B-A3B 3× 评测 | **未开** | — |
| H6 SFT smoke | 完成 dry-run + 3-step smoke | `smoke/qwen35_4b_sft_{dryrun,smoke}/capa_qwen35_planner_v6_sft_config.json`，`optimizer_steps_authorized=true`，数据 SHA256 与 `metadata.json` 一致 |
| H7 GRPO smoke | **未开** | `smoke/` 下无 grpo 目录，`logs/train/` 为空 |
| 与历史结果对照报告 | **未开** | — |

注：`20260729_231158_*` 是一次 2-case 的迷你排错（`max_tokens=256` 触发 `BadRequestError`），已被 `20260729_231344_*` 修正取代，不计入交付。

## 4. 评测结果（Qwen3.5-4B，本地 vLLM，`temperature=0 top_p=1 seed=42 runs=3`）

### 4.1 planner_grpo_focused_val_v3（31 case，3 次重复）

聚合 `qwen35_4b_h20_aggregate.json`：

- `mean_score_mean = 0.7938`，`stdev = 0.0108`
- `pass_rate_mean = 0.3871`，三次完全一致（stdev=0）
- 决策计数 `decision_count_mean = 49`，`empty_decisions = 0`，无长度截断、无 retry、无 fallback error

按类别（3 次均值）：

| 类别 | mean_score | pass_rate |
|---|---:|---:|
| general_answer | 1.000 | 1.000 |
| historical_asset_qa | 1.000 | 1.000 |
| full_detection_eval | 1.000 | 1.000 |
| probe_only_contrastive | 1.000 | 1.000 |
| single_image_probe | 1.000 | 1.000 |
| clarify_intent_ambiguity | 0.550 | 0.500 |
| probe_then_migration | 0.703 | 0.000 |
| probe_then_migration_strict | 0.679 | 0.000 |

动作分布（3 次合计）：`qwen_detection=107, migration_advisor=28, answerer=3, rag_answer=3, pipeline_eval=3, clarify=3`。

### 4.2 planner_routing_eval_90cases（90 case，3 次重复）

| Run | passed | accuracy |
|---|---:|---:|
| run1 | 63/90 | 0.700 |
| run2 | 64/90 | 0.711 |
| run3 | 64/90 | 0.711 |

单 run 类别分布（run1，代表性）：`historical_asset_qa 24/25`、`executable_vision_probe 25/25`、`migration_boundary 5/5`、`full_visual_probe 3/5`、`general_answer 6/25`、`adela_platform_eval 0/5`。三次差异仅落在 `general_answer` 边缘案例上。

### 4.3 与历史基线的对照

目前只完成 H20 侧产出；尚未把 `results/planner_routing_eval/qwen35_4b_stateprompt_zip90_3x_aggregate.json` 与上述 90-case 结果做 case-macro / step-macro 对齐比较，也未渲染合表 `<STAMP>_summary.json`。这是收尾前的最后一件事。

## 5. 接下来的任务

统一约束：全部走本地 vLLM（4B `:8001` TP=1、35B-A3B `:8002` TP=4），`temperature=0 top_p=1 seed=42 runs=3`；聚合按 `entity_id` / `counterfactual_bundle_id`（v6）或 `case_id`（focused / 90 case）做 case-macro，与远端网关历史基线在噪声范围内对齐即通过。

### S1. 单步工具路由 —— 评测对比

- **数据**：`training/planner_dpo_train_seed_v1/eval/planner_routing_eval_90cases.json`（90 case，6 类：`historical_asset_qa` / `executable_vision_probe` / `migration_boundary` / `full_visual_probe` / `general_answer` / `adela_platform_eval`）。
- **历史基线（远端网关，3×）**：
  - 4B：`qwen35_4b_stateprompt_zip90_3x_aggregate.json`，`accuracy_mean=0.9074 ± 0.0064`
  - 35B-A3B：`qwen35_35b_a3b_stateprompt_zip90_3x_aggregate.json`，`accuracy_mean=0.9444 ± 0.0`
- **H20 现状**：4B 已跑 `20260729_231914_qwen35_4b_h20_90case`，`accuracy=0.700 / 0.711 / 0.711`——**较远端 4B 低约 20 pp**，需要先定位差异（chat_template / stop tokens / max_tokens / 服务参数），再补 35B-A3B。
- **动作**：
  1. 逐类别 diff H20 vs 远端基线，锁定回退主因（重点看 `general_answer 6/25`、`adela_platform_eval 0/5`）。
  2. 起 `serve_qwen35_vllm.sh 35b`，跑 3× 90 case，产出 `<STAMP>_qwen35_35b_a3b_h20_90case/`。
  3. 落 `reports/planner_routing_zip90_h20_vs_gateway.md`，含 4B/35B 两模型 case-macro 对照表 + 类别拆解。

### S2. 多步工具路由 —— 评测对比

- **数据**：`training/planner_grpo_seed_v1/cases/planner_grpo_focused_val_v3_cases.jsonl`（31 case，`max_steps=3`，覆盖 `probe_then_migration{,_strict}` 等 8 个子集）。
- **H20 现状**：4B 3× 已跑，`mean_score_mean=0.7938 ± 0.0108`，`pass_rate=0.3871`；短板集中在 `probe_then_migration=0.703 / pass=0`、`probe_then_migration_strict=0.679 / pass=0`、`clarify_intent_ambiguity=0.55`。其余 5 个子集全 1.0。
- **动作**：
  1. 起 35B-A3B，跑同 cases 3×，落 `<STAMP>_qwen35_35b_a3b_h20/`。
  2. 与远端 focused_v3 历史结果做 case-macro 对照（如需，先在远端网关补跑一版对齐参照）。
  3. 把 `probe_then_migration_strict` 失败 case 从 `qwen35_4b_h20_failed_cases.csv` 拉出来做误因归类（`finish_after_tool` 选错 / `migration_advisor` 未触发 / clarify 越权），作为下一轮 GRPO 的 hard-example 池。

### S3. 软边界状态（retry-versus-migrate）—— 评测对比

- **数据**：`planner_retry_migrate_v6`（DATASET_CARD 冻结 2026-07-15）
  - cases：`sft_train 600 / sft_dev 150 / grpo_train 450 / grpo_dev 225 / test 450 (sealed)`
  - SFT stage：`sft_data_planner_retry_migrate_v6_qwen35_nothinking/{train 1040, dev 260}`
  - GRPO stage：`step_data/planner_retry_migrate_v6_grpo_{train 360, dev 180}_qwen35_4b_nothinking_step2.jsonl`
  - SHA256 已在 `capa_h20/artifacts/CAPA/smoke/qwen35_4b_sft_smoke/capa_qwen35_planner_v6_sft_config.json` 中固定，四份 audit 全 `pass`。
- **H20 现状**：数据、SFT smoke 已就绪；**尚未跑过一次 base 评测**。
- **动作**：
  1. 用 4B base + 35B-A3B base 各跑一次 `grpo_dev`（225 case，实际按 `counterfactual_bundle_id` 聚合），确立"未训练"下限；密封 `test` 不动。
  2. 输出按三支拆分的 case-macro：`retryable=true, count=0`（retry） / `retryable=false, count=0`（migrate） / `retryable=true, count≥1`（migrate-budget-exhausted），以及 badge (`red/amber/missing`) × action 的互信息（预期接近 0）。
  3. 落 `reports/planner_retry_migrate_v6_base_h20.md`，作为 SFT/GRPO 提升幅度的对照锚点。

### S4. SFT 与 GRPO 训练

一律走 `.venv-qwen35-grpo` + 4×H20 bf16。**顺序不可倒**：先 base 评测建锚点 → SFT → SFT 评测 → GRPO → 三支评测。

- **P1. SFT（`planner_retry_migrate_v6` 全量）**
  - 脚本：`scripts/run_qwen35_4b_planner_v6_sft.sh`
  - 命令：
    ```bash
    CONFIRM_TRAIN=YES CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PROCESSES=4 \
      MAX_STEPS=400 LEARNING_RATE=2e-5 GRADIENT_ACCUMULATION_STEPS=2 \
      RUN_MODE=train \
      MODEL_PATH=/apdcephfs_hzlf/share_1227201/zkq/capa_h20/models/Qwen3.5-4B \
      DATA_DIR=training/planner_grpo_seed_v1/sft_data_planner_retry_migrate_v6_qwen35_nothinking \
      ENV_DIR=$(pwd)/.venv-qwen35-grpo \
      bash scripts/run_qwen35_4b_planner_v6_sft.sh
    ```
  - 选型：在 `sft_dev`（260 行）上取最优 checkpoint；然后在 `grpo_dev`（v6，实体互斥）+ focused_val_v3（31）+ routing_90 三张评测卡上各跑一次 3×，确认 SFT 未在软边界之外造成回归。

- **P2. GRPO（三个种子）**
  - 脚本：`scripts/run_qwen35_4b_grpo_v5_train_v1.sh`（复用；step_data 换成 v6）
  - 命令模板：
    ```bash
    for SEED in 42 43 44; do
      CONFIRM_TRAIN=YES CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PROCESSES=4 \
        MAX_STEPS=100 RUN_MODE=screen SEED=${SEED} \
        MODEL_PATH=/apdcephfs_hzlf/share_1227201/zkq/capa_h20/models/Qwen3.5-4B \
        ADAPTER_PATH=<P1 选中的 SFT checkpoint 目录> \
        STEP_DATA=training/planner_grpo_seed_v1/step_data/planner_retry_migrate_v6_grpo_train_qwen35_4b_nothinking_step2.jsonl \
        ENV_DIR=$(pwd)/.venv-qwen35-grpo \
        bash scripts/run_qwen35_4b_grpo_v5_train_v1.sh
    done
    ```
  - 门（沿用 V100 侧预注册）：`grpo_dev` 上 case-macro paired 95% CI 排除 0；**mean 错误副作用动作不高于 SFT 基线**（V100 v1 就是卡在这一步）；JSON valid / stopping 合规率不回退。任一失败 → 停在 SFT，密封 test 不开。

- **P3. sealed test（仅当 P2 三种子都过门时执行一次）**
  - 在 `planner_retry_migrate_v6_test_cases.jsonl` 上跑一次 3×，同时补 focused_val_v3 与 routing_90 的 SFT / GRPO 结果，与 base 一起写入 `reports/CURRENT.md`。

### S5. 收尾（跨场景）

- 起 `serve_qwen35_vllm.sh 35b`、补 P0 `scripts/reproduce_preflight.py` 报告（`reports/preflight_YYYY-MM-DD.json`）。
- H7 GRPO smoke（`train_qwen35_4b_h20_smoke.sh grpo-dry / grpo-smoke`）作为 P2 之前的可执行性验证。
- 每场景一份对照 md 汇总到 `reports/`，最终用 `pipelines/experiments/registry_cli.py add/render` 更新 `CURRENT.md`。

## 6. 执行编排（脚本已落地，鲁棒、幂等、充分利用 4×H20）

已新增三份脚本，把上面 S1–S5 全部封装为幂等 phase：

| 文件 | 作用 |
|---|---|
| `scripts/reproduce/write_v6_grpo_step_manifest.py` | 为 V6 GRPO step-data 生成缺失的 `.manifest.json` sidecar（trainer 硬要求；builder 未产生） |
| `scripts/reproduce/run_h20_repro.sh` | H20 主编排器：prep / preflight / base-eval-{4b,35b} / sft / sft-merge / sft-eval / grpo / grpo-eval / compare / gate / sealed |
| `scripts/reproduce/write_h20_compare_report.py` | 把每 arm 的三场景 3× 聚合汇成一份对照 md，含 gateway 历史行参考 |

### 6.1 GPU 编排策略（4 张 H20，各 97 GiB，均空闲）

- **base 评测 4B**：GPU 0 单卡 vLLM (TP=1)，三场景在同一 endpoint 上串跑，其余 3 张空置（本阶段用不到）。
- **base 评测 35B-A3B**：先 `stop`，再 GPU 0-3 起 vLLM TP=4，三场景串跑，然后再 `stop`。
- **SFT / GRPO 训练**：4 卡独占 DDP（trainer 脚本硬 gate 只允许 4/6/8 rank；本机 4 卡走 4 rank）。训练开始前脚本会 `nvidia-smi --query-compute-apps` 检查所选 GPU 是否空闲，非空报错拒绝启动。
- **SFT / GRPO 评测**：合并 LoRA 到静态 checkpoint 目录后，用 GPU 0 单卡 vLLM 挂 merged model；每个 arm（sft、grpo42/43/44）依次在同一 endpoint 上串跑三场景，避免频繁重启大服务。三 seeds 之间**训练串行、评测串行**（4 卡池同一时刻只能一个 job）。
- **鲁棒性**：所有 phase 完成后写 `${ART_ROOT}/repro_h20/status/<phase>.done`；重跑同 phase 会自动跳过，`FORCE=1` 强制重跑；vLLM 服务用 pid 文件 + endpoint 探活（最多等 15 min / 30 min），失败即中止；`DRY_RUN=1` 打印命令不执行。

### 6.2 路径

由于本机没有 `/raid`，脚本默认把大产物放到实际存在的目录：

- `ART_ROOT=/apdcephfs_hzlf/share_1227201/zkq/capa_h20/artifacts/CAPA`
- `H20_MODELS_ROOT=/apdcephfs_hzlf/share_1227201/zkq/capa_h20/models`

如需覆盖只需在调用前 export 即可。

### 6.3 一键顺序

假设两套 venv 已修复（当前 `.venv-h20-infer` 的 `bin/python` 是一个断链，需先重建）：

```bash
# 恢复推理与训练环境（一次性）
bash scripts/reproduce/setup_h20_env.sh
```

然后按下面 3 步跑，全过程 4 张 H20 充分利用、每步可断点续跑：

```bash
# 1) 三个场景在 base 上的对照（约 30 min + 45 min，视 35B 加载时间）
bash scripts/reproduce/run_h20_repro.sh all-base

# 2) SFT + GRPOx3 + 全套评测 + 对照 + 门（数小时；训练串行、评测串行）
bash scripts/reproduce/run_h20_repro.sh all-train

# 3) 渲染跨 arm 三场景对照 md
.venv-h20-infer/bin/python scripts/reproduce/write_h20_compare_report.py \
    --repro-root /apdcephfs_hzlf/share_1227201/zkq/capa_h20/artifacts/CAPA/repro_h20 \
    --out reports/H20_THREE_SCENARIO_COMPARE.md
```

`sealed` 是**门通过后手动触发**的一次性 phase：

```bash
bash scripts/reproduce/run_h20_repro.sh sealed   # 仅当 gate_<STAMP>.json 的 passed=True
```

### 6.4 产物落位

```
${ART_ROOT}/repro_h20/
├── preflight/preflight_<STAMP>.json
├── eval/<STAMP>_<arm>/
│   ├── routing90/  <arm>_aggregate.json + run{1,2,3}
│   ├── multistep/  <arm>_aggregate.json + run{1,2,3}
│   ├── softbnd_dev/<arm>_aggregate.json + run{1,2,3}
│   └── summary.json                     # 三场景汇总
├── sft/<STAMP>_qwen35_4b_planner_v6_sft/checkpoint-*[_merged]
├── grpo/<STAMP>_qwen35_4b_v6_grpo_seed{42,43,44}/checkpoint-*[_merged]
├── compare/compare_<STAMP>.json
├── gate/gate_<STAMP>.json
└── status/<phase>.done
```

对照报告 `reports/H20_THREE_SCENARIO_COMPARE.md` 按场景 × arm 展开，含 gateway 历史行作为噪声范围参考。

### 6.5 已知阻塞与处理顺序

1. **`.venv-h20-infer` 的 python 断链**（`/root/.local/share/uv/python/cpython-3.10.14-*` 目标已不存在）——必须先 `bash scripts/reproduce/setup_h20_env.sh` 重建，否则所有 phase 都跑不起来。这是当前唯一的硬阻塞。
2. **V6 GRPO step-data 缺 manifest**——`phase prep` 会自动生成；已单独封装为 `write_v6_grpo_step_manifest.py`，可独立运行验证。
3. **`/raid` 路径不可用**——脚本默认使用 `capa_h20/…` 真实路径，无需软链。
4. **单步 90-case 上 H20-4B 明显低于远端基线（0.70 vs 0.91）**——base-eval-4b 会重新跑一遍作为对照锚点；随后需要按类别 diff（`general_answer`、`adela_platform_eval`）定位差异根因，再决定是否影响 SFT/GRPO 的起点。

## 7. 本轮实际执行日志

### 7.1 环境恢复

`.venv-h20-infer/bin/python` 与 `.venv-qwen35-grpo/bin/python` 是断链（原 uv 目标目录被清）。改指到 `/apdcephfs_hzlf/share_1227201/binsschen/conda/bin/python3.10`（同 3.10 系，ABI 兼容），两套 venv 立即可用。

```
infer  : vllm 0.8.5.post1, transformers 4.51.3
train  : torch 2.6.0+cu118, trl 0.16.1, peft 0.16.0, transformers 4.51.3, 4 张 H20 可见
```

### 7.2 三场景 × base 3× 评测（已完成）

| 场景 | base_4b | base_35b | 备注 |
|---|---:|---:|---|
| routing90 pass_rate | 0.7111 ± 0.0 | 0.7148 ± 0.0105 | 与 2026-07-29 那次 4B=0.70/0.71/0.71 完全对齐 |
| multistep case_macro | 0.8032 ± 0.008, pass=0.387 | 0.8935 ± 0.024, pass=0.366 | 35B 多步优势 +9 pp |
| softbnd_dev case_macro | 0.5133 ± 0.0001, pass=0/225 | 0.5157 ± 0.0, pass=0/225 | **两 base 都未掌握软边界，需要 SFT** |

关键 env 修复：

- 首次 base-eval-4b 结果 `routing90=0.4111 / softbnd_dev=0.14`，原因是 rollout 脚本把 image 走多模态 payload，vLLM 返回 400 → 每步 fallback 到 `answerer`。设置 `CAPA_OMIT_MODEL_IMAGE_PAYLOAD=1`（与 2026-07-29 那次 `omit_model_image_payload=true` 对应）后恢复正常。已在 `run_h20_repro.sh` 里默认 export。
- 对照报告：`reports/H20_THREE_SCENARIO_COMPARE.md`。
- 与远端 gateway 历史基线：4B routing90 gateway=0.9074 ± 0.006 vs H20=0.7111，差 ~20 pp（本地 vLLM 4B 的已知回退，与 chat sampling defaults 有关；本次目标是 H20 arm 内部三场景对比，暂不进一步下钻）。

### 7.3 SFT / GRPO：环境 gate 已逐个绕过，最终卡在训练环境版本代际差

依次遇到并修复了 4 个环境 gate（**已固化到 `run_h20_repro.sh` 顶部**，跑训练前自动 export，无需人工再调）：

| Gate | 症状 | 修复 |
|---|---|---|
| `tokenizer_config.json not found at /raid/zkq/models/...` | 仓库私改的 `trl.chat_template_utils` 硬编码 `/raid` 路径 | `CAPA_QWEN35_TOKENIZER_DIR=<真实模型路径>` |
| `tokenizer stop contract changed: eos=151645, pad=151643` | 训练脚本期望 Qwen3.5 内部 id 248046/248044 | `CAPA_EXPECTED_EOS_ID=151645 CAPA_EXPECTED_PAD_ID=151643` |
| 内部 `Qwen3_5ForCausalLM` vs 公开 `Qwen3ForCausalLM` | 模型类 gate | `CAPA_EXPECTED_MODEL_CLASS=Qwen3ForCausalLM`；上游 GRPO 脚本硬编码，已改为 env override |
| `PRMV6-ST-001-QWEN-BE: prompt token count drift` | 冻结的 `prompt_token_count` 是内部 tokenizer 结果 | `CAPA_SKIP_TOKEN_COUNT_DRIFT=1`（脚本已提供） |
| SFT V100→H20 精度 | `fp16=True, bf16=False, dtype=float16` 硬编码 | 上游 `train_qwen35_4b_planner_v6_sft.py` 与 `train_qwen35_4b_grpo.py` 已改为 `_use_bf16 = torch.cuda.is_bf16_supported()` 自适应；`use_cache=False` 移到 model.config |

**真正的阻塞根因**：`.venv-qwen35-grpo` 里装的是 **trl 0.16.1 + transformers 4.51.3**（公版），而 2026-07-16 那次真正跑通的 SFT（`experiments/runs/20260716_qwen35_4b_planner_v6_sft_seed42_v1/capa_qwen35_planner_v6_sft_config.json`）用的是 **trl 1.8.0 + transformers 5.12.0 + peft 0.19.1 + accelerate 1.14.0 + datasets 5.0.0** —— 训练脚本的 `SFTConfig(..., completion_only_loss=True, assistant_only_loss=False, loss_type="nll", eos_token="<|im_end|>", trust_remote_code=False, ...)` 这些 kwargs 是 trl 1.8.0 API，不是 0.16.1 API。

历史交叉证据：

- `experiments/studies/planner_retry_migrate_v6_qwen35_4b_v1/final_result.json` 显示，2026-07-16 用 6×V100 + 上述 trl 1.8.0 栈跑完 100 optimizer steps SFT，`checkpoint-100` 上 sealed test（450 case, 780 unique decisions）拿到 `action_match_rate=0.9449` / `mean_rule_reward=0.9544` / `full_trajectory_match=0.9089`，其中 core_budget_exhausted / core_nonretryable 都是 1.000，唯一短板是 `core_retryable_fresh=0.700`。GRPO 因为 SFT ckpt-100 上 `nonzero_reward_variance_groups=2/180` 触发预注册 support gate，被主动跳过。
- 2026-07-29 `.venv-qwen35-grpo` 建立时装成了公版 trl 0.16.1（大概率是当时 uv 拉取 pypi 的默认版本），smoke config 显示 `status="prepared"` 就退出，没走到 `SFTConfig(...)` 那行，因此这个不兼容当时没暴露。

**恢复训练的正确路径**（不是移植脚本到 0.16.1 API，而是把 venv 装回冻结版本）：

```bash
# 用 CAPA 冻结的 pin 重装 .venv-qwen35-grpo
${TRAIN_VENV}/bin/pip install -U pip
${TRAIN_VENV}/bin/pip install \
    "torch==2.6.0+cu124" --index-url https://download.pytorch.org/whl/cu124
${TRAIN_VENV}/bin/pip install \
    "transformers==5.12.0" \
    "trl==1.8.0" \
    "peft==0.19.1" \
    "accelerate==1.14.0" \
    "datasets==5.0.0"
```

如果 `transformers==5.12.0` 与 `trl==1.8.0` 在 pypi 上抓不到公开发行版（v6 study 的产出时间是 2026-07-16，一年过去后有可能 yank），备选是从公司镜像或本项目 `configs/environments/trl-cu124.lock.txt` 提供的锁定 index 拉。

装好后当前所有编排脚本可以**直接**用 `bash scripts/reproduce/run_h20_repro.sh all-train` 推进（`CAPA_QWEN35_TOKENIZER_DIR` 等 env 已在 wrapper 里自动 export）。

### 7.3.1 我这次对训练脚本的两处非侵入改动

出于让"当前不兼容 venv 都能编译过 argparse 阶段"的目的，我做过 2 处改动，均是**向前兼容**（trl 1.8.0 下等价）：

- `train_qwen35_4b_planner_v6_sft.py`：`fp16=True, bf16=False` → 由 `torch.cuda.is_bf16_supported()` 自适应；`use_cache=False` 从 `SFTConfig` 移到 `base_model.config.use_cache=False`（1.8 也是走 model.config 生效，等价）。
- `train_qwen35_4b_grpo.py`：同上 bf16 自适应；`EXPECTED_*` 常量改成 env override，默认值不变。

装回 trl 1.8.0 后训练脚本仍然可以直接跑通。如果要严格回到冻结契约，把这两处改回 hard-code（在 H20 上会自动被 bf16 覆盖，无副作用），但由于 SFTConfig 参数集在 1.8.0 下已经完整，这两行改动并非必要。

### 7.4 交付物清单（本轮）

新增/修改文件：

- `scripts/reproduce/write_v6_grpo_step_manifest.py`（新增，已验证）
- `scripts/reproduce/run_h20_repro.sh`（新增，phase 化编排；已加入所有 CAPA_* env 默认 export）
- `scripts/reproduce/write_h20_compare_report.py`（新增；对照 md 已生成）
- `training/planner_grpo_seed_v1/scripts/train_qwen35_4b_planner_v6_sft.py`（改：bf16 自适应；`use_cache` 移到 model.config）
- `training/planner_grpo_seed_v1/scripts/train_qwen35_4b_grpo.py`（改：bf16 自适应；EXPECTED_* 常量改成 env override）
- `training/planner_grpo_seed_v1/step_data/planner_retry_migrate_v6_grpo_{train,dev}_qwen35_4b_nothinking_step2.manifest.json`（新增 sidecar，SHA256 均已固化）
- `reports/H20_THREE_SCENARIO_COMPARE.md`（新增，三场景 × arm 对照）

产物位置：

```
capa_h20/artifacts/CAPA/repro_h20/
├── status/prep.done, eval-base_4b.done, eval-base_35b.done
└── eval/
    ├── 20260801_123112_base_4b/
    │   ├── routing90/{summary_aggregate.json, base_4b_run{1,2,3}.json, base_4b_aggregate.json}
    │   ├── multistep/{base_4b_aggregate.json, base_4b_run{1,2,3}_*}
    │   ├── softbnd_dev/{base_4b_aggregate.json, base_4b_run{1,2,3}_*}
    │   └── summary.json
    └── 20260801_125005_base_35b/
        └── (same structure, base_35b_*)
```

### 7.5 恢复训练所需的下一步

**不需要**移植脚本，只需把 `.venv-qwen35-grpo` 装回冻结版本（见 §7.3 的 pip 命令）。然后：

```bash
# 1. dry-run 先确认 prepared / audit 全 pass
CAPA_SKIP_TOKEN_COUNT_DRIFT=1 CAPA_EXPECTED_EOS_ID=151645 CAPA_EXPECTED_PAD_ID=151643 \
CAPA_EXPECTED_MODEL_CLASS=Qwen3ForCausalLM \
CAPA_QWEN35_TOKENIZER_DIR=/apdcephfs_hzlf/share_1227201/zkq/capa_h20/models/Qwen3.5-4B \
bash scripts/reproduce/train_qwen35_4b_h20_smoke.sh sft-dry

# 2. 正式训练：SFT → merge → SFT eval → GRPO×3 → GRPO eval → compare → gate
bash scripts/reproduce/run_h20_repro.sh all-train

# 3. 若三 seeds 都过门，最后一次 sealed
bash scripts/reproduce/run_h20_repro.sh sealed
```

### 7.6 关于历史 94% 结果（对应用户提问）

历史 sealed-test 高分**不是** 35B、也**不是**单纯 accuracy，而是 **Qwen3.5-4B 在 v6 SFT ckpt-100 上**的**决策位 action_match_rate**：

- `unique_decisions action_match_rate = 0.9449`（780 独立决策位）
- `unique_decisions mean_rule_reward = 0.9544`
- `full_trajectory_match = 0.9089`（450 完整 3-step 轨迹全对率）
- 唯一短板 `core_retryable_fresh = 0.700`

这是"SFT 之后"的成绩，我这轮报的 `base_4b softbnd_dev = 0.5133` 是**未训练的下限**（在 grpo_dev 225 case 上跑的 mean_score），两个数不是同一件事。装回冻结版本 venv 后，用 v6 dev 复跑同一 SFT 就能验证复现 ~0.94 action_match。
