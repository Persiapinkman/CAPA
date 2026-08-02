# CAPA 复现清单（部署 / 训练 / 评测）

面向重新部署 Demo、重跑 Planner 的 SFT → GRPO 三种子、再执行 3× 确定性评测并写回 registry 的完整链路。
所有阶段都必须先在**开发（development）**证据层通过；`sealed test` 只在**预注册开发门**通过后开启一次。

## 阶段总览

| 阶段 | 目的 | 主要脚本 |
|---|---|---|
| P0 preflight | 环境/硬件/仓库/模型/服务连通性核对 | `scripts/reproduce_preflight.py` |
| P1 env | 建两套 venv（demo 轻量；训练重）| `scripts/reproduce/reproduce_all.sh env` |
| P2 models | 下载 `Qwen2.5-7B-Instruct` 到 `/raid/zkq/models/` | `scripts/download_qwen25_7b_instruct.sh` |
| P3 services | 起 SOCKS 隧道 + RAG 隧道 + Qwen 检测/模型网关探活 | `pipelines/demo/open_rag_tunnel.sh` |
| P4 data | 注册 planner 数据集（清单 + SHA256 + leakage 审计） | `pipelines/data/*.py` |
| P5 test | 单测契约 | `unittest discover -s tests` |
| P6 demo smoke | 无副作用端到端 smoke | `pipelines/demo/run_full_demo_smoke.py` |
| P7 SFT | 训练 SFTv3 initializer | `scripts/run_qwen25_7b_trl_sft_lora.sh` |
| P8 merge | LoRA merge 输出 `merged-qwen25-7b-sft-v3-chatml` | `scripts/merge_lora_adapter.py` |
| P9 GRPO ×3 | seeds 42/43/44 | `scripts/run_qwen25_7b_trl_grpo_lora.sh` |
| P10 eval 3× | 每模型 temperature=0，3 次重复 | `pipelines/eval/run_generation_eval.py` |
| P11 compare | case-macro paired CI | `pipelines/eval/compare_generation_runs.py` |
| P12 gate | 预注册开发门 & 副作用安全门 | `pipelines/eval/check_runtime_routing_multiseed_gate.py` |
| P13 registry | 追加运行 + 渲染 `reports/CURRENT.md` | `pipelines/experiments/registry_cli.py` |

## P0 — 硬性前置清单

**硬件与操作系统**
- Linux（本仓库锁定 Linux）；CUDA 12.4 驱动、V100-SXM2-32GB × 8（4-rank fallback 可）。
- fp16-only：V100 无 BF16 支持，禁止改成 bf16。
- 本地大盘：`/raid/zkq/`，至少 500 GB 用于模型、adapter、eval 产物；Git 之外。

**代码库/账号**
- 仓库路径不要求，但要能读写 `/raid/zkq/artifacts/CAPA`（大产物、日志、trace）。
- HuggingFace 账号（下载 `Qwen/Qwen2.5-7B-Instruct`）；如需 wandb，需 `WANDB_API_KEY`。
- 内网 SSH 到 RAG 服务器（`open_rag_tunnel.sh` 交互式输入密码，脚本不缓存）。

**Python / 依赖两栈**
- Python 3.10.12（`pyproject.toml` 限定 `>=3.10,<3.11`）。
- `.venv`（demo/轻量）：`pip install -e '.[demo]'`，含 fastapi + uvicorn 及运行时依赖。
- `.venv-trl-grpo-cu124`（训练重）：`pip install -e '.[train-cu124]'`，见 `configs/environments/trl-cu124.lock.txt`：
  - `torch==2.6.0+cu124`, `transformers==4.57.6`, `trl==1.8.0`, `peft==0.19.1`, `accelerate==1.14.0`, `datasets==5.0.0`。
- attn-impl 固定 `sdpa`；训练布局 accelerate 配置在 `configs/environments/accelerate_v100_{4,8}gpu_fp16.yaml`。

**外部服务（`init_env.sh`）**
| 名字 | 环境变量 | 默认地址 | 用途 |
|---|---|---|---|
| SOCKS5 | `all_proxy` | `socks5h://127.0.0.1:8888` | 到公司大模型网关的出口 |
| 模型网关 | `DEMO_LLM_API_BASE` | `http://10.111.32.253:8000/v1` | Planner / Answerer / re_question / RexOmni |
| Qwen 检测 | `DEMO_QWEN_DETECTION_URL` | `http://10.111.32.254:9012/v1` | 单图检测 |
| RAG(gbrain) | `GBRAIN_RAG_BASE_URL` | `http://127.0.0.1:6061/api/v1/rag` | 本机 SSH 转发 |
| RAG(playbook) | `RAG_BASE_URL` | `http://127.0.0.1:6062/api/v1/playbook/query` | 本机 SSH 转发 |
| Flux | 凭据化外部 API | — | 有费用，默认探活跳过 |

**数据准备**
- `training/planner_grpo_seed_v1/{cases,sft_data*,step_data}/*.jsonl` 由 `pipelines/data/register_planner_dataset.py` 生成清单与 SHA256。
- `data/datasets/*/HUMAN_REVIEW.md` 是人工样例，不要覆盖。

**证据规则复述（不可跳过）**
- `planner_focused_v3` 只作为开发集；`compound245` 是回归套，非测试集。
- `planner_runtime_probe_curriculum_v1` 与 `planner_runtime_routing_v1` 的 test split 保持密封。
- 训练方法级 promotion 需要：case-cluster CI 排除 0、三个训练种子、关键类别无回归、副作用不增加、JSON/stopping 合格、测试集未参与选型。

## 阶段命令（假设 `cd $REPO`）

### P0 preflight

```bash
python scripts/reproduce_preflight.py --out reports/preflight_$(date +%F).json
```

任一 required 项失败即停止，人工修复后再进入 P1。

### P1 建环境

```bash
# 轻量 demo 环境
python3.10 -m venv .venv && .venv/bin/pip install -U pip
.venv/bin/pip install -e '.[demo]'

# 训练环境（CUDA 12.4）
python3.10 -m venv .venv-trl-grpo-cu124
.venv-trl-grpo-cu124/bin/pip install -U pip
.venv-trl-grpo-cu124/bin/pip install -r configs/environments/trl-cu124.lock.txt
.venv-trl-grpo-cu124/bin/pip install -e '.[train-cu124]'
```

### P2 拉模型

```bash
MODEL_ID=Qwen/Qwen2.5-7B-Instruct \
LOCAL_DIR=/raid/zkq/models/Qwen2.5-7B-Instruct \
bash scripts/download_qwen25_7b_instruct.sh
```

### P3 起服务（两个终端）

终端 A（RAG 隧道，保持运行）：
```bash
bash pipelines/demo/open_rag_tunnel.sh
```
终端 B（起 demo）：
```bash
source init_env.sh
.venv/bin/python demo/demo_server.py --port 18080
```
静态契约与 live 探活：
```bash
curl -sS 127.0.0.1:18080/health | jq .
curl -sS '127.0.0.1:18080/health/capabilities?live=1' | jq .
```

### P4 数据注册与审计

```bash
.venv-trl-grpo-cu124/bin/python pipelines/data/register_planner_dataset.py
.venv-trl-grpo-cu124/bin/python pipelines/data/register_runtime_routing_dataset.py
.venv-trl-grpo-cu124/bin/python pipelines/data/register_stateful_retrieval_dataset.py
.venv-trl-grpo-cu124/bin/python pipelines/experiments/registry_cli.py validate
```

### P5 单测

```bash
PYTHONPATH=src:. .venv-trl-grpo-cu124/bin/python -m unittest discover -s tests -v
```

### P6 Demo smoke（无副作用）

```bash
source init_env.sh
.venv/bin/python pipelines/demo/run_full_demo_smoke.py --include-migration
```
如需 Flux/pipeline，需显式 `--include-flux --include-pipeline --allow-side-effects` 并承担费用。

### P7 SFT（一次）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_PROCESSES=8 \
OUTPUT_DIR=/raid/zkq/artifacts/CAPA/outputs/planner-sft-qwen25-7b-v3 \
bash scripts/run_qwen25_7b_trl_sft_lora.sh
```

### P8 合并 SFT LoRA

```bash
.venv-trl-grpo-cu124/bin/python scripts/merge_lora_adapter.py \
  --base-model /raid/zkq/models/Qwen2.5-7B-Instruct \
  --adapter /raid/zkq/artifacts/CAPA/outputs/planner-sft-qwen25-7b-v3 \
  --output-dir /raid/zkq/artifacts/CAPA/outputs/merged-qwen25-7b-sft-v3-chatml
```

### P9 GRPO × 3 seeds

`configs/train/qwen25_grpo_stateful_retrieval_v1.json` 已固定 seeds `[42,43,44]`、`num_generations=8`、`beta=0`、`loss_type=dr_grpo`；驱动用 `run_qwen25_7b_trl_grpo_lora.sh` 循环三次：

```bash
for SEED in 42 43 44; do
  MODEL_PATH=/raid/zkq/artifacts/CAPA/outputs/merged-qwen25-7b-sft-v3-chatml \
  CASES=training/planner_grpo_seed_v1/cases/planner_runtime_probe_curriculum_v1_train_cases.jsonl \
  PROMPT_FORMAT=qwen_chatml \
  OUTPUT_DIR=/raid/zkq/artifacts/CAPA/outputs/runtime_probe_grpo${SEED} \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_PROCESSES=8 \
  GENERATION_BATCH_SIZE=8 NUM_GENERATIONS=8 \
  LEARNING_RATE=2e-6 TEMPERATURE=0.7 TOP_P=0.9 \
  MAX_COMPLETION_LENGTH=128 GRAD_ACCUM_STEPS=2 MAX_STEPS=80 SAVE_STEPS=40 \
  SEED=${SEED} \
  bash scripts/run_qwen25_7b_trl_grpo_lora.sh
done
```

### P10 3× 确定性开发评测

对 baseline (`sft_v3`) 和 3 个 GRPO 种子逐一评测：

```bash
for TAG in sft_v3 runtime_probe_grpo42 runtime_probe_grpo43 runtime_probe_grpo44; do
  .venv-trl-grpo-cu124/bin/python pipelines/eval/run_generation_eval.py \
    --run-id 20260729_runtime_probe_curriculum_v2_${TAG}_dev3x \
    --study-id planner_runtime_routing_grpo_v1 \
    --model-path /raid/zkq/artifacts/CAPA/outputs/${TAG} \
    --cases training/planner_grpo_seed_v1/cases/planner_runtime_probe_curriculum_v1_dev_cases.jsonl \
    --temperature 0 --top-p 1 --do-sample false --repeats 3 --seed 42 \
    --out-dir /raid/zkq/artifacts/CAPA/outputs/eval/${TAG}_dev3x
done
```

### P11 case-macro paired 比较

```bash
.venv-trl-grpo-cu124/bin/python pipelines/eval/compare_generation_runs.py \
  --baseline /raid/zkq/artifacts/CAPA/outputs/eval/sft_v3_dev3x \
  --candidates /raid/zkq/artifacts/CAPA/outputs/eval/runtime_probe_grpo42_dev3x \
               /raid/zkq/artifacts/CAPA/outputs/eval/runtime_probe_grpo43_dev3x \
               /raid/zkq/artifacts/CAPA/outputs/eval/runtime_probe_grpo44_dev3x \
  --out reports/compare_runtime_probe_$(date +%F).json
```

### P12 预注册开发门

```bash
.venv-trl-grpo-cu124/bin/python pipelines/eval/check_runtime_routing_multiseed_gate.py \
  --study experiments/studies/planner_runtime_routing_grpo_v1/study.json \
  --baseline /raid/zkq/artifacts/CAPA/outputs/eval/sft_v3_dev3x \
  --candidates /raid/zkq/artifacts/CAPA/outputs/eval/runtime_probe_grpo42_dev3x \
               /raid/zkq/artifacts/CAPA/outputs/eval/runtime_probe_grpo43_dev3x \
               /raid/zkq/artifacts/CAPA/outputs/eval/runtime_probe_grpo44_dev3x \
  --out reports/gate_runtime_probe_$(date +%F).json
```

**门规则**：
- case-macro paired 95% CI 排除 0，方向正向。
- **mean 错误副作用动作 ≤ baseline**（这里失败过一次；不达标则密封 test）。
- JSON valid / 停止合规率不回退超过预设阈值。

### P13 追加 registry + 渲染 CURRENT

```bash
for FILE in /raid/zkq/artifacts/CAPA/outputs/eval/*_dev3x/run_record.json; do
  .venv-trl-grpo-cu124/bin/python pipelines/experiments/registry_cli.py add "$FILE"
done
.venv-trl-grpo-cu124/bin/python pipelines/experiments/registry_cli.py render
```

### P14 sealed test（仅当 P12 通过时才允许，一次机会）

```bash
.venv-trl-grpo-cu124/bin/python pipelines/eval/run_generation_eval.py \
  --run-id 20260729_runtime_probe_curriculum_v2_grpo_sealed \
  --study-id planner_runtime_routing_grpo_v1 \
  --model-path /raid/zkq/artifacts/CAPA/outputs/runtime_probe_grpo42 \
  --cases training/planner_grpo_seed_v1/cases/planner_runtime_probe_curriculum_v1_test_cases.jsonl \
  --temperature 0 --top-p 1 --do-sample false --repeats 3 --seed 42 \
  --out-dir /raid/zkq/artifacts/CAPA/outputs/eval/grpo_sealed
```

## 一键驱动

```bash
# 首次全跑（含训练；约数小时）
bash scripts/reproduce/reproduce_all.sh all

# 或按阶段
bash scripts/reproduce/reproduce_all.sh preflight
bash scripts/reproduce/reproduce_all.sh env
bash scripts/reproduce/reproduce_all.sh models
bash scripts/reproduce/reproduce_all.sh data
bash scripts/reproduce/reproduce_all.sh unittest
bash scripts/reproduce/reproduce_all.sh smoke        # 需要 P3 服务在线
bash scripts/reproduce/reproduce_all.sh sft
bash scripts/reproduce/reproduce_all.sh merge
bash scripts/reproduce/reproduce_all.sh grpo         # 3 seeds
bash scripts/reproduce/reproduce_all.sh eval         # 3x deterministic
bash scripts/reproduce/reproduce_all.sh gate         # 预注册开发门
bash scripts/reproduce/reproduce_all.sh registry
```

`--dry-run` 打印命令但不执行；`--only-gate` 跳过训练直接看当前产物是否过门；副作用阶段（Flux/pipeline）需 `ALLOW_SIDE_EFFECTS=1`。

---

# 附：H20（4×Hopper）复现路径

原始项目在 V100+fp16 上定型。切到 4×H20 时的差异：

- **精度**：一律 bf16（H20 原生），禁用 fp16 workaround。
- **推理框架**：换用 vLLM 0.7.x，Hopper FA2 kernel；4B 单卡，35B-A3B tensor-parallel=4。
- **模型来源**：内部代号 `Qwen3.5-4B` / `Qwen3.5-35B-A3B` 是别名，实际权重通过 `QWEN35_4B_REPO`（默认 `Qwen/Qwen3-4B`）与 `QWEN35_35B_REPO`（默认 `Qwen/Qwen3-30B-A3B`）解析；本地目录仍旧保留 `/raid/zkq/models/Qwen3.5-{4B,35B-A3B}/` 以匹配训练/评测脚本硬编码。
- **两 venv**：`.venv-h20-infer`（vLLM）+ `.venv-qwen35-grpo`（trainer 栈 pinned）。
- **评测入口不变**：`training/planner_grpo_seed_v1/scripts/run_repeated_planner_grpo_eval.py`，只把 `--api-base` 指向本地 vLLM。
- **训练脚本不变**：`run_qwen35_4b_planner_v6_sft.sh` / `run_qwen35_4b_grpo_v5_train_v1.sh`，用它们自带的 `dry-run` / `g4` / `canary` 模式做 smoke。

## H20 阶段清单

| 阶段 | 作用 | 脚本 |
|---|---|---|
| H0 env | 建 `.venv-h20-infer` + `.venv-qwen35-grpo` | `scripts/reproduce/setup_h20_env.sh` |
| H1 models | 下载 Qwen3.5-4B 与 Qwen3.5-35B-A3B | `scripts/reproduce/download_qwen35_models.sh` |
| H2 serve-4b | vLLM 4B（TP=1，8001）| `scripts/reproduce/serve_qwen35_vllm.sh 4b` |
| H3 serve-35b | vLLM 35B-A3B（TP=4，8002）| `scripts/reproduce/serve_qwen35_vllm.sh 35b` |
| H4 eval-4b | planner routing 3× eval | `scripts/reproduce/eval_qwen35_h20.sh 4b` |
| H5 eval-35b | planner routing 3× eval | `scripts/reproduce/eval_qwen35_h20.sh 35b` |
| H6 smoke-sft | 4B SFT dry-run + 3 step canary | `scripts/reproduce/train_qwen35_4b_h20_smoke.sh sft-{dry,smoke}` |
| H7 smoke-grpo | 4B GRPO dry-run + g4 单步 | `scripts/reproduce/train_qwen35_4b_h20_smoke.sh grpo-{dry,smoke}` |

## H20 一键命令

```bash
# 1) 环境 + 权重（首次约 60 min，取决于带宽）
bash scripts/reproduce/reproduce_all.sh h20-env h20-models

# 2) 4B 评测：拉起 vLLM -> 评测 -> 关停
bash scripts/reproduce/reproduce_all.sh h20-serve-4b h20-eval-4b h20-stop

# 3) 4B 训练 smoke（先 dry-run，再 canary/g4）
bash scripts/reproduce/reproduce_all.sh h20-smoke-sft h20-smoke-grpo

# 4) 35B-A3B 评测（占 4 卡）
bash scripts/reproduce/reproduce_all.sh h20-serve-35b h20-eval-35b h20-stop

# 或整套一次：
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/reproduce/reproduce_all.sh h20-all
```

## 与历史结果的对照

原 `results/planner_routing_eval/qwen35_35b_a3b_stateprompt_zip90_3x_aggregate.json` 和 `qwen35_4b_stateprompt_zip90_3x_aggregate.json` 是在公司远端网关（`http://10.111.32.253:8000/v1`）跑出的三重复评测。H20 上使用本地 vLLM 后：

- 相同 `--cases`（`planner_grpo_focused_val_v3_cases.jsonl`）+ 相同 `run_repeated_planner_grpo_eval.py` + 相同确定性参数（`temperature=0 top_p=1 seed=42 runs=3`）。
- 唯一差异：**推理后端**（远端 → 本地 vLLM）与 **权重来源**（内部检查点 → 公开 HF 近亲），因此结果**不保证 bit-level 一致**；只要 case-macro/step-macro 与聚合表在噪声范围内，就视为复现成功。
- 交付：`/raid/zkq/artifacts/CAPA/outputs/eval_h20/<STAMP>_qwen35_4b_h20/` 和 `<STAMP>_qwen35_35b_a3b_h20/` 目录下的 `*_aggregate.json` 与 `*_summary.json`，另外 `<STAMP>_summary.json` 是两模型合表。

## 4B 训练能否直接跑到收敛？

**Yes**（数据都在仓库内）：

- SFT：`training/planner_grpo_seed_v1/sft_data_planner_retry_migrate_v6_qwen35_nothinking/{train,dev}.jsonl`（1040/260 行，SHA256 已 pin 在 `metadata.json`）。
- GRPO：`training/planner_grpo_seed_v1/step_data/planner_multistep_grpo_value_v5_train_v1_qwen35_4b_nothinking_step2.jsonl`（480 行 + manifest）。

正式训练命令（超出 smoke 范围）：

```bash
# SFT full run
CONFIRM_TRAIN=YES CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PROCESSES=4 \
  MAX_STEPS=400 LEARNING_RATE=2e-5 GRADIENT_ACCUMULATION_STEPS=2 \
  RUN_MODE=train \
  MODEL_PATH=/raid/zkq/models/Qwen3.5-4B \
  DATA_DIR=training/planner_grpo_seed_v1/sft_data_planner_retry_migrate_v6_qwen35_nothinking \
  ENV_DIR=$(pwd)/.venv-qwen35-grpo \
  bash scripts/run_qwen35_4b_planner_v6_sft.sh

# GRPO screen (100 opt steps)
CONFIRM_TRAIN=YES CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PROCESSES=4 \
  MAX_STEPS=100 RUN_MODE=screen SEED=42 \
  MODEL_PATH=/raid/zkq/models/Qwen3.5-4B \
  ADAPTER_PATH=/raid/zkq/artifacts/CAPA/experiments/runs/<SFT_RUN_DIR>/checkpoint-100 \
  STEP_DATA=training/planner_grpo_seed_v1/step_data/planner_multistep_grpo_value_v5_train_v1_qwen35_4b_nothinking_step2.jsonl \
  ENV_DIR=$(pwd)/.venv-qwen35-grpo \
  bash scripts/run_qwen35_4b_grpo_v5_train_v1.sh
```

`CONFIRM_TRAIN=YES` 与非空 `OUTPUT_DIR` 是上游脚本的硬 guard，不要绕过。
