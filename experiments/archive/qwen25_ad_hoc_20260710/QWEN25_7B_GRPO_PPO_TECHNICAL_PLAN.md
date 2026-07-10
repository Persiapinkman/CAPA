# CAPA Qwen2.5-7B-Instruct GRPO/PPO 后训练技术方案

- 日期：2026-07-10
- 当前目标：从零重启 CAPA Planner 后训练路线，旧 Qwen3.5 训练只作为工程教训，不作为目标延续。
- 必做算法：GRPO 和 PPO 都要做。
- 当前任务：多步工具路由，即 Planner 在每一步输出结构化 JSON 决策；后续再扩展为真实 agent RL。
- 基座模型：`/raid/zkq/models/Qwen2.5-7B-Instruct`
- 精度和参数策略：V100 上统一 `fp16`；只做 LoRA/小头训练，不做 7B 全参训练。
- 主记录位置：本文件；旧实验台账仍保留在 `experiments/EXPERIMENT_LOG.md` 供追溯。

---

## 1. 目标锁定

### 1.1 本阶段优化对象

本阶段不是泛化聊天能力，也不是完整 agent 环境中的长期规划，而是 CAPA 当前已经可评测的多步工具路由：

1. 用户请求进入 Planner。
2. Planner 基于工具说明、图像可用性、历史状态和 mock observation，输出一个 JSON 决策。
3. 每个 case 最多 3 step。
4. 评测按 case 级 pass-all：每一步动作、关键参数、`finish_after_tool`、禁止动作和状态转移都要正确。

核心 hard cases：

- `probe_then_migration`：先视觉探针，再迁移建议；不能跳过探针，也不能探针后提前停止。
- `migration_feasibility_with_image`：有图时迁移建议要正确带上图像/视觉探针上下文。
- `full_detection_eval` / `general_answer` / `adela_eval` 等边界类，作为回归保护。

### 1.2 不做什么

- 不做 Qwen2.5-7B 全参训练。
- 不优先做长上下文 agent 轨迹 RL。
- 不先追求 vLLM 高吞吐训练 rollout；V100 兼容性优先。
- 不把旧 Qwen3.5-4B/9B/35B 结果当作新路线的 baseline。

---

## 2. 项目当前进度

### 2.1 现有可复用工程

| 模块 | 当前状态 |
|---|---|
| Planner prompt / 工具路由 | `demo/agent.py`、`demo/tools/*` 已可生成多步 Planner 上下文 |
| 多步数据 | `training/planner_grpo_seed_v1/cases/planner_grpo_focused_4b_cases.jsonl` |
| 多步评测 | `training/planner_grpo_seed_v1/scripts/run_repeated_planner_grpo_eval.py` |
| 规则 reward | `training/planner_grpo_seed_v1/scripts/reward_planner_grpo.py` |
| verl 数据转换 | `training/planner_grpo_seed_v1/scripts/prepare_verl_grpo_data.py`，可复用为 RL parquet 数据 |
| verl reward | `training/planner_grpo_seed_v1/scripts/verl_reward_planner_grpo.py`，可复用 reward 逻辑 |
| Qwen2.5 下载脚本 | `scripts/download_qwen25_7b_instruct.sh` |
| Qwen2.5 verl GRPO LoRA 启动脚本 | `scripts/run_qwen25_7b_verl_grpo_lora.sh`，仅作为 gated candidate |
| V100/verl 兼容 patch | `scripts/patch_verl041_v100_compat.py`，说明 verl 在 V100 上不是原生无风险路径 |

### 2.2 数据现状

当前 verl parquet 已生成：

| 文件 | 数量 |
|---|---:|
| `training/planner_grpo_seed_v1/verl_data/train.parquet` | 221 step samples |
| `training/planner_grpo_seed_v1/verl_data/val.parquet` | 24 step samples |
| 合计 | 245 step samples |

来源是 focused 多步 case，保留 step-level ground truth、类别、禁止动作、reward spec、前一步动作和完整期望动作序列。

二次审查注意事项：

- 当前 parquet 底层 pyarrow schema 中 `prompt` 是 nested `list<struct<role, content>>`，但普通 pandas 读取会显示为 `None`；因此不能只用 pandas 表面检查判断数据可用，必须用目标框架的 dataloader 做 smoke。
- focused 训练集与 compound245 formal eval 有重叠：focused 154 case 中有 86 case 出现在 eval245；eval245 全部来自 `planner_grpo_train_cases.jsonl`。因此 compound245 只能作为当前任务回归集，不能作为唯一泛化评测集。
- 需要新增一个 held-out eval：按 case_id 和 query template 去重，至少保留 `probe_then_migration`、`migration_feasibility_with_image`、`general_answer` 三类未见 query/fixture 组合。

### 2.3 Qwen2.5-7B-Instruct T0 基线

已有单轮 T0 多步评测：

- 报告：`training/planner_grpo_seed_v1/reports/repro_eval/qwen25_7b_instruct_compound245_t0_run1_aggregate.json`
- 模型：`Qwen2.5-7B-Instruct`
- case 数：245
- 通过：112
- 失败：133
- `pass_rate = 0.457143`
- `mean_score = 0.903151`

关键类别：

| 类别 | pass_rate | 结论 |
|---|---:|---|
| `single_image_probe` | 1.0000 | 单步探针强 |
| `historical_asset_qa` | 1.0000 | RAG 边界强 |
| `migration_feasibility` | 0.7368 | 纯文本迁移可训 |
| `probe_then_migration` | 0.0000 | 当前第一优先级 |
| `migration_feasibility_with_image` | 0.0000 | 当前第一优先级 |
| `general_answer` | 0.0000 | 需要 guardrail |

注意：这只是 1 repeat，不是 formal 3 repeat。正式 T0 应补跑 3x。

### 2.4 旧路线提供的工程教训

旧 Qwen3.5-4B 全参 FSDP GRPO 多次失败的根因是在线 `generate()` 的 activation/KV/SDPA prefill 峰值，不是单纯参数或 optimizer state。FSDP 能切参数和 optimizer state，但不能消除 rollout 生成峰值。

因此新路线采用：

- Qwen2.5-7B-Instruct 作为唯一基座。
- LoRA actor。
- `fp16`。
- GRPO/PPO 先以 TRL/Transformers + HF generate/eager attention 做稳定性 smoke；verl 只有通过 V100 smoke 后再进入长跑候选。
- PPO 必须拆清楚 rollout、ref、critic 的显存职责。

---

## 3. 本机硬件与环境约束

### 3.1 硬件

2026-07-10 当前 `nvidia-smi`：

- 8 x Tesla V100-SXM2-32GB
- compute capability 7.0
- 每卡约 32GB，当前基本空闲
- CPU/RAM 足够：约 40 物理核，500GB RAM

### 3.2 V100 关键限制

1. V100 不支持原生 bf16，必须显式覆盖为 `fp16`。
2. Qwen2.5-7B-Instruct 的 `config.json` 标注 `torch_dtype=bfloat16`，训练/推理脚本必须覆盖 dtype。
3. FlashAttention-2 主要面向 Ampere/Ada/Hopper，V100 不能假设可用。
4. vLLM 旧版文档支持 compute capability 7.0，但实际版本、attention backend、CUDA/PyTorch ABI 都可能影响 V100 稳定性。

### 3.3 当前 Python 环境

使用 `.venv-train-cu124`：

| 包 | 版本 |
|---|---|
| torch | 2.6.0+cu124 |
| transformers | 4.51.3 |
| trl | 0.9.6 |
| peft | 0.19.1 |
| accelerate | 1.14.0 |
| vllm | 0.8.5.post1 |
| verl | 0.4.1 |
| ray | 2.56.0 |

本地框架检查结论：

- `.venv-train-cu124`：`trl==0.9.6`，能 import `PPOTrainer`，不能 import `GRPOTrainer`。
- `.venv-train`：`trl==1.7.0.dev0`，但当前 vLLM 版本与 TRL 期望不匹配，`GRPOTrainer` import 被 `StructuredOutputsParams` 兼容问题卡住，`PPOTrainer` 顶层 API 也不可用。
- 因此需要新建一个干净的 TRL/cu124 环境做首选 smoke：先不装 vLLM，或确保 `use_vllm=False` 时不会 import 失败；不要直接把现有 `.venv-train` 或 `.venv-train-cu124` 当作完整 TRL RL 环境。
- 本地当前只有 `/raid/zkq/models/Qwen2.5-7B-Instruct`，没有 `/raid/zkq/models/Qwen2.5-1.5B-Instruct`；正式 PPO critic/value 路线开始前必须先下载并 smoke 1.5B。

必须保留的环境设置：

```bash
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/raid/zkq/projects/CAPA/training/verl_flash_attn_shim:$PYTHONPATH
export VERL_HF_ROLLOUT_KEEP_WRAP_POLICY=1
```

---

## 4. 业界可复用经验

### 4.1 V100-32G 上不要硬上 7B 全参 PPO

DeepSpeed-Chat/DeepSpeed-HE 的公开 RLHF 经验显示，RLHF 不是普通 SFT：actor、reference、reward/critic、生成缓存和训练状态都会叠加。其论文中的单卡模型规模表给出的 V100-32G RLHF 可承载量远低于 7B，V100-32G 单卡对应 OPT-2.7B 级别，而 A100-40G 才到 OPT-6.7B 级别。

可复用结论：

- 7B PPO 不应做单卡全参。
- 多 V100 也必须使用 LoRA、offload、短 response、小 rollout batch。
- PPO 的 critic 不应再用 7B；本项目采用 Qwen2.5-1.5B value/critic。

来源：https://arxiv.org/pdf/2308.01320

### 4.2 生成阶段是 RLHF 工程瓶颈

DeepSpeed-Chat 强调 RLHF 的 generation phase 是系统设计重点，Hybrid Engine 专门优化生成和训练两阶段的数据移动与吞吐。CAPA 旧 FSDP GRPO 的 OOM 也印证了这一点：参数能 shard，但在线生成峰值仍可能爆。

可复用结论：

- rollout 与训练显存要分开估算。
- 对 V100 先牺牲吞吐换稳定：`ROLLOUT_N=2`、`max_response_length=384/512`、micro batch 1。
- 评测服务和训练 rollout 不要默认共用同一套高吞吐推理假设。

来源：https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepspeed-chat/README.md

### 4.3 ZeRO/Offload 的经验只作为兜底，不作为第一选择

ZeRO-Offload 论文指出，fp16 Adam 混合精度训练的模型状态约为 `16 * 参数量` 字节，并通过 CPU offload 降低 GPU model-state 压力。它在 V100 上证明过大模型训练可行，但代价是通信和实现复杂度。

可复用结论：

- 对 CAPA 当前 245 step 小数据，不值得先引入复杂全参/offload 主路线。
- 可把 offload 用在 ref/critic 或 PPO critic optimizer，而不是 actor 全参训练。

来源：https://arxiv.org/abs/2101.06840

### 4.4 GRPO 是当前更适合 V100 的第一条 RL 算法，不等于 verl 是最低风险框架

GRPO 用同一 prompt 的一组 completions 估计相对 advantage，不需要单独训练 value model。verl 官方 baselines 已列出 Qwen2.5-7B-Instruct 的 GRPO-LoRA 记录，也列出 Qwen2-7B GRPO 和 PPO 类 baseline，说明这条技术路线属于主流可复用路径。

但框架选择必须单独判断。verl 官方 attention 文档说明 FSDP worker 默认使用 `flash_attention_2`，并建议在硬件不兼容时改用 `eager` 或 `sdpa`。GitHub issue 中也有 Qwen2.5 + GRPO 在 V100 上因 FlashAttention/Triton 失败的案例。CAPA 当前脚本之所以需要 `training/verl_flash_attn_shim` 和 `scripts/patch_verl041_v100_compat.py`，本身就说明 verl 不是 V100 的零改动稳定路径。

可复用结论：

- CAPA 第一阶段仍优先做 GRPO LoRA，但首选实现应是 TRL/Transformers HF generate 路径。
- verl 作为第二候选：只有在 `fp16 + eager/sdpa + HF rollout + LoRA` smoke 连续通过后，才允许进入长跑。
- PPO 同步做，但先按工程风险更高的路线管理。

来源：https://verl.readthedocs.io/en/latest/algo/baseline.html

### 4.5 PPO 必须明确 critic/value 设计

TRL PPO 文档的指标包含 `loss/value_avg`、`val/ratio`、`objective/kl` 等，说明标准 PPO 训练中 value function 是核心部分。CAPA 不能把无 critic 的 clipped policy loss 称为最终 PPO，只能称为 PPO-lite smoke。

可复用结论：

- 正式 PPO 使用 1.5B critic/value model。
- ref model 冻结，actor LoRA 更新，critic 小模型更新 value head/LoRA。
- 如果 verl PPO 的异构 7B actor + 1.5B critic 在本环境不顺，应保留项目内 PPO rollout-buffer 实现。

来源：https://huggingface.co/docs/trl/en/ppo_trainer

### 4.6 SFT 仍然是 RL 前置稳定器

Qwen 官方训练文档提供 Qwen2.5 SFT 的 LLaMA-Factory 路线，支持单卡/多卡、全参、LoRA、Q-LoRA、DoRA。CAPA 不一定要用 LLaMA-Factory，但应复用其行业共识：先用 SFT 固化 JSON 格式和路由正例，再做 RL。

可复用结论：

- SFT 不作为最终目标，但作为 GRPO/PPO 的初始化 checkpoint。
- SFT 数据只放正确 Planner JSON，不放采样噪声。

来源：https://qwen.readthedocs.io/en/latest/training/llama_factory.html

### 4.7 V100 推理和 rollout 框架要保守

vLLM 0.7.1 文档要求 NVIDIA compute capability >= 7.0，覆盖 V100；但同时 vLLM 明确建议使用干净环境，因为 CUDA/PyTorch ABI 和 kernel 编译会影响兼容性。FlashAttention-2/V100/bf16 组合风险高。

TRL 官方文档也把 vLLM 作为加速在线方法的可选路径，并明确 colocate 模式可能造成显存竞争，server 模式需要独立 GPU。当前 CAPA 的核心瓶颈不是吞吐，而是先在 V100 上稳定完成 LoRA RL；所以第一阶段不应默认启用 vLLM rollout。

可复用结论：

- 训练 rollout 第一选择 TRL/Transformers HF generate + eager/sdpa attention。
- eval serving 可继续用 vLLM，但必须 smoke：最小 JSON、短 prompt、长 prompt、3x eval。
- 所有脚本显式设置 `dtype=float16` 或等价参数。

来源：https://docs.vllm.ai/en/v0.7.1/getting_started/installation/gpu/

---

## 5. 总体技术路线

```text
Qwen2.5-7B-Instruct
  -> T0 formal eval 3x
  -> SFT-LoRA bootstrap
  -> GRPO-LoRA on step-level multi-step routing
     first implementation: TRL/HF generate; verl only after V100 smoke
  -> PPO-LoRA with Qwen2.5-1.5B critic/value
  -> formal compound eval 3x + single-step regression eval
  -> adapter merge/export + demo gateway smoke
```

### 5.1 SFT-LoRA bootstrap

目的：

- 固化 Planner JSON 格式。
- 降低 RL 初期无效 JSON 和错误工具名。
- 为 GRPO/PPO 提供同一个 SFT 初始化点。

数据：

- 从 focused case 的 `expected_decisions` 生成 step-level SFT 正例。
- 包含 mock 前序 observation，使 step 2 能看到 step 1 状态。
- 额外加入少量 guardrail：`general_answer`、`clarify`、`full_detection_eval`、`adela_eval`。

当前缺口：

- 仓库现有 `demo/eval/train_planner_sft.py` 默认使用旧 DPO chosen 数据和 Qwen3.5-4B；还没有 Qwen2.5-7B 的 step-level SFT 数据构建/启动脚本。
- SFT 必须从 multi-step case 构造 prompt/completion，不能直接复用旧单步 DPO chosen 数据作为本轮主 SFT。

建议超参：

| 项 | 值 |
|---|---|
| base | `/raid/zkq/models/Qwen2.5-7B-Instruct` |
| dtype | fp16 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| target modules | `q_proj,k_proj,v_proj,o_proj` 起步；显存允许再试 `all-linear` |
| max prompt | 4608 |
| max response | 512 |
| lr | `1e-5` 到 `2e-5` |
| epoch | 2 到 3，小数据早停 |
| batch | per GPU 1，gradient accumulation 8/16 |

产物：

- `outputs/planner-sft-qwen25-7b-lora/`
- 作为 GRPO/PPO 的 actor init。

### 5.2 GRPO-LoRA

目的：

- 直接优化多步路由 reward。
- 首先攻克 `probe_then_migration` 和 `migration_feasibility_with_image`。
- 不引入 learned critic，降低 V100 显存压力。

当前已有 verl 启动脚本，但它不是首选长跑入口，只能作为 V100 smoke 候选：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NGPUS_PER_NODE=4 \
MODEL_PATH=/raid/zkq/models/Qwen2.5-7B-Instruct \
LORA_RANK=16 \
LORA_ALPHA=32 \
TRAIN_BATCH_SIZE=8 \
PPO_MINI_BATCH_SIZE=8 \
ROLLOUT_N=2 \
ROLLOUT_NAME=hf \
ROLLOUT_DTYPE=float16 \
MAX_PROMPT_LENGTH=4608 \
MAX_RESPONSE_LENGTH=512 \
PPO_MAX_TOKEN_LEN_PER_GPU=8192 \
ACTOR_LR=5e-6 \
TOTAL_EPOCHS=1 \
SAVE_FREQ=20 \
bash scripts/run_qwen25_7b_verl_grpo_lora.sh
```

建议 smoke 顺序：

1. `MAX_RESPONSE_LENGTH=256`，`ROLLOUT_N=2`，只跑 2 到 5 个 optimizer step。
2. 成功后改 `MAX_RESPONSE_LENGTH=384`。
3. 成功后再用 512。
4. 只有显存稳定且 reward 方差不足时，才从 `ROLLOUT_N=2` 提到 4。

如果使用 verl，关键配置必须包括：

- `algorithm.adv_estimator=grpo`
- `algorithm.use_kl_in_reward=False`
- `actor_rollout_ref.model.lora_rank=16`
- `actor_rollout_ref.actor.kl_loss_coef=0.001`
- `actor_rollout_ref.rollout.name=hf`
- `actor_rollout_ref.rollout.dtype=float16`
- `actor_rollout_ref.ref.fsdp_config.param_offload=True`
- `+actor_rollout_ref.model.override_config._attn_implementation=eager` 或等价 eager/sdpa 覆盖

TRL 首选实现要求：

- 新建干净 cu124 环境，安装支持 `GRPOTrainer` 的 TRL 版本。
- 初始不安装 vLLM，或确保 `use_vllm=False` 时不会 import 失败。
- model load 显式 `torch_dtype=torch.float16`，attention 走 `eager` 或 `sdpa`。
- PEFT LoRA：`r=16, alpha=32`，先只训 attention projection，稳定后再试 `all-linear`。
- 当前仓库 `training/planner_grpo_seed_v1/scripts/train_planner_grpo.py` 是 TRL GRPO 原型，但依赖可 import 的 `GRPOTrainer`；需要在干净 TRL 环境中重跑 import/data/collator/generate smoke 后才能作为首选脚本。

### 5.3 PPO-LoRA

目的：

- 与 GRPO 做同任务、同基座、同 reward 的对照。
- 为后续真正 agent RL 保留 actor-critic 路线。

正式 PPO 组件：

| 组件 | 选择 |
|---|---|
| actor | Qwen2.5-7B-Instruct + SFT LoRA init + PPO LoRA 更新 |
| reference | SFT actor 冻结副本；必要时 CPU/ref offload |
| critic/value | Qwen2.5-1.5B-Instruct + value head；优先 value head + LoRA 小训练 |
| reward | 与 GRPO 同一规则 reward |
| rollout | 先 HF/eager，后续再评估 vLLM |
| dtype | fp16 |

为什么 critic 用 1.5B：

- 7B critic 会显著抬高显存和 optimizer state。
- 当前 reward 是短 horizon、step-level、规则可验证，critic 只需降低 PPO advantage 方差，不需要 7B 语义能力。
- 未来真实 agent RL 可再升级 critic。

PPO 实现路线：

1. 优先在 TRL/项目内 PPO 上做 smoke：因为本地 `.venv-train-cu124` 已确认有 `PPOTrainer`，且 PPO 可先不用 vLLM。
2. verl PPO 作为第二候选：`algorithm.adv_estimator=gae`，actor 7B LoRA，critic 1.5B value model；必须先验证 V100 attention/dtype/offload。
3. 如果 TRL/verl 对异构 actor/critic 或 LoRA critic 支持受限，则保留项目内 PPO：
   - rollout buffer 存 `prompt,response,old_logprob,reward,value,mask`。
   - actor 只更新 LoRA。
   - critic 只更新 value head/LoRA。
   - ref 冻结并计算 KL。
4. 旧 smoke 里无 critic 的 clipped objective 只能作为 PPO-lite 工程测试，不能作为最终 PPO 结论。

当前缺口：

- 现有 `training/planner_ppo_seed_v1/scripts/train_planner_ppo.py` 是无 critic 的 PPO-lite，不满足“正式 PPO + 1.5B critic”。
- 该脚本当前没有 LoRA 注入，默认模型仍是 Qwen3.5-4B，且 `save_model()` 中提前 `return`，保存逻辑不可达；不能作为正式 PPO 长跑入口。
- 正式 PPO 需要先补齐：1.5B critic/value 模型下载、value head、actor LoRA、ref logprob、critic loss、checkpoint 保存、rollout buffer schema。

PPO 初始超参：

| 项 | 值 |
|---|---|
| actor lr | `1e-6` 到 `3e-6` |
| critic lr | `5e-6` 到 `1e-5` |
| rollout n | 1 或 2 |
| ppo epochs | 1 到 2 |
| clip range | 0.2 |
| value clip | 0.2 |
| KL coef | 0.01 起步，按 `objective/kl` 调整 |
| max response | 256/384 起步 |

---

## 6. 奖励设计

### 6.1 当前 step-level reward

当前 `reward_planner_grpo.py` 已有基础项：

| 项 | 默认权重 | 含义 |
|---|---:|---|
| `json_valid` | 0.10 | 输出可解析 JSON |
| `decision_type_valid` | 0.10 | `tool/clarify/end` 类型正确 |
| `action_match` | 0.35 | 工具动作正确 |
| `argument_match` | 0.25 | 关键参数正确 |
| `finish_after_tool` | 0.10 | 是否该结束 |
| `no_forbidden_action` | 0.10 | 不触发禁止动作 |

verl reward 又补充了 process reward：

- `no_skip_required_probe`
- `no_premature_stop`
- `no_repeated_tool`
- `final_tool_finish`

口径差异：

- focused 训练集里 91 个 probe -> migration case 含 `no_premature_stop/no_repeated_tool/no_skip_required_probe/final_tool_finish` 过程奖励。
- 当前 compound245 formal eval case 的 `reward_spec` 只显式包含 `no_forbidden_action`；formal pass-all 仍会判每步动作/参数，但过程奖励分解与训练 reward 不完全同口径。
- 需要在报告中同时输出 pass-all、step failure reason、process violation counters，避免只看 reward 均值。

### 6.2 奖励调整方向

当前 T0 的 `mean_score=0.903151` 但 pass 只有 0.457，说明 partial reward 偏宽。训练可以保留 dense reward，但评测必须用 pass-all；同时对 hard categories 做更强 process shaping。

建议：

1. 对 `probe_then_migration` step 1：探针动作和 `finish_after_tool=false` 必须高权重。
2. 对 `probe_then_migration` step 2：重复探针、提前 answerer、缺 `use_visual_probe/use_image` 要重罚。
3. 对 `migration_feasibility_with_image`：`use_image=true`、`use_visual_probe=true` 和 `user_query` 保留图像语义。
4. 对 `general_answer`：禁止误入工具链。
5. 训练 reward 输出继续归一化到 `[0,1]`，但日志必须记录失败原因分布。

### 6.3 防 reward hacking

必须做的检查：

- JSON 可解析但字段无意义：不能仅靠 `json_valid` 得分。
- 输出多个 decision：只接受当前 step 第一个有效 decision 或明确规范处理。
- 复制 prompt 中工具名但参数错：`argument_match` 必须足够强。
- 总是输出 migration_advisor：通过 forbidden action 和 category-specific reward 抑制。
- 总是输出 qwen_detection：通过 step 2 `no_repeated_tool` 抑制。

---

## 7. 训练与评测框架

### 7.1 训练框架

优先级：

1. SFT：Transformers/TRL SFTTrainer 或项目内 completion-only Trainer；只要产出 PEFT adapter 即可。
2. GRPO 首选：干净 TRL/cu124 环境 + `GRPOTrainer` + `use_vllm=False` + HF generate/eager attention。
3. GRPO 候选：verl 0.4.1 + 本地 V100 patch + HF rollout；只有 smoke 通过才长跑。
4. PPO 首选：TRL/项目内 PPO rollout-buffer；明确 actor/ref/critic/value。
5. PPO 候选：verl PPO；若异构 critic、LoRA 或 V100 attention 不稳定，不进入长跑。

### 7.2 推理和 formal eval

正式评测协议沿用 `experiments/EVALUATION_POLICY.md`：

- `temperature=0`
- `top_p=1`
- `do_sample=false`
- `seed=42`
- 3 repeats
- 输出 aggregate、case audit CSV、failed cases CSV

多步评测命令模板：

```bash
MODEL=Qwen2.5-7B-Instruct \
API_BASE=http://127.0.0.1:8004/v1 \
REPORT_PREFIX=qwen25_7b_sft_grpo_compound245_3x \
CASES=/raid/zkq/projects/CAPA/training/planner_grpo_seed_v1/cases/planner_grpo_compound245_eval_cases.jsonl \
RUNS=3 \
PYTHON_BIN=/raid/zkq/projects/CAPA/.venv-train-cu124/bin/python \
bash scripts/run_grpo_repro_eval_3x.sh
```

单步回归也要跑：

```bash
MODEL=Qwen2.5-7B-Instruct \
API_BASE=http://127.0.0.1:8004/v1 \
REPORT_PREFIX=qwen25_7b_sft_grpo_zip90_3x \
RUNS=3 \
PYTHON_BIN=/raid/zkq/projects/CAPA/.venv-train-cu124/bin/python \
bash scripts/run_vllm_repro_eval_3x.sh
```

---

## 8. 实验排期

### Phase 0：冻结从零基线

输出：

- Qwen2.5-7B-Instruct compound 245 formal 3x。
- 单步 90 formal 3x。
- case audit 中按类别统计失败原因。

通过条件：

- serving 无重复乱码、无大量 timeout、无空决策。
- 所有报告落在 `training/planner_grpo_seed_v1/reports/repro_eval/`。

### Phase 1：SFT-LoRA

输出：

- `outputs/planner-sft-qwen25-7b-lora/`
- SFT 后 compound 245 3x 和 single-step 90 3x。

目标：

- JSON invalid 接近 0。
- compound pass-all 明显高于 T0。
- 不能牺牲 `single_image_probe` 和 `historical_asset_qa`。

### Phase 2：GRPO-LoRA

输出：

- TRL 首选产物：`outputs/planner-grpo-qwen25-7b-trl-lora-r16/`
- verl 候选产物：`outputs/planner-grpo-qwen25-7b-verl-lora-r16/`
- 每 20 step checkpoint。
- smoke、短跑、完整 1 epoch 三档记录。

目标：

- `probe_then_migration` 从 0 起步提升。
- `migration_feasibility_with_image` 从 0 起步提升。
- compound pass-all 首个目标：`>=0.65`；稳定目标：`>=0.75`。

### Phase 3：PPO-LoRA + 1.5B critic

输出：

- actor LoRA checkpoint。
- critic/value checkpoint。
- PPO metrics：KL、policy loss、value loss、clipfrac、reward、entropy。

目标：

- 与 GRPO 同数据、同 reward、同 eval 协议对比。
- 如果 PPO 比 GRPO 慢但更稳定，要保留。
- 如果 PPO 不稳定，先定位 critic/value loss 和 KL，而不是直接放弃 PPO。

### Phase 4：对比与固化

比较表：

| 模型/adapter | compound pass-all | pass-rate mean | single-step acc | probe_then_migration | migration_with_image | 平均耗时 | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen2.5-7B T0 | TBD 3x | TBD | TBD | TBD | TBD | TBD | run1 pass=0.457 |
| SFT-LoRA | TBD | TBD | TBD | TBD | TBD | TBD |  |
| SFT + GRPO-LoRA | TBD | TBD | TBD | TBD | TBD | TBD |  |
| SFT + PPO-LoRA | TBD | TBD | TBD | TBD | TBD | TBD |  |

最终只接受 formal 3x 结果进入 `experiments/EXPERIMENT_LOG.md`。

---

## 9. 风险与应对

| 风险 | 表现 | 应对 |
|---|---|---|
| V100 bf16 误用 | 慢、报错、数值异常 | 所有脚本强制 fp16；忽略模型 config 的 bf16 |
| FlashAttention/vLLM 不兼容 | serving 崩溃或乱码 | 训练 rollout 用 HF/eager；eval serving 先 smoke |
| GRPO rollout OOM | generate 阶段 OOM | `ROLLOUT_N=2`、response 256/384、micro batch 1、prompt 截断 |
| reward 太宽 | mean score 高但 pass 低 | 强化 process reward；按 pass-all 选模型 |
| PPO critic 太重 | OOM 或极慢 | critic 用 1.5B，value head/LoRA，小 batch/offload |
| RL 破坏单步能力 | single-step regression 下降 | 每个 checkpoint 同跑 90-case regression |
| 小数据过拟合 | 训练集涨、formal 不涨 | 保留 val split；增加 hard negative 和扰动 prompt |
| train/eval 重叠 | compound245 提升但 held-out 不涨 | 新增 held-out eval；正式结论同时报 seen/unseen |
| PPO 实现不完整 | 无 critic 或不能保存 checkpoint | 把现有 PPO-lite 仅作 smoke；正式 PPO 先补 1.5B critic/value 与保存 |
| SFT 数据不匹配 | SFT 只学旧单步边界 | 构建 multi-step step-level SFT 数据，不复用旧 DPO 作为主数据 |
| 框架数据加载假阳性 | parquet 文件存在但目标框架读不对 | 用 TRL/verl dataloader 各自跑 batch smoke，检查 prompt/token/reward |

---

## 10. 立即执行清单

1. 补跑 Qwen2.5-7B-Instruct compound 245 formal 3x，替代当前 run1 baseline。
2. 从 `planner_grpo_*cases.jsonl` 切分 seen/held-out，生成不重叠 held-out eval。
3. 生成 Qwen2.5 multi-step SFT step-level 数据和 SFT-LoRA 脚本。
4. 先跑 SFT-LoRA smoke，再跑 1 到 3 epoch。
5. 用 SFT adapter 初始化 GRPO-LoRA，不直接从 raw instruct 开始。
6. 新建干净 TRL/cu124 RL 环境，先验证 `GRPOTrainer` import、`use_vllm=False`、Qwen2.5-7B fp16 LoRA 1-step smoke。
7. verl 只做对照 smoke：2 到 5 step，`ROLLOUT_N=2`，`MAX_RESPONSE_LENGTH=256/384`，eager/sdpa attention；不通过则不进入长跑。
8. 下载并验证 Qwen2.5-1.5B-Instruct，补齐正式 PPO：7B actor LoRA + 1.5B critic/value。
9. 每个可用 checkpoint 必须跑 compound245 seen、held-out compound、single-step 90 三类 eval。

---

## 11. 参考资料

- DeepSpeed-Chat paper: https://arxiv.org/pdf/2308.01320
- DeepSpeed-Chat README: https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepspeed-chat/README.md
- ZeRO-Offload paper: https://arxiv.org/abs/2101.06840
- verl algorithm baselines: https://verl.readthedocs.io/en/latest/algo/baseline.html
- TRL GRPO Trainer: https://huggingface.co/docs/trl/en/grpo_trainer
- TRL PPO Trainer: https://huggingface.co/docs/trl/en/ppo_trainer
- Qwen LLaMA-Factory training docs: https://qwen.readthedocs.io/en/latest/training/llama_factory.html
- vLLM GPU installation docs v0.7.1: https://docs.vllm.ai/en/v0.7.1/getting_started/installation/gpu/
- NVIDIA V100 bf16 forum note: https://forums.developer.nvidia.com/t/bfloat-is-not-supported/290976
- GRPO + LoRA with verl engineering note: https://huggingface.co/blog/Weyaxi/engineering-handbook-grpo-lora-with-verl
