# Qwen3.5-4B 多步工具路由 GRPO 训练技术方案（审计稿）

日期：2026-07-15
状态：`APPROVED / G0-G2_PASSED / SUPPORT_GATE_FAILED / TRAINING_BLOCKED`
目标模型：`/raid/zkq/models/Qwen3.5-4B`
硬件：`8 × Tesla V100-SXM2-32GB`，Driver `550.163.01`，CUDA Driver API `12.4`

本文件冻结采用的框架、版本、显存策略、参数基线和训练门禁。用户已审计通过，独立环境及
G0 静态契约已于 2026-07-15 创建并通过。G1/G2 随后以真实权重通过。80-case × 8
support audit 已完成，但预先冻结的 support gate 失败，因此 G3/G4/G5 和 optimizer
step 均未启动，也没有创建新的模型或 adapter 权重。

2026-07-15 运行时修订：原先 `max_completion_length=128` 的假设被真实采样否证。
`migration_advisor.user_query` 的线上 schema 要求尽量完整保留原始需求；V5 train prompt 的
长 query 使 8/8 路由正确样本在 128 tokens 处截断。保持 prompt/schema 不变的 512-token
长度探针得到自然 EOS 长度 `81/127/137/266`，最坏为 266。因此首轮上限修订为 **320**：
比实测最坏值保留 54 tokens 余量，且仍比历史 512 低 37.5%。本修订必须重新通过 G2–G4，
不授权跳过 support gate 或直接启动训练。

2026-07-15 第二次运行时修订：单题长度探针对全数据分布不充分。正式 640 个样本中有
25 个未在 320 tokens 内生成 EOS，截断率为 3.90625%，高于 1% 硬门。因此 320
仅是本次审计的被测配置，不再视为可进入训练的已验证上限；新数据冻结后必须用全
support pool 重做长度分布和 G2。

## 1. 决策摘要

第一版 Qwen3.5-4B GRPO 采用：

- `Transformers 5.12.0 + TRL 1.8.0 + PEFT 0.19.1 + Accelerate 1.14.0`；
- `PyTorch 2.6.0+cu124`，与本机 Driver/CUDA 12.4 路径保持一致；
- 8 卡 `DDP + LoRA`，不使用 full-parameter FSDP/ZeRO；
- `use_vllm=false`，训练环境中不安装 vLLM；
- `num_generations=4` 是**全局同 prompt 的四个采样**，通过 TRL RepeatSampler 分布到
  不同 rank；`per_device_train_batch_size=1`，保证每张 GPU 本地一次只生成一条 completion；
- Qwen3.5 原生 non-thinking Chat Template；
- text-only `AutoModelForCausalLM`，不加载视觉塔；
- fp16 autocast、`GradScaler(init_scale=1)`、关闭 cuDNN、SDPA、gradient checkpointing；
- hybrid LoRA 同时覆盖 8 个 full-attention 层和 24 个 Gated DeltaNet 层；
- completion 上限由旧方案的 512 降到经实测确定的 320 tokens；
- 正式运行前依次通过版本、模板、LoRA 覆盖、最坏长度显存、finite-gradient、全 GRPO
  无更新和跨越历史 step 67 的稳定性门禁。

选择原生 Transformers DDP 的原因不是它理论吞吐最高，而是它在当前 V100 上依赖闭合、
采样与训练使用同一实现、不需要远端权重同步，并且可以直接消除旧单卡的 local generation
batch=4。vLLM server 只作为吞吐优化的第二阶段候选，不作为首轮正确性路径。

## 2. 旧 OOM 的重新归因

### 2.1 单卡 LoRA：用户判断成立

历史 `2026-07-07_4b_lora_grpo_focused` 使用单张 V100：

| 参数 | 历史值 |
|---|---:|
| `world_size` | 1 |
| `num_generations` | 4 |
| 本地 generation batch | 4 条同 prompt completion |
| `max_prompt_length` | 3072 |
| `max_completion_length` | 512，恢复尝试为 384 |
| completion mean/max | 311.8 / 512 |
| clipped ratio | 37.5% |
| OOM | optimizer step 67，`generate()` SDPA prefill |

在单卡 world size 1 下，四条 completion 确实都在同一张 GPU 上生成；同卡还常驻可训练
模型、LoRA、训练状态和此前分配的 CUDA 缓存。step 67 的长样本触发额外 2.11 GiB 分配
失败。恢复后在同一步再次 OOM，说明失败与固定样本/长度峰值高度相关，而不是随机瞬时故障。

### 2.2 但 `num_generations=4` 不是全部原因

历史 full-parameter FSDP 已把 `num_generations` 降到 2、completion 降到 160，仍发生
generation OOM。FSDP 能 shard 参数和 optimizer state，但 TRL 生成时需要在每个 rank
取得可生成的完整模型；generation activation、cache 和 Qwen3.5 linear-attention 状态仍是
rank-local 峰值。因此：

1. 单卡 LoRA 的直接问题是本地一次生成 4 条；
2. FSDP 的附加问题是 generation 阶段 full-parameter materialization 和本地状态峰值；
3. 旧 completion 上限 512 且大量打满，显著放大两类问题；
4. V100 没有 Qwen3.5 Gated DeltaNet fast kernel，纯 PyTorch fallback 进一步降低余量；
5. OOM 主因不是 LoRA optimizer state，换 8-bit optimizer不能解决 generation 峰值。

所以新方案同时修改 local batch、参数并行方式和 completion 上限，不能只改其中一个开关。

## 3. 训练拓扑

### 3.1 推荐：8 卡 DDP LoRA，同进程原生 generation

固定参数：

```text
world_size = 8
per_device_train_batch_size = 1
num_generations = 4
generation_batch_size = 8
steps_per_generation = 1  # 由 TRL 推导，不与 generation_batch_size 同时显式传入
```

TRL 1.8 的 `RepeatSampler` 明确把同一 prompt 的重复样本分发到不同进程，使同组 reward
可以跨 GPU gather 后归一化。在上述设置下，一个 generation wave 的布局是：

```text
GPU/rank 0  -> prompt A / completion A1
GPU/rank 1  -> prompt A / completion A2
GPU/rank 2  -> prompt A / completion A3
GPU/rank 3  -> prompt A / completion A4

GPU/rank 4  -> prompt B / completion B1
GPU/rank 5  -> prompt B / completion B2
GPU/rank 6  -> prompt B / completion B3
GPU/rank 7  -> prompt B / completion B4
```

每卡本地 generation batch 为 1，而不是 4。每个 rank 保留完整 4B text-only fp16 模型，
但只同步约 14.38M 个 LoRA 可训练参数。由于 LoRA 梯度量很小，跨两个 NUMA domain 的 DDP
all-reduce 不是主要瓶颈；使用 cu124/NCCL，不沿用历史 cu128 环境下的 gloo workaround。

### 3.2 共享机器上的 4-rank 等效回退

若外部用户持续占用部分 GPU，允许在明确选择的四张空闲卡上运行以下等效拓扑；不得抢占、
终止或把外部进程计入本任务：

| 参数 | 8-rank 主拓扑 | 4-rank 回退 |
|---|---:|---:|
| world size | 8 | 4 |
| generation batch | 8 | 4 |
| local generation batch | 1 | 1 |
| num generations | 4 | 4 |
| gradient accumulation | 4 | 8 |
| completions / optimizer step | 32 | 32 |
| prompt groups / optimizer step | 8 | 8 |

4-rank 回退仍把同 prompt 的四条 completion 分到四个不同 rank，保持每卡 local batch=1；
它只增加墙钟时间，不改变每次 optimizer update 的 completion/group 数。run record 必须写明
topology，G3/G4 必须在实际采用的 topology 上重跑。资源恢复后可再做 8-rank 吞吐复验，
但不得把两个 topology 的 checkpoint 混在同一 seed 链中续训。

### 3.3 为什么第一版不拆成“训练卡 + rollout 卡”

外置 rollout 可以进一步隔离显存，但会新增以下正确性问题：

- rollout 模型与训练 LoRA 的权重同步；
- rollout logprob 与训练 logprob 的数值差异及 importance-sampling correction；
- Qwen3.5、vLLM、TRL、PyTorch 四方版本闭合；
- V100 上 Qwen3.5 hybrid kernel 与服务端实际吞吐；
- 远端失败、超时和 stale-policy rollout 的恢复语义。

当前 4B text-only 模型单序列 4.1k prompt + backward 的既有峰值为 22.191 GiB，理论上
在 32 GiB V100 上有足够空间先验证 local batch=1 的 DDP 路径。只有该路径通过正确性门但
吞吐不可接受时，才进入外置 rollout 优化。

### 3.4 不采用的第一阶段方案

| 方案 | 第一阶段结论 | 原因 |
|---|---|---|
| TRL + 8卡 DDP LoRA + HF generation | **采用** | 依赖最少；同一 policy 实现；每卡一条 completion |
| TRL + vLLM colocate | 不采用 | 再次让 rollout 与训练竞争同卡显存 |
| TRL + vLLM server | 暂缓 | 需独立版本栈、权重同步和 V100/Qwen3.5 kernel 验证 |
| full-parameter FSDP/ZeRO-3 | 不采用 | 4B LoRA 已足够；历史 generation full-param 峰值反复 OOM |
| veRL / OpenRLHF / Ray 全栈迁移 | 不采用 | 当前规则 verifier、state prompt 和评测已集成 TRL；迁移变量过多 |
| QLoRA / bitsandbytes | 不采用 | 4B fp16 能放入单卡；量化引入额外数值变量且不解决生成批峰值 |

## 4. 版本方案

### 4.1 冻结版本

已创建独立环境：

```text
/raid/zkq/artifacts/CAPA/runtime/venv-qwen35-grpo-cu124-v1
```

不覆盖 `.venv-train`、`.venv-train-cu124` 或 `.venv-trl-grpo-cu124`。

| 组件 | 固定版本 | 说明 |
|---|---|---|
| Python | 3.10.12 | 与现有环境一致 |
| PyTorch | 2.6.0+cu124 | 对齐 Driver 550 / CUDA 12.4；不使用 cu128 训练路径 |
| Transformers | 5.12.0 | 稳定版，包含 `qwen3_5` 与 text-only CausalLM 映射 |
| TRL | 1.8.0 | 稳定版，包含当前 GRPO batching、DDP RepeatSampler 和 rollout 接口 |
| PEFT | 0.19.1 | 已与 TRL 1.8 import smoke |
| Accelerate | 1.14.0 | DDP 与自定义 GradScaler kwargs |
| Datasets | 5.0.0 | 与现有数据入口一致 |
| huggingface-hub | 1.19.0 | Transformers 5 稳定组合 |
| tokenizers | 0.22.2 | 已验证本地 Qwen3.5 tokenizer/template |
| safetensors | 0.8.0 | 本地权重读取 |
| vLLM | **不安装** | 防止 TRL optional import 被不匹配版本污染 |
| flash-attn | **不安装** | V100/Qwen3.5 hybrid 路径不作为首轮变量 |
| bitsandbytes | **不安装** | 不采用量化训练或 8-bit optimizer |

### 4.2 为什么不能复用现有环境

| 环境 | 能做什么 | 阻塞项 |
|---|---|---|
| `.venv-train` | Transformers 能加载 Qwen3.5 | TRL import 被 vLLM 0.10.2 的 `StructuredOutputsParams` 缺失阻塞 |
| `.venv-trl-grpo-cu124` | TRL `GRPOTrainer` 可导入 | Transformers 4.57.6 不识别 `model_type=qwen3_5` |
| `.venv-train-cu124` | cu124/FSDP 旧环境 | Transformers 4.51.3、TRL 0.9.6 均过旧，且含 vLLM 0.8.5 |

TRL 1.8 的 vLLM extra 要求 vLLM 0.16–0.23；这些版本又各自固定较新的 PyTorch/不同
Transformers 范围。Qwen3.5 官方模型卡还建议使用较新的 serving 主线。把 vLLM 强行装进
训练环境会重新制造当前的依赖冲突。因此第一版环境必须保持 vLLM-free。

### 4.3 已完成的正式 G0 静态 smoke

已在新环境执行 dependency/import/config/template/CUDA capability smoke；90 个安装包通过
`uv pip check`，CAPA 训练入口导入通过。没有加载模型权重和执行 CUDA forward。结果：

```text
torch          2.6.0+cu124
transformers   5.12.0
trl            1.8.0
peft           0.19.1
accelerate     1.14.0
datasets       5.0.0
huggingface-hub 1.19.0
tokenizers     0.22.2

AutoConfig                    -> Qwen3_5Config / qwen3_5
AutoModelForCausalLM mapping  -> Qwen3_5ForCausalLM
GRPOTrainer import            -> passed
rollout_func API              -> present
```

`enable_thinking=false` 的模板尾部为：

```text
<|im_start|>assistant
<think>

</think>

```

机器可读结果：
`experiments/studies/planner_multistep_tool_routing_grpo_qwen35_4b_v1/qwen35_grpo_environment_g0_20260715.json`。
该 smoke 只证明 Python 依赖、项目导入、配置解析和硬件能力契约闭合；真实 fp16 load、
CUDA forward/backward 和 8-rank GRPO 显存仍属于后续硬门，不能据此直接开训。

## 5. 模型加载与 Prompt 契约

### 5.1 只加载 text policy

必须使用：

```python
AutoModelForCausalLM.from_pretrained(
    "/raid/zkq/models/Qwen3.5-4B",
    dtype=torch.float16,
    attn_implementation="sdpa",
    low_cpu_mem_usage=True,
    trust_remote_code=False,
)
```

Transformers 5.12 将完整 `Qwen3_5Config` 映射到 `Qwen3_5ForCausalLM`，其加载规则忽略
视觉塔权重。禁止使用 `Qwen3_5ForConditionalGeneration`、`AutoProcessor` 或 VLM trainer，
因为本任务输入和 reward 都是文本状态/结构化动作，不需要视觉 encoder。

加载后必须审计：

- 模型类严格为 `Qwen3_5ForCausalLM`；
- 不存在已实例化的 `visual` 模块；
- unexpected keys 只允许模型声明的 vision/MTP ignore 集；
- 不允许静默 missing text weights；
- `trust_remote_code=false`；
- tokenizer、config、权重全部来自同一个本地目录和固定 checksum。

### 5.2 EOS/PAD 契约

本地 checkpoint 存在一个必须显式处理、但目前不构成已证实 bug 的 ID 差异：

```text
tokenizer.eos_token      = <|im_end|>     = 248046
tokenizer.pad_token      = <|endoftext|>  = 248044
text_config.eos_token_id = <|endoftext|>  = 248044
```

ChatML assistant turn 应在 `<|im_end|>` 停止。TRL 1.8 原生 Transformers generation 会把
tokenizer EOS/PAD 写入 generation config，因此推荐路径应使用 `eos_token_id=248046`、
`pad_token_id=248044`。不能只依赖 model text config 的 EOS，也不能沿用通用代码里的
“如果不确定就令 pad=eos”逻辑覆盖本 checkpoint 已有的特殊 token 配置。

静态门和首轮 generation smoke 必须断言上述 token/ID 映射，并验证模型在完整 JSON 后生成
`<|im_end|>`、不会继续生成到 320-token 上限。现有证据不能证明该 ID 差异造成过历史 OOM；
它只是新版入口必须固定的停止契约。

### 5.3 原生 non-thinking Chat Template

不得复用 Qwen2.5 的手写 `<|im_start|>assistant\n` 尾部。训练 step data 应由 Qwen3.5
tokenizer 统一生成：

```python
tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
```

数据构建时冻结：

- rendered prompt；
- token IDs 或 token count；
- tokenizer/config SHA256；
- 模板尾部断言；
- prompt 生成代码版本。

Trainer 接收已经 render 的 plain-text prompt，不再次 apply template，防止双重模板。训练、
support audit、开发评测和正式评测必须使用相同 non-thinking contract。

### 5.4 长度策略

- 先对最终训练 step data 做 p50/p95/p99/max token audit；
- 初始硬门 `max_prompt_tokens=4608`；超过即构建失败，禁止静默 truncate；
- 若真实 max 合法地超过 4608，只能在审计数据后把上限提高到 5120，并重新做显存门；
- `max_completion_length=320`；实测自然 EOS 最坏为 266 tokens；
- 若 calibration clipped ratio 超过 1%，先修停止/template/输出契约，不直接把上限恢复到
  384/512；
- 正式评测仍使用其冻结的独立 max_tokens，不由训练 completion 上限反向修改 sealed 协议。

## 6. LoRA 方案

Qwen3.5-4B 有 8 个 full-attention 层和 24 个 linear-attention/Gated DeltaNet 层。
Qwen2.5 的 `q_proj,k_proj,v_proj,o_proj` 只覆盖前者，会漏掉 24/32 层。

第一版固定：

```text
r = 16
alpha = 32
dropout = 0.0
bias = none
task_type = CAUSAL_LM
target_modules =
  q_proj,k_proj,v_proj,o_proj,
  in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj
```

预期覆盖：

| 类型 | 层数 | LoRA module 数 |
|---|---:|---:|
| full attention：q/k/v/o | 8 | 32 |
| Gated DeltaNet：qkv/z/a/b/out | 24 | 120 |
| 合计 | 32 | 152 |

在 r=16 时，既有审计预期 trainable parameter 为 `14,376,960`，约占 text policy
`0.3407%`。正式入口必须把 module 数、模块名和 trainable parameter count 写入 run record；
任何不一致直接退出。

第一版不把 `gate_proj/up_proj/down_proj` 加入 LoRA，以避免同时改变 mixer 和 MLP 而扩大
实验变量。`lora_dropout=0.0` 用于保持 generation policy 与训练 logprob 的确定性契约；
探索性 dropout 不进入首轮。

## 7. 第一版参数基线

### 7.1 分布式与 generation

| 参数 | 建议值 | 说明 |
|---|---:|---|
| strategy | DDP LoRA | 不使用 FSDP/ZeRO |
| world size | 8 | 8 张 V100 全部作为相同 DDP rank |
| `per_device_train_batch_size` | 1 | 每卡本地一条 completion |
| `num_generations` | 4 | 每个 prompt 四采样，跨 rank 分布 |
| `generation_batch_size` | 8 | 每 wave 两个 prompt group |
| `steps_per_generation` | 不显式设置 | TRL 推导为 1 |
| `gradient_accumulation_steps` | 4 | 每 optimizer step 32 completions / 8 prompt groups |
| `num_iterations` | 1 | 不复用旧 rollout，保持近似 on-policy |
| `max_prompt_tokens` | 4608 | 构建期硬门，不 truncate |
| `max_completion_length` | 320 | 自然 EOS 实测 max=266；相对旧 512 降 37.5% |
| `temperature` | 0.7 | 与 support audit 保持一致 |
| `top_p` | 0.9 | 保留动作探索并抑制长尾无效输出 |
| repetition penalty | 1.0 | 不引入额外采样偏置 |
| generation `use_cache` | true | 仅 generation；必须单独测显存 |
| training `use_cache` | false | 配合 gradient checkpointing |
| generation EOS/PAD | 248046 / 248044 | `<|im_end|>` / `<|endoftext|>` |
| `remove_invalid_values` | true | 防止非有限 logits 传播 |
| `renormalize_logits` | true | 与 invalid-value removal 配套 |

在 8 卡布局下，不把 `num_generations=4` 降到 2 作为默认 OOM 修复。G=2 的 advantage
估计更噪，且不能解决 FSDP full-parameter generation 的结构问题。只有完整显存门证明
每卡 local batch 已为 1 仍无法运行时，才重新审计 G。

### 7.2 优化参数

| 参数 | 建议值 |
|---|---:|
| optimizer | `adamw_torch` |
| learning rate | `5e-6` |
| weight decay | `0.0` |
| scheduler | `constant_with_warmup` |
| warmup steps | 5 |
| max grad norm | `1.0` |
| loss type | `dr_grpo` |
| reward scaling | `group` |
| epsilon | `0.2` |
| beta / KL | `0.0`（第一版） |
| mask truncated completions | false |
| max optimizer steps | 100（seed42 screen 上限） |
| checkpoint steps | 25 / 50 / 75 / 100 |
| logging steps | 1 |
| save total limit | 4 |
| `torch_empty_cache_steps` | 1（首轮） |
| `ddp_find_unused_parameters` | false |
| `average_tokens_across_devices` | true |
| seeds | 42 / 43 / 44 |

选择 `beta=0.0` 是为了不加载/计算 reference policy，并与当前项目已验证的规则 reward
路径保持一致。side-effect、错误工具和回归通过 hard reward cap 与外部评测门控制。若开发集
出现明确 policy drift，只能预注册一个 `beta=0.001` 的独立对照；不得读取 sealed V5 后
临时调 beta。

`mask_truncated_completions=false` 是因为格式 reward 已对打满上限给零分，错误完成需要产生
负 advantage；同时 clipped ratio 必须在训练前降到 1% 以下。若大量 truncation，训练门直接
失败，不依靠该参数掩盖数据/停止问题。

### 7.3 fp16 与 V100 专项设置

固定：

```text
fp16 = true
bf16 = false
attn_implementation = sdpa
gradient_checkpointing = true
gradient_checkpointing_kwargs = {use_reentrant: false}
torch_compile = false
torch.backends.cudnn.enabled = false
GradScaler.init_scale = 1.0
GradScaler.growth_interval = 100000
PYTORCH_CUDA_ALLOC_CONF = expandable_segments:True
```

不能只依赖 Accelerate 默认 scale=65536。既有 4,177-token backward 审计表明高 scale
产生非有限梯度，而 scale=1 时 304/304 gradient tensors finite。建议创建一个很小的
`V100GRPOTrainer` 子类，在 `_build_accelerator_args` 返回值中加入：

```python
GradScalerKwargs(init_scale=1.0, growth_interval=100000)
```

并增加 `on_pre_optimizer_step` finite-gradient hard gate：只要任一 rank 出现 NaN/Inf，
立即停止并保存诊断，不允许把 scaler 自动 skip 当作成功训练。

关闭 cuDNN 是为了规避本机 V100 上 Qwen3.5 depthwise fp16 Conv1d 的
`CUDNN_STATUS_NOT_INITIALIZED`。这项设置必须由新入口显式执行并写入 run record，不能
依赖旧模块的隐式副作用。

## 8. 正式训练前门禁

### G0：版本与静态契约（已通过）

- exact package versions 与 `pip check`；
- 训练环境中 `find_spec("vllm") is None`；
- Qwen3.5 config/tokenizer/GRPOTrainer import；
- `AutoModelForCausalLM -> Qwen3_5ForCausalLM`；
- EOS/PAD 精确映射为第 5.2 节的 248046/248044；
- native non-thinking template tail；
- TRL 参数签名包含本方案使用的字段。

正式环境、90-package freeze、项目软链接和机器可读 G0 清单均已创建；所有子项通过。

### G1：真实权重 text-only load

- 单卡 fp16 加载成功；
- 模型类和 missing/unexpected keys 满足第 5.1 节；
- 不加载视觉塔；
- forward 使用 32、512、4096-token 三档输入均 finite；
- 记录 allocated/reserved memory 和 tokens/s。

### G2：LoRA 与 finite-gradient

- 152 个 LoRA modules；
- trainable params 精确为 14,376,960，或对版本差异给出逐模块解释后重新冻结；
- worst-case prompt + 320-token completion，fp16 autocast、scale=1；
- 所有 gradient tensors finite；
- 不执行 optimizer step；
- 单 rank peak allocated 建议不超过 28 GiB，峰值时至少保留 2 GiB 可用显存。

### G3：4/8-rank generation 分发证明

在 rank 日志中记录 `physical_gpu_id / global_rank / prompt_group_id / completion_index`，断言：

- 每个 prompt 恰好四条 completion；
- 四条位于四个不同 rank；
- 每个 rank 的 local generation batch 恰好为 1；
- 两个 group 的 reward gather/reshape 顺序正确；
- 不同 rank 使用不同的 device-specific RNG seed/state；若强策略碰巧生成相同 token 序列，记录
  warning，但只有 RNG seed/state 相同才判 G3 失败。

这是本次 OOM 修复的核心证据，不能只根据配置文件推断。

### G4：全 GRPO 无更新显存门

使用最长合法 train prompt，跑完整：

```text
DDP load -> generate G=4 -> reward -> policy logprob -> GRPO loss -> backward
```

不执行 optimizer step、不保存权重。要求：

- 本次记录中的全部 4 或 8 rank 通过；
- 无 OOM、NaN/Inf、空 completion、API fallback；
- max allocated <= 28 GiB，或至少保留 2 GiB 实测 headroom；
- completion clipped ratio <= 1%；
- 各阶段显存峰值单独记录，而不是只记录进程末尾值。

### G5：短 canary 与跨 step-67 soak

1. 先做 5 optimizer-step throwaway canary；
2. 再做 seed42 最多 100-step screen，必须越过历史失败点 67；
3. 每 step 记录 reward mean/std、zero-std group rate、clip ratio、gradient finite、
   allocated/reserved memory、step time；
4. checkpoint 25/50/75/100；
5. 任一 nonfinite、显存持续爬升、错误 side-effect 激增即停止；
6. seed42 只可在 disjoint development 集选择固定 checkpoint step；seed43/44 必须复现同一步，
   不允许每个 seed 各自挑最好点。

任何门失败都不进入正式三 seed 训练。

## 9. 数据与 GRPO support 门

### 9.1 数据隔离

- V5 confirmation 永久 sealed，不用于 SFT、support audit、GRPO 或 checkpoint selection；
- 新训练池使用不同 case ID、entity、query、模板、错误别名和 fixture family；
- 245 只作 legacy regression，不参与训练或选择；
- V5 calibration 可用于数据集设计诊断，但不把逐题 prompt/gold 复制进训练；
- overlap manifest 同时扫描 normalized query、实体、模板和 fixture，不只比较 case ID。

### 9.2 先决定是否需要 SFT initializer

对新训练池的 `grpo_target_step=2`，冻结 80-case 分层 support pool：8 个 scenario 各 10 条，
每类 migrate/retry 精确为 6/4，共覆盖全部 60 个训练实体且单实体最多出现两次。使用 base
Qwen3.5-4B、`temperature=0.7`、每题 8 次采样；这与仓库此前 48-case primary-step support
gate 的抽样做法一致，同时保留更大的题族和实体覆盖。不得根据采样结果换题。由于既有 V5
结果表明 retry 是 4B 的 100% anchor、migrate 才是主要缺口，以下方差/多样性门只应用于
48 条 migrate target；32 条 retry 独立要求 exact-action support >= 0.95，不对 anchor 施加
“不得饱和”的矛盾要求：

| 指标 | 门槛 |
|---|---:|
| nonzero reward std rate | >= 0.80 |
| usable support rate | >= 0.80 |
| exact action support rate | >= 0.80 |
| mean distinct valid actions | >= 1.40 |
| fully saturated rate | <= 0.25 |

全池另要求 completion clipped rate <= 0.01。

若通过，直接从 base 做 GRPO，control 为同一 base initializer。若失败，只能用完全 disjoint
数据做最短 SFT warm-up，再冻结一个 initializer；`initializer-only` 与
`same initializer + GRPO` 才构成因果对照。不得用 sealed V5 bad case 做 SFT 热身。

### 9.2.1 2026-07-15 实测结果：未通过

80-case × 8 完整审计已按冻结 pool 和 gate spec 完成，未换题。通过项为 migrate
nonzero-std、usable-support 和 exact-action support；失败项为：

| 失败项 | 实测 | 门槛 |
|---|---:|---:|
| migrate mean distinct valid actions | 1.0000 | >= 1.40 |
| migrate fully saturated rate | 0.6875 | <= 0.25 |
| retry exact-action support rate | 0.5625 | >= 0.95 |
| completion clipped rate | 0.0390625 | <= 0.01 |

实测说明 migrate 是已饱和保持类，retry 才是真正的路由边界，与本节在采样前根据
旧评测做出的 target/anchor 假设相反。不允许在看到结果后交换标签或放松阈值后把同一次
审计改判为通过。详细证据见 `SUPPORT_GATE_RESULT_20260715.md`。

### 9.3 Reward 契约

保持 V5 的 route-dominant 结构：

- action match 为主项；
- wrong action cap；
- argument type/required args；
- `finish_after_tool`；
- forbidden action；
- no premature stop / no repeated detector / no skip probe；
- retry anchor 与 migrate target 同时存在，禁止学成 always-migrate。

外层 reward 聚合第一版固定：

```text
task_reward_weight = 0.95
format_reward_weight = 0.05
score_first_json_only = true
penalize_truncated_completions = true
prefix_penalty_tokens = 16
tail_penalty_tokens = 64
```

4B 当前主要缺口是 route 而不是 JSON 可解析性，因此 format reward 只保留 5% 作为输出契约，
不能让格式项压过 observation-conditioned action。

训练日志必须同时报告 task reward、format reward、route exact、argument exact、stop exact 和
forbidden-action count，不能只报告一个合成均值。

## 10. 证明 GRPO 价值的实验设计

35B 对 4B 的差距只能证明存在 headroom，不能证明 GRPO 有效。正式因果对照为：

| Arm | 初始化 | 后续训练 |
|---|---|---|
| Control | 固定 base 或固定 SFT initializer | 无 GRPO |
| GRPO seed42 | 与 Control 完全相同 | 本方案 GRPO |
| GRPO seed43 | 与 Control 完全相同 | 同参数，仅 seed 不同 |
| GRPO seed44 | 与 Control 完全相同 | 同参数，仅 seed 不同 |

评测顺序：

1. 训练前补齐 Qwen3.5-4B 在 V5 的 3x baseline；
2. 使用 disjoint development 集选择固定 checkpoint step；
3. Control 与三个 GRPO seed 均跑 V5 3x；
4. 主指标为 exact action route，secondary 为 strict trajectory；
5. 再跑 compound245 3x 和 side-effect guardrail；
6. 不以 V5 confirmation 结果回调超参或补 seed。

建议预注册两级目标：

- **GRPO value gate**：三 seed 平均 route 相对 Control 至少 +10pp、cluster bootstrap
  95% CI 下界 > 0，至少 2/3 seed 为正，retry anchor 无超过 2pp 的实质回退；
- **工程 promotion gate**：V5 route >= 95%、strict >= 90%，245 无超过 2pp 的实质回退，
  wrong side-effect action 不增加，并通过单 V100 部署基准。

## 11. 审计通过后的工程实现状态

以下改动已实施到 G0–G2、support audit 和 dry-run；G3–G5 因 support gate 失败而
按规则未启动：

1. 新建独立、锁版本的 qwen35 GRPO 环境和 manifest；
2. 新建 Qwen3.5 专用训练入口，不把模型特例继续塞入 Qwen2.5 脚本；
3. 抽取共用 reward/data 逻辑，保持旧 Qwen2.5 入口可复现；
4. 实现 `V100GRPOTrainer` 的 low-scale GradScaler 与 finite-gradient hard gate；
5. 实现 native non-thinking step-data builder 和模板/hash 审计；
6. 实现 hybrid LoRA coverage assert；
7. 实现 rank/group/local-generation-batch instrumentation；
8. 增加 G0–G4 的无更新 smoke 命令与机器可读报告；
9. 新训练启动脚本默认只 dry-run，必须显式 `--confirm-train` 才允许 optimizer step；
10. 固定 config、dataset hash、git commit、GPU mapping 和 package lock 到 run record。

## 12. 失败后的调整顺序

如果 G4 仍 OOM，按以下顺序处理，禁止无记录地同时改多个变量：

1. 先验证每 rank local generation batch 是否确为 1；
2. 检查 prompt 是否违反 4608 token 门及是否意外加载 vision model；
3. 检查 generation cache、completion 是否在 320 内正常 EOS；
4. 记录 OOM 所在阶段和每 rank peak，排除 CUDA allocator 持续爬升；
5. 若 320 仍有截断，先复核 schema/输出冗余；只有自然 EOS 分布支持时才改单一长度变量；
6. 若训练模型与 generation 共存仍是唯一瓶颈，再设计独立 Transformers rollout worker；
7. 只有 Qwen3.5/vLLM/V100 的独立兼容 smoke 通过后，才评估 TRL vLLM server mode；
8. 不回到 full-parameter FSDP，不把 G=2 当作默认永久方案。

## 13. 待用户审计的明确决策点

1. 是否同意首版采用 8卡 DDP LoRA + 原生 Transformers generation，而非外置 vLLM；
2. 是否同意稳定版本组合 `torch 2.6.0+cu124 / transformers 5.12.0 / trl 1.8.0`；
3. 是否同意 `G=4 / generation_batch=8 / per-device=1 / grad-accum=4`；
4. 是否同意 hybrid LoRA 152 modules、r16/alpha32/dropout0；
5. prompt 4608 硬门和 completion 320（128 已被真实截断证据否证）；
6. 是否同意第一版 `beta=0 / dr_grpo / group reward scaling / lr=5e-6`；
7. 是否同意先通过 G0–G4，再允许任何 optimizer step；
8. 是否同意 seed42 100-step screen 跨越历史 step 67 后，才扩展到三 seed；
9. 是否同意 V5 route >=95%、strict >=90% 作为工程 promotion 门。

## 14. 依据

- 本项目 Qwen3.5-4B 兼容性审计：
  `reports/QWEN35_4B_MULTISTEP_V2_EVAL_AND_GRPO_COMPAT_20260714.md`
- 历史单卡 LoRA OOM：
  `experiments/runs/2026-07-07_4b_lora_grpo_focused/`
- 历史 full-parameter FSDP recovery：
  `experiments/runs/2026-07-08_4b_fullparam_grpo_fsdp_RECOVERY_CHAIN.md`
- V5 sealed qualification：
  `experiments/studies/planner_multistep_tool_routing_grpo_qwen35_4b_v1/GRPO_VALUE_EVAL_V5_20260715.md`
- [Qwen3.5-4B 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-4B)
- [Transformers 5.12.0 Qwen3.5 实现](https://github.com/huggingface/transformers/tree/v5.12.0/src/transformers/models/qwen3_5)
- [TRL 1.8.0 GRPO 文档](https://huggingface.co/docs/trl/v1.8.0/grpo_trainer)
- [vLLM 0.16.0 包元数据](https://pypi.org/project/vllm/0.16.0/)
- [vLLM 0.20.0 包元数据](https://pypi.org/project/vllm/0.20.0/)

---

Training status: **blocked by frozen support gate**.
Optimizer steps in this proposal turn: **0**.
